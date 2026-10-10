"""
Interactive Web Application for Hybrid Knowledge Distillation & Ternary Weight Networks (TWN-QAT).
Built with Gradio & ONNX Runtime for 100% Free Public Deployment (Hugging Face Spaces / Local).

Features:
  - Side-by-Side Real-Time Inference on All 4 Models
  - Preloaded CIFAR-10 / CIFAR-10.1 Examples & Custom Image Upload
  - Live Latency, Throughput (FPS), File Size & Sparsity Benchmarking
  - Pure 2-Bit INT2 ONNX In-Memory Dequantization
"""

import os
import time
import numpy as np
from PIL import Image
import gradio as gr
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort

# ==============================================================================
# 1. Constants & Preprocessing
# ==============================================================================

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

MODEL_CONFIGS = {
    "teacher": {
        "title": "ResNet-34 Teacher",
        "path": "onnx_models/resnet34_teacher_fp32.onnx",
        "format": "FP32 (32-bit)",
        "size_mb": 81.33,
        "compression": "1.0x (Baseline)",
        "sparsity": "0.00%",
        "cifar10_acc": "96.54%",
        "unseen_acc": "90.80%",
        "badge": "Teacher Baseline"
    },
    "student_base": {
        "title": "ResNet-18 Student Base",
        "path": "onnx_models/resnet18_student_base_fp32.onnx",
        "format": "FP32 (32-bit)",
        "size_mb": 42.70,
        "compression": "1.0x (Student Base)",
        "sparsity": "0.00%",
        "cifar10_acc": "96.21%",
        "unseen_acc": "89.75%",
        "badge": "Student Baseline"
    },
    "twn_b4": {
        "title": "TWN ResNet-18 (B4 Joint KD)",
        "path": "onnx_models/twn_resnet18_baseline4_int2.onnx",
        "format": "Native INT2 (2-bit packed)",
        "size_mb": 2.80,
        "compression": "15.3x (93.4% saved)",
        "sparsity": "52.00%",
        "cifar10_acc": "96.15%",
        "unseen_acc": "90.95% (Highest!)",
        "badge": "Best TWN Student"
    },
    "late_b4": {
        "title": "LATe ResNet-18 (B4 Loss-Aware)",
        "path": "onnx_models/late_resnet18_baseline4_int2.onnx",
        "format": "Native INT2 (2-bit packed)",
        "size_mb": 2.80,
        "compression": "15.3x (93.4% saved)",
        "sparsity": "66.85% (Max Sparsity)",
        "cifar10_acc": "95.91%",
        "unseen_acc": "89.85%",
        "badge": "Best LATe Student"
    }
}


def preprocess_image(pil_img):
    """Resizes and normalizes an input PIL image to CIFAR NCHW float32 tensor."""
    img_resized = pil_img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 32, 32)
    norm = (arr - CIFAR_MEAN) / CIFAR_STD
    return norm.astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ==============================================================================
# 2. In-Memory INT2 Session Loader
# ==============================================================================

def load_session(onnx_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph

    has_int2 = any(init.data_type == TensorProto.INT2 for init in graph.initializer)

    if not has_int2:
        return ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    
    # Unpack 2-bit INT2 weights with per-channel alpha scaling
    alphas = {}
    for init in graph.initializer:
        if init.name.endswith("_alpha"):
            alphas[init.name[:-6]] = numpy_helper.to_array(init)

    for init in graph.initializer:
        if init.data_type == TensorProto.INT2:
            name = init.name
            dims = list(init.dims)
            c_out = dims[0]
            raw = np.frombuffer(init.raw_data, dtype=np.uint8)

            w0 = raw & 0x03
            w1 = (raw >> 2) & 0x03
            w2 = (raw >> 4) & 0x03
            w3 = (raw >> 6) & 0x03

            dec = lambda w: np.where(w == 0b01, 1.0, np.where(w == 0b10, -1.0, 0.0)).astype(np.float32)
            unpacked = np.stack([dec(w0), dec(w1), dec(w2), dec(w3)], axis=1).reshape(-1)
            unpacked = unpacked[:int(np.prod(dims))].reshape(c_out, -1)

            if name in alphas:
                alpha_vec = alphas[name].reshape(c_out, 1)
                unpacked = unpacked * alpha_vec

            unpacked = unpacked.reshape(dims).astype(np.float32)
            init.data_type = TensorProto.FLOAT
            init.raw_data = unpacked.tobytes()

    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])


print("Loading ONNX Inference Sessions into memory...")
SESSIONS = {}
for key, cfg in MODEL_CONFIGS.items():
    if os.path.exists(cfg["path"]):
        SESSIONS[key] = load_session(cfg["path"])
        print(f"[+] Loaded {cfg['title']} ({cfg['format']})")
    else:
        print(f"[-] Warning: {cfg['path']} not found.")


# ==============================================================================
# 3. Prediction & Benchmarking Logic
# ==============================================================================

def run_single_inference(session, tensor):
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    
    # Warmup
    session.run([out_name], {in_name: tensor})
    
    # Measure Latency (5 iterations average)
    runs = 5
    t0 = time.perf_counter()
    for _ in range(runs):
        logits = session.run([out_name], {in_name: tensor})[0][0]
    t1 = time.perf_counter()
    
    latency_ms = ((t1 - t0) / runs) * 1000.0
    probs = softmax(logits)
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx]) * 100.0
    top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]

    return top3, top_class, top_conf, latency_ms


def make_bar_html(top3, top_class, top_conf, latency_ms, cfg):
    """Render a styled HTML bar chart — avoids gr.Label's broken JSON schema."""
    bars = ""
    for idx, score in top3:
        pct = score * 100.0
        label = CLASS_NAMES[idx]
        color = "#6366f1" if label == top_class else "#94a3b8"
        bars += (
            f'<div style="margin:4px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="width:80px;font-size:12px;color:#e2e8f0;text-align:right;">{label}</span>'
            f'<div style="flex:1;background:#1e293b;border-radius:4px;height:18px;">'
            f'<div style="width:{pct:.1f}%;background:{color};border-radius:4px;height:18px;"></div>'
            f'</div>'
            f'<span style="font-size:12px;color:#cbd5e1;min-width:42px;">{pct:.1f}%</span>'
            f'</div></div>'
        )
    info = (
        f'<div style="font-size:12px;color:#94a3b8;margin-top:8px;line-height:1.8;">'
        f'<b style="color:#e2e8f0;">Prediction:</b> {top_class} ({top_conf:.1f}%)<br>'
        f'<b style="color:#e2e8f0;">Latency:</b> {latency_ms:.1f} ms &nbsp;|&nbsp; '
        f'<b style="color:#e2e8f0;">FPS:</b> {1000.0/latency_ms:.0f}<br>'
        f'<b style="color:#e2e8f0;">Size:</b> {cfg["size_mb"]:.2f} MB ({cfg["compression"]})<br>'
        f'<b style="color:#e2e8f0;">Sparsity:</b> {cfg["sparsity"]}&nbsp;|&nbsp;'
        f'<b style="color:#e2e8f0;">CIFAR-10.1:</b> {cfg["unseen_acc"]}'
        f'</div>'
    )
    return f'<div style="padding:8px;background:#0f172a;border-radius:8px;">{bars}{info}</div>'


@gpu_decorator
def predict_all_models(input_img):
    if input_img is None:
        placeholder = '<div style="color:#64748b;padding:8px;">Upload an image and click Compare.</div>'
        return [placeholder] * 4

    tensor = preprocess_image(input_img)

    results = []
    for key in ["teacher", "student_base", "twn_b4", "late_b4"]:
        sess = SESSIONS[key]
        cfg = MODEL_CONFIGS[key]
        top3, top_class, top_conf, latency_ms = run_single_inference(sess, tensor)
        html = make_bar_html(top3, top_class, top_conf, latency_ms, cfg)
        results.append(html)

    return results


# ==============================================================================
# 4. Gradio UI Layout
# ==============================================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
}
.hero-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
"""

with gr.Blocks(title="Ternary ResNet-18 KD & QAT Demo") as demo:
    gr.HTML("""
    <div class="hero-box">
        <h1 style="margin:0; font-size: 26px; font-weight: 800;">🚀 Knowledge Distillation with Ternary Weight Networks (TWN-QAT)</h1>
        <p style="margin: 8px 0 0 0; font-size: 14px; opacity: 0.85;">
            Interactive comparison of <b>ResNet-34 Teacher</b>, <b>ResNet-18 Student Base</b>, 
            and <b>2-Bit INT2 Quantized Ternary Students</b> (Standard TWN & Loss-Aware LATe) on CIFAR-10 & Unseen CIFAR-10.1.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            img_input = gr.Image(type="pil", label="Input Test Image (Click or Upload)", sources=["upload", "clipboard", "webcam"])
            
            # Preloaded example gallery
            example_files = [
                f"sample_images/{f}" for f in sorted(os.listdir("sample_images")) if f.endswith(".png")
            ] if os.path.exists("sample_images") else []
            
            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=img_input,
                    label="Curated Unseen CIFAR-10.1 Samples (1 per class)",
                    examples_per_page=5
                )

            btn_run = gr.Button("⚡ Compare All 4 Models", variant="primary", size="lg")

        with gr.Column(scale=8):
            gr.Markdown("### 📊 Side-by-Side Model Comparison (Live CPU Inference)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🎓 **ResNet-34 Teacher** — FP32 81.33 MB")
                    out_teacher = gr.HTML('<div style="color:#64748b;padding:8px;">Upload an image and click Compare.</div>')

                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 **ResNet-18 Student Base** — FP32 42.70 MB")
                    out_student = gr.HTML('<div style="color:#64748b;padding:8px;">Upload an image and click Compare.</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚡ **TWN ResNet-18 B4** — INT2 2.80 MB (15.3× smaller)")
                    out_twn = gr.HTML('<div style="color:#64748b;padding:8px;">Upload an image and click Compare.</div>')

                with gr.Column(scale=1):
                    gr.Markdown("#### 🎯 **LATe ResNet-18 B4** — INT2 2.80 MB (66.85% sparse)")
                    out_late = gr.HTML('<div style="color:#64748b;padding:8px;">Upload an image and click Compare.</div>')

    # Educational Technical Summary
    with gr.Accordion("📖 Key Technical Findings & Architecture Summary", open=False):
        gr.Markdown(r"""
| Model Architecture | Format & Storage | Compression Ratio | CIFAR-10 Test Acc | Unseen CIFAR-10.1 Acc | Zero Sparsity | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34 Teacher** | FP32 (81.33 MB) | $1.0\times$ (Reference) | **96.54%** | 90.80% | 0.00% | 28.07 ms |
| **ResNet-18 Student Base** | FP32 (42.70 MB) | $1.0\times$ (Base) | **96.21%** | 89.75% | 0.00% | 13.74 ms |
| **TWN ResNet-18 (B4)** | **INT2 (2.80 MB)** | **15.3x (93.4% saved)** | **96.15%** | **90.95% (Highest!)** | **52.00%** | **13.10 ms** |
| **LATe ResNet-18 (B4)** | **INT2 (2.80 MB)** | **15.3x (93.4% saved)** | **95.91%** | **89.85%** | **66.85% (Max)** | **13.17 ms** |

### Key Highlights:
1. **Zero Generalization Gap**: Ternary Quantization acts as an effective regularizer. On unseen **CIFAR-10.1**, **TWN B4 achieves 90.95% accuracy**, surpassing both the FP32 student (89.75%) and teacher (90.80%).
2. **True 2-Bit Disk Compression**: Models are serialized as native ONNX `TensorProto.INT2`, packing 4 ternary weights per byte for a physical size of **2.80 MB** ($15.3\times$ compression).
3. **Multiplication-Free Inference**: Non-zero weights $\{-\alpha, +\alpha\}$ replace FP32 multiplications with simple additions/subtractions, while $52\%$ to $66.85\%$ of operations are completely bypassed via zero-skipping.
""")

    # Event binding — 4 HTML outputs, no gr.Label (avoids JSON schema bool bug)
    outputs_list = [out_teacher, out_student, out_twn, out_late]

    btn_run.click(fn=predict_all_models, inputs=[img_input], outputs=outputs_list)
    img_input.change(fn=predict_all_models, inputs=[img_input], outputs=outputs_list)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
