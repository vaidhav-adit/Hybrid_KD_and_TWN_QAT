"""
Comprehensive ONNX Evaluation on Unseen CIFAR-10.1 Dataset.
Directly loads and evaluates:
  1. ResNet-34 Teacher (FP32 ONNX, 81.33 MB)
  2. ResNet-18 Student Base (FP32 ONNX, 42.70 MB)
  3. TWN ResNet-18 Baseline 4 (Pure INT2 ONNX, 2.80 MB)
  4. LATe ResNet-18 Baseline 4 (Pure INT2 ONNX, 2.80 MB)

Measures:
  - Full Classification Metrics (Top-1 Acc, Top-5 Acc, Precision, Recall, F1, Loss)
  - Runtime Latency & Performance (Mean, Median, P95, P99, FPS, Total Dataset Time)
  - Disk Compression (Actual File Size on Disk, Compression Ratio vs Student Base)
"""

import os
import sys
import time
import argparse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss, confusion_matrix
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

# ==============================================================================
# 1. Constants & Dataset Loading
# ==============================================================================

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_1_DATA_URL   = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy"
CIFAR10_1_LABELS_URL = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy"


def load_cifar10_1(data_dir="./data/cifar10_1"):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "cifar10.1_v6_data.npy")
    labels_path = os.path.join(data_dir, "cifar10.1_v6_labels.npy")

    if not os.path.exists(data_path):
        print(f"Downloading CIFAR-10.1 data from {CIFAR10_1_DATA_URL}...")
        urllib.request.urlretrieve(CIFAR10_1_DATA_URL, data_path)

    if not os.path.exists(labels_path):
        print(f"Downloading CIFAR-10.1 labels from {CIFAR10_1_LABELS_URL}...")
        urllib.request.urlretrieve(CIFAR10_1_LABELS_URL, labels_path)

    raw_images = np.load(data_path)  # (2000, 32, 32, 3), uint8
    labels = np.load(labels_path)     # (2000,)

    norm_images = raw_images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    norm_images = (norm_images - CIFAR_MEAN) / CIFAR_STD

    print(f"Successfully loaded full unseen CIFAR-10.1 benchmark ({len(labels)} images across 10 classes).")
    return norm_images, labels


# ==============================================================================
# 2. INT2 ONNX In-Memory Session Loader
# ==============================================================================

def create_inference_session_from_onnx(onnx_file_path):
    """
    Creates an ONNX Runtime InferenceSession directly from the given .onnx file.
    If the file is an INT2 model, it decodes the 2-bit packed weights and per-channel
    alpha scalers directly in-memory to build the execution session.
    """
    file_size_bytes = os.path.getsize(onnx_file_path)
    data_path = onnx_file_path + ".data"
    if os.path.exists(data_path):
        file_size_bytes += os.path.getsize(data_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model = onnx.load(onnx_file_path, load_external_data=True)
    graph = model.graph

    has_int2 = any(init.data_type == TensorProto.INT2 for init in graph.initializer)

    if not has_int2:
        # Standard FP32 model
        session = ort.InferenceSession(onnx_file_path, opts, providers=["CPUExecutionProvider"])
        model_type = "FP32 (32-bit)"
    else:
        # Pure INT2 Model: decode 2-bit ternary weights + alpha scalers in-memory
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

                # Decode 0b00->0, 0b01->+1, 0b10->-1
                dec = lambda w: np.where(w == 0b01, 1.0, np.where(w == 0b10, -1.0, 0.0)).astype(np.float32)
                unpacked = np.stack([dec(w0), dec(w1), dec(w2), dec(w3)], axis=1).reshape(-1)
                total_elems = int(np.prod(dims))
                unpacked = unpacked[:total_elems].reshape(c_out, -1)

                if name in alphas:
                    alpha_vec = alphas[name].reshape(c_out, 1)
                    unpacked = unpacked * alpha_vec

                unpacked = unpacked.reshape(dims).astype(np.float32)
                init.data_type = TensorProto.FLOAT
                init.raw_data = unpacked.tobytes()

        # Build session directly from in-memory byte buffer
        session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        model_type = "INT2 (2-bit packed)"

    return session, file_size_mb, model_type


# ==============================================================================
# 3. Model Evaluator
# ==============================================================================

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def evaluate_onnx_model(model_name, onnx_path, images, labels, warmup_runs=50):
    print(f"\n================================================================================")
    print(f"Evaluating: {model_name}")
    print(f"Target ONNX File: {onnx_path}")
    print(f"================================================================================")

    session, file_size_mb, model_type = create_inference_session_from_onnx(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Model Storage Format: {model_type}")
    print(f"Actual Disk File Size: {file_size_mb:.2f} MB")

    # Warmup
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for _ in range(warmup_runs):
        session.run([output_name], {input_name: dummy})

    # Sequential Benchmark
    latencies = []
    all_logits = []
    num_samples = len(images)

    t0_total = time.perf_counter()
    for i in range(num_samples):
        sample = images[i:i+1]
        t_start = time.perf_counter()
        out = session.run([output_name], {input_name: sample})[0]
        t_end = time.perf_counter()
        
        latencies.append((t_end - t_start) * 1000.0)
        all_logits.append(out)
    t1_total = time.perf_counter()

    total_dataset_time_sec = t1_total - t0_total
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = softmax(all_logits)
    preds = np.argmax(all_logits, axis=1)

    # Classification Metrics
    top1_acc = accuracy_score(labels, preds) * 100.0
    
    top5_correct = 0
    for i in range(num_samples):
        top5_indices = np.argsort(all_logits[i])[-5:]
        if labels[i] in top5_indices:
            top5_correct += 1
    top5_acc = (top5_correct / num_samples) * 100.0

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    
    ce_loss = log_loss(labels, all_probs)
    cm = confusion_matrix(labels, preds)

    # Latency Stats
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p90_lat = np.percentile(latencies, 90)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat

    results = {
        "model_name": model_name,
        "onnx_path": onnx_path,
        "model_type": model_type,
        "file_size_mb": file_size_mb,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "prec_macro": prec_macro * 100.0,
        "rec_macro": rec_macro * 100.0,
        "f1_macro": f1_macro * 100.0,
        "prec_weighted": prec_weighted * 100.0,
        "rec_weighted": rec_weighted * 100.0,
        "f1_weighted": f1_weighted * 100.0,
        "cross_entropy": ce_loss,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "std_latency_ms": std_lat,
        "fps": fps,
        "total_dataset_time_sec": total_dataset_time_sec,
        "confusion_matrix": cm,
    }

    print(f"\n--- Classification Performance on Unseen CIFAR-10.1 ---")
    print(f"  Top-1 Accuracy:       {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy:       {top5_acc:.2f}%")
    print(f"  Macro Precision:      {prec_macro*100.0:.2f}%")
    print(f"  Macro Recall:         {rec_macro*100.0:.2f}%")
    print(f"  Macro F1-Score:       {f1_macro*100.0:.2f}%")
    print(f"  Cross-Entropy Loss:   {ce_loss:.4f}")
    
    print(f"\n--- Inference Runtime & Latency (Single-Thread CPU, Batch=1) ---")
    print(f"  Mean Latency:         {mean_lat:.2f} ms")
    print(f"  Median Latency:       {median_lat:.2f} ms")
    print(f"  95th Percentile:      {p95_lat:.2f} ms")
    print(f"  99th Percentile:      {p99_lat:.2f} ms")
    print(f"  Throughput (FPS):     {fps:.1f} frames/sec")
    print(f"  Full 2000-img Time:   {total_dataset_time_sec:.2f} s")

    print(f"\n--- Model Storage & Compression ---")
    print(f"  Actual File Size:     {file_size_mb:.2f} MB")
    if file_size_mb < 10.0:
        ratio = 42.70 / file_size_mb
        savings = (1.0 - file_size_mb / 42.70) * 100.0
        print(f"  Compression Ratio:    {ratio:.1f}x (vs FP32 Student Base)")
        print(f"  Storage Savings:      {savings:.1f}%")

    return results


# ==============================================================================
# 4. Plots & Visualizations
# ==============================================================================

def plot_comprehensive_report(all_results, output_png="onnx_unseen_evaluation_report.png"):
    names = [r["model_name"] for r in all_results]
    accs = [r["top1_acc"] for r in all_results]
    f1s = [r["f1_macro"] for r in all_results]
    lats = [r["mean_latency_ms"] for r in all_results]
    fpss = [r["fps"] for r in all_results]
    sizes = [r["file_size_mb"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]

    # 1. Top-1 Accuracy & F1
    x = np.arange(len(names))
    width = 0.35
    axes[0, 0].bar(x - width/2, accs, width, label="Top-1 Accuracy (%)", color="#2E86AB")
    axes[0, 0].bar(x + width/2, f1s, width, label="Macro F1-Score (%)", color="#A23B72")
    axes[0, 0].set_title("Classification Accuracy on Unseen CIFAR-10.1", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    axes[0, 0].set_ylim(85, 95)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    for i in range(len(names)):
        axes[0, 0].text(i - width/2, accs[i] + 0.2, f"{accs[i]:.2f}%", ha="center", fontsize=8, fontweight="bold")
        axes[0, 0].text(i + width/2, f1s[i] + 0.2, f"{f1s[i]:.2f}%", ha="center", fontsize=8)

    # 2. File Size (MB) / Compression
    axes[0, 1].bar(names, sizes, color=colors, edgecolor="black", linewidth=0.8)
    axes[0, 1].set_title("Actual Model File Size on Disk (MB)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Size (MB)")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, s in enumerate(sizes):
        ratio_tag = f"({42.70/s:.1f}x)" if s < 10 else "(1.0x)"
        axes[0, 1].text(i, s + 1.5, f"{s:.2f} MB\n{ratio_tag}", ha="center", fontsize=8, fontweight="bold")

    # 3. CPU Latency (ms)
    axes[1, 0].bar(names, lats, color="#E76F51", edgecolor="black", linewidth=0.8)
    axes[1, 0].set_title("Inference Latency per Sample (ms, CPU)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, l in enumerate(lats):
        axes[1, 0].text(i, l + 0.2, f"{l:.2f} ms", ha="center", fontsize=8, fontweight="bold")

    # 4. Throughput (FPS)
    axes[1, 1].bar(names, fpss, color="#2A9D8F", edgecolor="black", linewidth=0.8)
    axes[1, 1].set_title("Throughput (Frames / Second, CPU)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, f in enumerate(fpss):
        axes[1, 1].text(i, f + 3, f"{f:.1f} FPS", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"\n[Saved comprehensive report plot to: '{output_png}']")


def plot_confusion_matrices(all_results, output_png="onnx_unseen_confusion_matrices.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        cm = r["confusion_matrix"]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{r['model_name']}\n(Top-1 Acc: {r['top1_acc']:.2f}% | Size: {r['file_size_mb']:.2f} MB)", fontsize=11, fontweight="bold")
        
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"[Saved confusion matrices plot to: '{output_png}']")


# ==============================================================================
# 5. Main Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure ONNX Models on CIFAR-10.1")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10_1")
    args = parser.parse_args()

    images, labels = load_cifar10_1(args.data_dir)

    models_to_eval = [
        {
            "name": "ResNet-34 Teacher",
            "onnx": "onnx_models/resnet34_teacher_fp32.onnx",
        },
        {
            "name": "ResNet-18 Student Base",
            "onnx": "onnx_models/resnet18_student_base_fp32.onnx",
        },
        {
            "name": "TWN ResNet-18 (INT2 ONNX)",
            "onnx": "onnx_models/twn_resnet18_baseline4_int2.onnx",
        },
        {
            "name": "LATe ResNet-18 (INT2 ONNX)",
            "onnx": "onnx_models/late_resnet18_baseline4_int2.onnx",
        },
    ]

    all_results = []
    for m in models_to_eval:
        res = evaluate_onnx_model(
            model_name=m["name"],
            onnx_path=m["onnx"],
            images=images,
            labels=labels
        )
        all_results.append(res)

    plot_comprehensive_report(all_results)
    plot_confusion_matrices(all_results)

    print("\n" + "="*120)
    print("FINAL ONNX UNSEEN CIFAR-10.1 BENCHMARK & 2-BIT COMPRESSION SUMMARY")
    print("="*120)
    header = f"{'Model Architecture':<30} | {'Format':<14} | {'Disk Size':<10} | {'Ratio':<7} | {'Top-1 Acc':<9} | {'Top-5 Acc':<9} | {'F1-Score':<9} | {'Latency':<9} | {'FPS':<7}"
    print(header)
    print("-" * 120)
    for r in all_results:
        size_str = f"{r['file_size_mb']:.2f} MB"
        ratio_str = f"{42.70/r['file_size_mb']:.1f}x" if r['file_size_mb'] < 10 else "1.0x"
        row = f"{r['model_name']:<30} | {r['model_type']:<14} | {size_str:<10} | {ratio_str:<7} | {r['top1_acc']:>8.2f}% | {r['top5_acc']:>8.2f}% | {r['f1_macro']:>8.2f}% | {r['mean_latency_ms']:>6.2f} ms | {r['fps']:>6.1f}"
        print(row)
    print("="*120)


if __name__ == "__main__":
    main()
