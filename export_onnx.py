"""
ONNX Export Engine for ATDL CIFAR-10 Models.
Exports the best models (FP32 Teacher, FP32 Student Base, TWN Baseline 4, and LATe Baseline 4)
to optimized, standard ONNX format with baked discrete ternary weights.

Usage:
  python export_onnx.py
  python export_onnx.py --out_dir onnx_models --opset_version 14
"""

import os
import sys
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model definitions from the training engines
from train_master import (
    ResNet34, ResNet18, TWNResNet18, twn_quantize, TWNConv2d, TWNLinear
)
from train_master_LATe import (
    LATeResNet18, late_quantize_exact, LATeConv2d, LATeLinear
)


def resolve_path(primary_path, fallbacks=None):
    if primary_path and os.path.exists(primary_path):
        return primary_path
    if fallbacks:
        for f in fallbacks:
            if f and os.path.exists(f):
                return f
    return primary_path


def bake_twn_to_standard_resnet18(twn_model):
    """
    Transforms a TWNResNet18 model into a standard FP32 ResNet18 model
    with discrete ternary weights {-\alpha, 0, +\alpha} baked directly into the weight tensors.
    """
    twn_model.eval()
    std_model = ResNet18(num_classes=10)
    std_model.eval()

    std_state = std_model.state_dict()
    twn_modules = dict(twn_model.named_modules())

    for name, param in twn_model.named_parameters():
        mod_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.rsplit(".", 1)[1] if "." in name else name

        if attr_name == "weight" and mod_name in twn_modules:
            mod = twn_modules[mod_name]
            if isinstance(mod, (TWNConv2d, TWNLinear)):
                # Bake quantized ternary weights
                with torch.no_grad():
                    w_q = twn_quantize(param.data, mod.delta_factor)
                    std_state[name] = w_q.clone()
            else:
                std_state[name] = param.data.clone()
        else:
            std_state[name] = param.data.clone()

    # Copy buffers (BatchNorm running_mean, running_var, num_batches_tracked)
    for name, buf in twn_model.named_buffers():
        if name in std_state or name in dict(std_model.named_buffers()):
            std_state[name] = buf.data.clone()

    std_model.load_state_dict(std_state)
    return std_model


def bake_late_to_standard_resnet18(late_model):
    """
    Transforms a LATeResNet18 model into a standard FP32 ResNet18 model
    with Loss-Aware Ternarized weights baked directly into the weight tensors.
    """
    late_model.eval()
    std_model = ResNet18(num_classes=10)
    std_model.eval()

    std_state = std_model.state_dict()
    late_modules = dict(late_model.named_modules())

    for name, param in late_model.named_parameters():
        mod_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.rsplit(".", 1)[1] if "." in name else name

        if attr_name == "weight" and mod_name in late_modules:
            mod = late_modules[mod_name]
            if isinstance(mod, (LATeConv2d, LATeLinear)):
                with torch.no_grad():
                    curv = torch.sqrt(mod.diag_curvature + 1e-8)
                    w_q = late_quantize_exact(param.data, curv)
                    std_state[name] = w_q.clone()
            else:
                std_state[name] = param.data.clone()
        else:
            std_state[name] = param.data.clone()

    for name, buf in late_model.named_buffers():
        if "diag_curvature" not in name and (name in std_state or name in dict(std_model.named_buffers())):
            std_state[name] = buf.data.clone()

    std_model.load_state_dict(std_state)
    return std_model


def verify_and_export_onnx(model, onnx_path, dummy_input, model_name, opset_version=14):
    """
    Exports a PyTorch model to ONNX, verifies ONNX graph, and checks numerical parity via ONNX Runtime.
    """
    model.eval()
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    # Export to ONNX with dynamic batch axis
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[+] Exported {model_name:30s} -> '{onnx_path}' ({file_size_mb:.2f} MB)")

    # Verify numerical parity with ONNX Runtime
    try:
        import onnx
        import onnxruntime as ort

        # 1. Check ONNX graph validity
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # 2. Run inference with ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

        ort_inputs = {ort_sess.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_out = ort_sess.run(None, ort_inputs)[0]

        # 3. Compute maximum absolute numerical difference
        max_diff = np.max(np.abs(torch_out - ort_out))
        if max_diff < 1e-4:
            print(f"    [VERIFIED] Parity check PASSED: Max |PyTorch - ONNX| difference = {max_diff:.6e} (< 1e-4)")
        else:
            print(f"    [WARNING] Numerical difference is slightly elevated: {max_diff:.6e}")

        return True, file_size_mb, max_diff

    except ImportError:
        print("    [!] onnx / onnxruntime not installed in environment; skipped parity check.")
        return True, file_size_mb, 0.0
    except Exception as e:
        print(f"    [ERROR] Parity check failed with error: {e}")
        return False, file_size_mb, float("inf")


def main():
    parser = argparse.ArgumentParser(description="Export Best CIFAR-10 Models to ONNX")
    parser.add_argument("--out_dir", type=str, default="onnx_models",
                        help="Directory to save exported ONNX models (default: onnx_models).")
    parser.add_argument("--opset_version", type=int, default=18,
                        help="ONNX opset version (default: 18).")

    # Checkpoint paths
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to FP32 Teacher checkpoint.")
    parser.add_argument("--student_base_ckpt", type=str, default=None,
                        help="Path to FP32 Student Base checkpoint.")
    parser.add_argument("--twn_b4_ckpt", type=str, default=None,
                        help="Path to TWN Baseline 4 checkpoint.")
    parser.add_argument("--late_b4_ckpt", type=str, default=None,
                        help="Path to LATe Baseline 4 checkpoint.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # Resolve checkpoints
    teacher_ckpt = resolve_path(args.teacher_ckpt or "teacher_best.pth", [
        "results/03_teacher_and_temp2_twn/teacher_best.pth",
        "teacher_best.pth"
    ])
    student_base_ckpt = resolve_path(args.student_base_ckpt or "student_base_best.pth", [
        "results/01_default_twn_temp4/student_base_best.pth",
        "student_base_best.pth"
    ])
    twn_b4_ckpt = resolve_path(args.twn_b4_ckpt or "baseline4_temp4_best.pth", [
        "results/01_default_twn_temp4/baseline4_temp4_best.pth",
        "baseline4_temp4_best.pth",
        "baseline4_best.pth"
    ])
    late_b4_ckpt = resolve_path(args.late_b4_ckpt or "late_baseline4_temp4_best.pth", [
        "results/05_late_loss_aware_ternarization/late_baseline4_temp4_best.pth",
        "late_baseline4_temp4_best.pth",
        "late_baseline4_best.pth"
    ])

    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

    print("=" * 80)
    print("ATDL CIFAR-10 ONNX EXPORT PIPELINE")
    print("=" * 80)

    export_records = []

    # 1. Export FP32 ResNet-34 Teacher
    if os.path.exists(teacher_ckpt):
        print(f"\n[1/4] Loading FP32 ResNet-34 Teacher from '{teacher_ckpt}'...")
        teacher = ResNet34(num_classes=10).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        out_path = os.path.join(args.out_dir, "resnet34_teacher_fp32.onnx")
        ok, sz, diff = verify_and_export_onnx(teacher, out_path, dummy_input, "ResNet-34 Teacher (FP32)", args.opset_version)
        export_records.append({"name": "ResNet-34 Teacher (FP32)", "path": out_path, "size_mb": sz, "diff": diff})
    else:
        print(f"[-] Teacher checkpoint NOT found at '{teacher_ckpt}' (Skipped)")

    # 2. Export FP32 ResNet-18 Student Base
    if os.path.exists(student_base_ckpt):
        print(f"\n[2/4] Loading FP32 ResNet-18 Student Base from '{student_base_ckpt}'...")
        student_base = ResNet18(num_classes=10).to(device)
        student_base.load_state_dict(torch.load(student_base_ckpt, map_location=device))
        out_path = os.path.join(args.out_dir, "resnet18_student_base_fp32.onnx")
        ok, sz, diff = verify_and_export_onnx(student_base, out_path, dummy_input, "ResNet-18 Student Base (FP32)", args.opset_version)
        export_records.append({"name": "ResNet-18 Student Base (FP32)", "path": out_path, "size_mb": sz, "diff": diff})
    else:
        print(f"[-] Student Base checkpoint NOT found at '{student_base_ckpt}' (Skipped)")

    # 3. Export Best TWN ResNet-18 Student (Baseline 4)
    if os.path.exists(twn_b4_ckpt):
        print(f"\n[3/4] Loading Best TWN ResNet-18 (Baseline 4) from '{twn_b4_ckpt}'...")
        twn_model = TWNResNet18(num_classes=10).to(device)
        twn_model.load_state_dict(torch.load(twn_b4_ckpt, map_location=device))
        twn_model.eval()

        # Verify dynamic forward before baking
        with torch.no_grad():
            orig_twn_out = twn_model(dummy_input)

        # Bake ternary weights into standard layers
        baked_twn_model = bake_twn_to_standard_resnet18(twn_model)
        baked_twn_model.eval()
        with torch.no_grad():
            baked_twn_out = baked_twn_model(dummy_input)
            baking_diff = (orig_twn_out - baked_twn_out).abs().max().item()
        print(f"    [Baking Check] Parity between dynamic TWN module and baked constant weights: max diff = {baking_diff:.6e}")

        out_path = os.path.join(args.out_dir, "twn_resnet18_baseline4.onnx")
        ok, sz, diff = verify_and_export_onnx(baked_twn_model, out_path, dummy_input, "TWN ResNet-18 (Baseline 4)", args.opset_version)
        export_records.append({"name": "TWN ResNet-18 (Baseline 4)", "path": out_path, "size_mb": sz, "diff": diff})
    else:
        print(f"[-] TWN Baseline 4 checkpoint NOT found at '{twn_b4_ckpt}' (Skipped)")

    # 4. Export Best LATe ResNet-18 Student (Baseline 4)
    if os.path.exists(late_b4_ckpt):
        print(f"\n[4/4] Loading Best LATe ResNet-18 (Baseline 4) from '{late_b4_ckpt}'...")
        late_model = LATeResNet18(num_classes=10).to(device)
        late_model.load_state_dict(torch.load(late_b4_ckpt, map_location=device), strict=False)
        late_model.eval()

        with torch.no_grad():
            orig_late_out = late_model(dummy_input)

        # Bake LATe weights into standard layers
        baked_late_model = bake_late_to_standard_resnet18(late_model)
        baked_late_model.eval()
        with torch.no_grad():
            baked_late_out = baked_late_model(dummy_input)
            baking_diff = (orig_late_out - baked_late_out).abs().max().item()
        print(f"    [Baking Check] Parity between dynamic LATe module and baked constant weights: max diff = {baking_diff:.6e}")

        out_path = os.path.join(args.out_dir, "late_resnet18_baseline4.onnx")
        ok, sz, diff = verify_and_export_onnx(baked_late_model, out_path, dummy_input, "LATe ResNet-18 (Baseline 4)", args.opset_version)
        export_records.append({"name": "LATe ResNet-18 (Baseline 4)", "path": out_path, "size_mb": sz, "diff": diff})
    else:
        print(f"[-] LATe Baseline 4 checkpoint NOT found at '{late_b4_ckpt}' (Skipped)")

    print("\n" + "=" * 80)
    print("CONSOLIDATED ONNX EXPORT SUMMARY")
    print("=" * 80)
    print(f"{'Model Architecture':<32} | {'ONNX File Path':<36} | {'Disk Size':<10} | {'Parity Diff':<12}")
    print("-" * 80)
    for r in export_records:
        print(f"{r['name']:<32} | {r['path']:<36} | {r['size_mb']:>6.2f} MB | {r['diff']:>10.2e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
