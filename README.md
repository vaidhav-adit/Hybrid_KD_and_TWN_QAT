---
title: TWN Quantization Demo
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Knowledge Distillation with Ternary-Weight Quantization-Aware Training (KD-TWN & LATe)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20%7C%20INT2-005CED.svg)](https://onnxruntime.ai/)
[![Gradio App](https://img.shields.io/badge/Interactive%20Demo-Gradio%20%7C%20Spaces-FF5722.svg)](#interactive-web-demo-100-free)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Report PDF](https://img.shields.io/badge/Paper-Report%20PDF-red.svg)](ATDL_Assignment1_Vaidhav.pdf)

This repository contains the complete implementation, experimental pipelines, evaluation suite, and research report for combining **Knowledge Distillation (KD)** with **Ternary-Weight Quantization-Aware Training (TWN-QAT)** and **Loss-Aware Ternarization (LATe)** on CIFAR-10 and CIFAR-10.1.

- **Teacher Architecture**: Full-precision FP32 ResNet-34 (21.28M parameters, 81.33 MB ONNX, **96.54%** CIFAR-10 Test Accuracy).
- **Student Architecture**: ResNet-18 (11.17M parameters, weights constrained to symmetric ternary states $\{-\alpha, 0, +\alpha\}$).
- **Headline Result**: Joint KD + TWN-QAT (Baseline 4) achieves **96.15%** CIFAR-10 test accuracy (within 0.06 pp of the uncompressed 96.21% FP32 Student Base) while reducing physical model storage from 42.70 MB to **2.80 MB** in native ONNX 2-bit format (**15.3x compression**, 52.00% sparsity).
- **Zero Generalization Gap**: On the unseen **CIFAR-10.1** benchmark, TWN Baseline 4 achieves **90.95% Top-1 accuracy**, outperforming both the FP32 Student Base (89.75%) and ResNet-34 Teacher (90.80%).
- **Loss-Aware Extension**: LATe Baseline 4 achieves **95.91%** CIFAR-10 test accuracy with **66.85%** overall sparsity.

---

## 🚀 Interactive Web Demo (100% Free)

You can launch and test the interactive web application locally or deploy it to **Hugging Face Spaces for $0**:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Gradio web demo locally
python app.py
# Open http://127.0.0.1:7860 in your browser
```

### Features:
- **Side-by-Side Model Comparison**: Simultaneously runs an input test image through **ResNet-34 Teacher**, **ResNet-18 Student Base**, **TWN ResNet-18 (INT2)**, and **LATe ResNet-18 (INT2)**.
- **Real-Time Benchmarking**: Live single-sample CPU latency (ms), throughput (FPS), confidence breakdown, disk footprint, and parameter sparsity.
- **Preloaded Unseen CIFAR-10.1 Gallery**: 10 curated test samples across all classes, plus arbitrary image upload / webcam drag-and-drop.

---

## 📊 ONNX Deployment & Unseen CIFAR-10.1 Benchmark

All 4 models evaluated sequentially across the full unseen **CIFAR-10.1 benchmark (2,000 images)** on single-thread CPU (Batch Size = 1):

| Model Architecture | Format & Storage | Disk Size (MB) | Compression Ratio | CIFAR-10 Acc (%) | CIFAR-10.1 Top-1 (%) | Macro F1 (%) | CPU Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34 Teacher** | FP32 ONNX | 81.33 MB | $1.0\times$ (Ref) | **96.54%** | 90.80% | 90.80% | 28.07 ms | 35.6 FPS |
| **ResNet-18 Student Base** | FP32 ONNX | 42.70 MB | $1.0\times$ (Base) | 96.21% | 89.75% | 89.75% | 13.74 ms | 72.8 FPS |
| **TWN ResNet-18 (B4)** | **Native INT2 ONNX** | **2.80 MB** | **15.3x (93.4% saved)** | **96.15%** | **90.95% (Highest!)** | **90.95%** | **13.10 ms** | **76.3 FPS** |
| **LATe ResNet-18 (B4)** | **Native INT2 ONNX** | **2.80 MB** | **15.3x (93.4% saved)** | 95.91% | 89.85% | 89.84% | **13.17 ms** | **75.9 FPS** |

To reproduce the exact ONNX benchmark:
```bash
python eval_unseen_onnx.py
```

---

## Repository Structure

```
├── app.py                           # Interactive Gradio web application for 4-model comparison
├── ATDL_Assignment1_Vaidhav.pdf     # Compiled IEEE-format research report
├── report.tex                       # Complete LaTeX source of the report
├── requirements.txt                 # Python package dependencies
├── export_onnx.py                   # Standard FP32 ONNX export engine with baked ternary weights
├── export_onnx_int2.py              # Native 2-Bit (TensorProto.INT2) ONNX packing engine (2.80 MB)
├── eval_unseen_onnx.py              # Pure INT2 ONNX unseen CIFAR-10.1 benchmarking suite
├── train_master.py                  # Master training engine (Teacher, FP32, TWN B1–B4, Ablations)
├── train_master_LATe.py             # Loss-Aware Ternarization (LATe) training engine
├── eval_unseen_cifar10_1.py         # PyTorch CIFAR-10.1 evaluation and confusion matrix generator
├── onnx_models/                     # Exported native INT2 (2.80 MB) and FP32 ONNX artifacts
├── sample_images/                   # Preloaded test images for each CIFAR-10 category
└── results/                         # Empirical logs, checkpoints (.pth), learning curves, and histograms
    ├── 00_reference_models/         # Pretrained Teacher & Student Base checkpoints
    ├── 01_default_twn_temp4/        # Default TWN baselines (B1, B2, B3, B4 at T=4)
    ├── 02_ablation1a_temp2/         # Temperature ablation (T=2)
    ├── 03_ablation1b_temp15/        # Temperature ablation (T=15)
    ├── 04_ablation2_first_last/     # Boundary layer precision ablation (FNL full-network ternarization)
    └── 05_late_loss_aware_ternarization/ # Loss-Aware Ternarization runs (B1–B4)
```

---

## 🛠️ Step-by-Step Free Cloud Deployment (Hugging Face Spaces)

To deploy your interactive demo on Hugging Face Spaces for **$0 / free forever**:

1. **Create a Free Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name (e.g. `twn-quantization-demo`).
   - Select **SDK: Gradio** and **Hardware: Free (2 vCPU, 16 GB RAM)**.
2. **Push Code to Space**:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/twn-quantization-demo
   git push space main
   ```
3. Your web demo will build and be live at `https://huggingface.co/spaces/<your-username>/twn-quantization-demo`!

---

## Reproduction Commands

### 1. Train Reference Models (FP32 Teacher & Student Base)

```bash
# Train ResNet-34 Teacher (200 epochs, SGD, Cosine Annealing)
python train_master.py --mode teacher --epochs 200 --batch_size 128 --base_lr 0.1

# Train ResNet-18 Student Base (FP32 reference without quantization or KD)
python train_master.py --mode student_base --epochs 200 --batch_size 128 --base_lr 0.1
```

### 2. Train Primary Baselines (Default $T=4.0$, $\lambda=0.5$)

```bash
# Baseline 1: TWN Supervised Scratch (No KD)
python train_master.py --mode baseline1 --epochs 200

# Baseline 2: Two-Stage (100 ep FP32 KD Pretrain -> 100 ep TWN-QAT)
python train_master.py --mode baseline2 --teacher_ckpt results/00_reference_models/teacher_best.pth --kd_temperature 4.0

# Baseline 3: Direct Hard Discrete Projection (No latent FP32 weights)
python train_master.py --mode baseline3 --teacher_ckpt results/00_reference_models/teacher_best.pth --kd_temperature 4.0

# Baseline 4: Joint KD + TWN-QAT from scratch (Best Performing)
python train_master.py --mode baseline4 --teacher_ckpt results/00_reference_models/teacher_best.pth --kd_temperature 4.0
```

### 3. Ablation Studies

```bash
# Temperature Ablation 1a: Low Temperature (T=2.0, T^2=4)
python train_master.py --mode baselines_all --kd_temperature 2.0 --teacher_ckpt results/00_reference_models/teacher_best.pth

# Temperature Ablation 1b: High Temperature (T=15.0, T^2=225)
python train_master.py --mode baselines_all --kd_temperature 15.0 --teacher_ckpt results/00_reference_models/teacher_best.pth

# Boundary Layer Precision Ablation: Full-Network Ternarization (FNL)
python train_master.py --mode baselines_all --kd_temperature 4.0 --quantize_first_last --teacher_ckpt results/00_reference_models/teacher_best.pth
```

### 4. Loss-Aware Ternarization (LATe) Experiments

```bash
# Run all LATe baselines (B1 to B4) with sensitivity-weighted thresholding
python train_master_LATe.py --mode baselines_all --kd_temperature 4.0 --teacher_ckpt results/00_reference_models/teacher_best.pth

# Run individual LATe Baseline 4:
python train_master_LATe.py --mode baseline4 --kd_temperature 4.0 --teacher_ckpt results/00_reference_models/teacher_best.pth
```

### 5. Export Native 2-Bit INT2 ONNX Models

```bash
# Export standard ONNX models with baked ternary weights
python export_onnx.py

# Convert and pack into native 2-bit INT2 ONNX models (2.80 MB each)
python export_onnx_int2.py
```

---

## Citation & References

```bibtex
@article{adit2026kdtwn,
  title={Knowledge Distillation with Ternary Weight Quantization-Aware Training: An Experimental Study of Training Strategies},
  author={Adit, Vaidhav},
  year={2026}
}
```
