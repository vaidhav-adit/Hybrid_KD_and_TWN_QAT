# Knowledge Distillation with Ternary-Weight Quantization-Aware Training (KD-TWN & LATe)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Report PDF](https://img.shields.io/badge/Paper-Report%20PDF-red.svg)](ATDL_Assignment1_Vaidhav.pdf)

This repository contains the complete implementation, experimental pipelines, evaluation suite, and research report for **ATDL Assignment 1**: combining **Knowledge Distillation (KD)** with **Ternary-Weight Quantization-Aware Training (TWN-QAT)** and **Loss-Aware Ternarization (LATe)** on CIFAR-10 and CIFAR-10.1.

- **Teacher Architecture**: Full-precision FP32 ResNet-34 (21.28M parameters, 85.27 MB, **96.54%** CIFAR-10 Test Accuracy).
- **Student Architecture**: ResNet-18 (11.17M parameters, weights constrained to symmetric ternary states $\{-\alpha, 0, +\alpha\}$).
- **Headline Result**: Joint KD + TWN-QAT (Baseline 4) achieves **96.15%** CIFAR-10 test accuracy (within 0.06 pp of the uncompressed 96.21% FP32 Student Base) while reducing reported model storage from 42.63 MB to 2.72 MB (**15.7x compression**, 52.00% sparsity).
- **Loss-Aware Extension**: LATe Baseline 4 achieves **95.91%** CIFAR-10 test accuracy with **66.85%** overall sparsity.

---

## Repository Structure

```
├── ATDL_Assignment1_Vaidhav.pdf     # Compiled IEEE-format research report
├── report.tex                       # Complete LaTeX source of the report
├── requirements.txt                 # Python package dependencies
├── train_master.py                  # Master training engine (Teacher, FP32, TWN B1–B4, Ablations)
├── train_master_LATe.py             # Loss-Aware Ternarization (LATe) training engine
├── eval_unseen_cifar10_1.py         # Unseen CIFAR-10.1 evaluation and confusion matrix generator
├── resnet34_cifar10_teacher.ipynb   # Interactive Teacher pretraining notebook
├── resnet18_cifar10.ipynb           # Interactive Student reference notebook
├── PROJECT_DOCUMENTATION.md         # Detailed technical notes and mathematical formulations
└── results/                         # Empirical logs, checkpoints (.pth), learning curves, and histograms
    ├── 01_default_twn_temp4/        # Default TWN baselines (B1, B2, B3, B4 at T=4)
    ├── 02_twn_ablation_temp15/      # Temperature ablation (T=15)
    ├── 03_teacher_and_temp2_twn/    # Pretrained Teacher & Temperature ablation (T=2)
    ├── 04_twn_all_layers_quantized/ # Boundary layer precision ablation (FNL full-network ternarization)
    └── 05_late_loss_aware_ternarization/ # Loss-Aware Ternarization runs (B1–B4)
```

---

## Detailed Script Descriptions

### 1. `train_master.py`
The master execution and sweep runner for standard TWN and reference models.
- **Key Capabilities**:
  - Pretrains the FP32 Teacher (ResNet-34) and Student Base (ResNet-18).
  - Implements standard Ternary Weight Networks (TWN, Li et al., 2016) with trainable scaling factor $\alpha$ and dead-zone threshold $\Delta = 0.75 \mathbb{E}[|W|]$.
  - Supports Straight-Through Estimation (STE) over persistent latent FP32 weights.
  - Implements all 4 primary training baselines:
    - **Baseline 1 (`--mode baseline1`)**: TWN trained from scratch with supervised Cross-Entropy (no KD).
    - **Baseline 2 (`--mode baseline2`)**: Two-stage training (100-epoch FP32 KD warmup + 100-epoch TWN-QAT fine-tuning).
    - **Baseline 3 (`--mode baseline3`)**: Direct hard discrete projection in-place without persistent latent FP32 parameters.
    - **Baseline 4 (`--mode baseline4`)**: Joint end-to-end KD + TWN-QAT from scratch (200 epochs).
  - Supports temperature ablations (`-T / --kd_temperature`) and first/last-layer precision ablation (`--quantize_first_last`).

### 2. `train_master_LATe.py`
Implements **Loss-Aware Ternarization (LATe)** (Hou & Kwok, ICLR 2018).
- **Key Capabilities**:
  - Replaces the heuristic $\Delta = 0.75 \mathbb{E}[|W|]$ threshold with an exact search over sorted candidate weight partitions using cumulative prefix sums.
  - Weights reconstruction error by an online diagonal sensitivity estimate $d_i \approx \mathbb{E}[g_i^2]$ tracked via Exponential Moving Average (EMA).
  - Trains LATe Baselines 1 through 4 under the identical 200-epoch protocol.

### 3. `eval_unseen_cifar10_1.py`
Autonomous benchmarking tool for evaluating generalization under natural distribution shift on **CIFAR-10.1** (Recht et al., 2018).
- **Key Capabilities**:
  - Automatically downloads and extracts the CIFAR-10.1 v6 dataset (2,000 images across 10 classes).
  - Performs standardized comparative evaluation across Teacher, Student Base, TWN Student, and LATe Student models.
  - Computes class-wise accuracies and generates side-by-side 10x10 confusion matrix plots.

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/vaidhav-adit/Hybrid_KD_and_TWN_QAT.git
cd Hybrid_KD_and_TWN_QAT

# Install dependencies
pip install -r requirements.txt
```

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
# Run all 4 primary baselines sequentially
python train_master.py --mode baselines_all --kd_temperature 4.0

# OR run individual baselines:
# Baseline 1: TWN Supervised Scratch (No KD)
python train_master.py --mode baseline1 --epochs 200

# Baseline 2: Two-Stage (100 ep FP32 KD Pretrain -> 100 ep TWN-QAT)
python train_master.py --mode baseline2 --teacher_ckpt teacher_best.pth --kd_temperature 4.0

# Baseline 3: Direct Hard Discrete Projection (No latent FP32 weights)
python train_master.py --mode baseline3 --teacher_ckpt teacher_best.pth --kd_temperature 4.0

# Baseline 4: Joint KD + TWN-QAT from scratch (Best Performing)
python train_master.py --mode baseline4 --teacher_ckpt teacher_best.pth --kd_temperature 4.0
```

### 3. Ablation Studies

```bash
# Temperature Ablation 1a: Low Temperature (T=2.0, T^2=4)
python train_master.py --mode baselines_all --kd_temperature 2.0 --teacher_ckpt teacher_best.pth

# Temperature Ablation 1b: High Temperature (T=15.0, T^2=225)
python train_master.py --mode baselines_all --kd_temperature 15.0 --teacher_ckpt teacher_best.pth

# Boundary Layer Precision Ablation: Full-Network Ternarization (FNL)
python train_master.py --mode baselines_all --kd_temperature 4.0 --quantize_first_last --teacher_ckpt teacher_best.pth
```

### 4. Loss-Aware Ternarization (LATe) Experiments

```bash
# Run all LATe baselines (B1 to B4) with sensitivity-weighted thresholding
python train_master_LATe.py --mode baselines_all --kd_temperature 4.0 --teacher_ckpt teacher_best.pth

# Run individual LATe Baseline 4:
python train_master_LATe.py --mode baseline4 --kd_temperature 4.0 --teacher_ckpt teacher_best.pth
```

### 5. Benchmark on Unseen CIFAR-10.1 Dataset

```bash
# Evaluate Teacher, Student Base, TWN B4, and LATe B4 together
python eval_unseen_cifar10_1.py \
  --teacher_ckpt results/03_teacher_and_temp2_twn/teacher_best.pth \
  --student_base_ckpt student_base_best.pth \
  --twn_student_ckpt results/01_default_twn_temp4/baseline4_temp4_best.pth \
  --late_student_ckpt results/05_late_loss_aware_ternarization/late_baseline4_temp4_best.pth

# Evaluate a single checkpoint on CIFAR-10.1
python eval_unseen_cifar10_1.py \
  --model_ckpt results/01_default_twn_temp4/baseline4_temp4_best.pth \
  --model_type twn
```

### 6. Standalone Checkpoint Evaluation

```bash
# Evaluate any saved checkpoint on CIFAR-10 validation/test split
python train_master.py --mode eval --model_type twn --eval_ckpt results/01_default_twn_temp4/baseline4_temp4_best.pth
```

---

## Experimental Results Summary

| Model / Configuration | Val Acc (%) | Test Acc (%) | CIFAR-10.1 (%) | Sparsity (%) | Storage (MB) | Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34 Teacher (FP32)** | **97.88** | **96.54** | **90.80** | 0.00% | 85.27 | $1.0\times$ |
| **ResNet-18 Student Base (FP32)** | 97.02 | 96.21 | 89.75 | 0.00% | 42.63 | $1.0\times$ |
| **Baseline 1 (TWN Scratch, No KD)** | 95.94 | 95.65 | -- | 52.00% | 2.72 | $15.7\times$ |
| **Baseline 2 (Two-Stage KD + TWN)** | 96.54 | 96.01 | -- | 52.00% | 2.72 | $15.7\times$ |
| **Baseline 3 (Hard Direct Projection)** | 82.66 | 82.01 | -- | 34.91% | 2.72 | $15.7\times$ |
| **Baseline 4 (Joint KD + TWN-QAT)** | **96.48** | **96.15** | **90.95** | 52.00% | 2.72 | **$15.7\times$** |
| *Ablation 1a (Joint TWN, $T=2$)* | 96.20 | 95.85 | -- | 52.00% | 2.72 | $15.7\times$ |
| *Ablation 1b (Joint TWN, $T=15$)* | 96.38 | 96.02 | -- | 52.00% | 2.72 | $15.7\times$ |
| *Ablation 2 (Full Ternary FNL)* | 96.36 | 95.90 | -- | 60.15% | 2.70 | $15.8\times$ |
| *Extension (LATe Baseline 4)* | 96.54 | 95.91 | 89.90 | **66.85%** | 2.72 | $15.7\times$ |

---

## Citation & References

```bibtex
@article{adit2026kdtwn,
  title={Knowledge Distillation with Ternary Weight Quantization-Aware Training: An Experimental Study of Training Strategies},
  author={Adit, Vaidhav},
  year={2026}
}
```
