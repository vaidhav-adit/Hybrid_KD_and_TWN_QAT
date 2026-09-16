# Knowledge Distillation with Ternary-Weight Quantization-Aware Training (KD-TWN & LATe)

Deep Learning research project implementing Knowledge Distillation (KD) combined with Quantization-Aware Training (QAT) using Ternary Weight Networks (TWN) and Loss-Aware Ternarization (LATe) on CIFAR-10.

- Teacher Model: Full-precision FP32 ResNet-34
- Student Model: Ternary-quantized ResNet-18 (weights constrained to {-alpha, 0, +alpha})

## Key Scripts
- `train_master.py`: Master pipeline for training Teacher, FP32 Student Base, and TWN Student baselines (Baselines 1 to 4).
- `train_master_LATe.py`: Implementation and training pipeline for Loss-Aware Ternarization (LATe).
- `eval_unseen_cifar10_1.py`: Evaluation and generalization verification on the unseen CIFAR-10.1 benchmark.

## Results Summary
For comprehensive tables, ablation studies, and evaluation logs, see `results_and_unseen_eval_summary.md` and the `results/` directory.
