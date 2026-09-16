"""
Master Training, Sweep, and Evaluation Script for LATe (Loss-Aware Ternarization) CIFAR-10 Experiments.
Based on the ICLR paper "Loss-aware Weight Quantization of Deep Networks" (Hou & Kwok, 2018).

Supported Modes:
  1. baselines_all : Runs all 4 LATe Baselines (1, 2, 3, 4) sequentially using the pretrained Teacher.
  2. baseline1     : LATe ResNet-18 trained from scratch (CE Loss only, no KD).
  3. baseline2     : Pretrain FP32 Student with KD (100 epochs) -> Convert to LATe -> QAT Fine-tuning with KD (100 epochs).
  4. baseline3     : Hard-Projected LATe ResNet-18 + KD from scratch (W_{t+1} = Q_LATe(W_t - eta * g_t), no latent FP32 accumulator).
  5. baseline4     : LATe ResNet-18 QAT + KD from scratch (Latent FP32 maintained throughout with STE).
  6. eval          : Diagnostic evaluation, sparsity verification, and model size analysis on a saved checkpoint.

Key Hyperparameter Switches via CLI:
  --kd_temperature / -T : Distillation temperature T (default: 4.0).
  --quantize_first_last : Toggle ternarization of first stem conv & last linear layer (default: False = kept in FP32).
  --kd_lambda           : Weight on KD loss vs CE loss (default: 0.5).
  --label_smoothing     : Label smoothing on CE loss (default: 0.0 = pure hard one-hot CE).
  --deterministic       : Enable bit-exact deterministic CUDA/cuDNN execution.
  --no_nesterov         : Disable Nesterov momentum (default: enabled).
  --epochs              : Total training epochs (default: 200).
  --base_lr             : Base learning rate (default: 0.1).
  --batch_size          : Batch size (default: 128).
  --curvature_beta      : EMA momentum beta for diagonal curvature tracking (default: 0.9).
"""

import os
import sys
import time
import copy
import random
import argparse
import types
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as T


# ==============================================================================
# 1. Reproducibility & Global Constants
# ==============================================================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. Data Pipeline
# ==============================================================================

class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        c, h, w = img.shape
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


class CIFARSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        img = self.transform(img)
        return img, label


def get_dataloaders(data_dir="./data", batch_size=128, val_size=5000, num_workers=4, cutout_length=16, seed=42):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(cutout_length),
    ])

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_train_raw = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    n_total = len(full_train_raw)
    n_val = val_size
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)

    train_set = CIFARSubset(full_train_raw, train_idx, train_transform)
    val_set = CIFARSubset(full_train_raw, val_idx, eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset summary: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_loader, val_loader, test_loader


# ==============================================================================
# 3. Loss-Aware Ternarization (LATe) Core Algorithm
# ==============================================================================

def late_quantize_exact(w, diag_curvature=None):
    """
    Exact Loss-Aware Ternarization (LATe) solver.
    Solves: min_{alpha >= 0, b in {-1, 0, +1}^n} sum_i d_i * (alpha * b_i - w_i)^2
    
    Closed-form exact solution via sorted prefix search:
      Let |w_i| be sorted descending.
      For partition k (top k elements active, remaining set to zero):
        alpha(k) = (sum_{j=1..k} d_j * |w_j|) / (sum_{j=1..k} d_j)
        Maximized surrogate objective: S(k) = (sum_{j=1..k} d_j * |w_j|)^2 / (sum_{j=1..k} d_j)
      Find k* = argmax S(k).
      Optimal threshold Delta* = alpha(k*) / 2.
    """
    with torch.no_grad():
        w_flat = w.view(-1)
        abs_w = w_flat.abs()
        n = abs_w.numel()

        if diag_curvature is not None:
            d_flat = diag_curvature.view(-1)
            # Normalize layer curvature so its mean is 1.0, preserving relative sensitivity without numerical underflow
            d_mean = d_flat.mean().clamp(min=1e-8)
            d_flat = d_flat / d_mean
        else:
            d_flat = torch.ones_like(w_flat)

        # Sort absolute values descending
        sorted_abs_w, sorted_idx = torch.sort(abs_w, descending=True)
        sorted_d = d_flat[sorted_idx]

        # Cumulative sums for exact weighted objective
        weighted_w = sorted_d * sorted_abs_w
        p_k = torch.cumsum(weighted_w, dim=0)
        d_k = torch.cumsum(sorted_d, dim=0).clamp(min=1e-8)

        # Objective surrogate S(k) = (P_k)^2 / D_k
        s_k = (p_k ** 2) / d_k
        best_k_idx = torch.argmax(s_k)

        # Optimal alpha and exact threshold Delta = alpha / 2
        alpha = p_k[best_k_idx] / d_k[best_k_idx]
        if torch.isnan(alpha) or torch.isinf(alpha) or alpha < 1e-5:
            alpha = abs_w.mean().clamp(min=1e-5)
        delta = 0.5 * alpha

        # Generate ternary mask {-1, 0, +1}
        mask = (abs_w > delta).to(w_flat.dtype)
        if mask.sum() == 0:
            delta = 0.5 * abs_w.mean().clamp(min=1e-5)
            mask = (abs_w > delta).to(w_flat.dtype)
            alpha = (abs_w * mask).sum() / mask.sum().clamp(min=1.0)
            if torch.isnan(alpha) or alpha < 1e-5:
                alpha = abs_w.mean().clamp(min=1e-5)

        w_ternary = (alpha * torch.sign(w_flat) * mask).view_as(w)

    # Straight-Through Estimator (STE)
    return w + (w_ternary - w).detach()


def late_stats(w, diag_curvature=None):
    """Computes ternary distribution stats using the exact LATe threshold Delta = alpha / 2."""
    with torch.no_grad():
        w_flat = w.view(-1)
        abs_w = w_flat.abs()
        if diag_curvature is not None:
            d_flat = diag_curvature.view(-1)
            d_mean = d_flat.mean().clamp(min=1e-8)
            d_flat = d_flat / d_mean
        else:
            d_flat = torch.ones_like(w_flat)

        sorted_abs_w, sorted_idx = torch.sort(abs_w, descending=True)
        sorted_d = d_flat[sorted_idx]
        weighted_w = sorted_d * sorted_abs_w
        p_k = torch.cumsum(weighted_w, dim=0)
        d_k = torch.cumsum(sorted_d, dim=0).clamp(min=1e-8)
        s_k = (p_k ** 2) / d_k
        best_k_idx = torch.argmax(s_k)
        alpha = (p_k[best_k_idx] / d_k[best_k_idx]).item()
        if np.isnan(alpha) or np.isinf(alpha) or alpha < 1e-5:
            alpha = abs_w.mean().clamp(min=1e-5).item()
        delta = 0.5 * alpha

        pos = (w > delta).float().mean().item()
        neg = (w < -delta).float().mean().item()
        zero = max(0.0, 1.0 - pos - neg)
    return {"zero_frac": zero, "pos_frac": pos, "neg_frac": neg, "alpha": alpha, "delta": delta}


# ==============================================================================
# 4. Neural Network Architectures (LATe Student & FP32 Teacher)
# ==============================================================================

# --- FP32 Modules for Pretrained Teacher & FP32 Student Reference ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def ResNet34(num_classes=10):
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet18(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# --- LATe Quantized Layers (Baselines 1, 2, 4) ---
class LATeConv2d(nn.Conv2d):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))

    def update_curvature(self):
        if self.weight.grad is not None:
            grad_sq = self.weight.grad.data.detach() ** 2
            if not torch.isnan(grad_sq).any() and not torch.isinf(grad_sq).any():
                self.diag_curvature.mul_(self.curvature_beta).add_(grad_sq, alpha=1.0 - self.curvature_beta)

    def forward(self, x):
        curv = torch.sqrt(self.diag_curvature + 1e-8)
        w_q = late_quantize_exact(self.weight, curv)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LATeLinear(nn.Linear):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))

    def update_curvature(self):
        if self.weight.grad is not None:
            grad_sq = self.weight.grad.data.detach() ** 2
            if not torch.isnan(grad_sq).any() and not torch.isinf(grad_sq).any():
                self.diag_curvature.mul_(self.curvature_beta).add_(grad_sq, alpha=1.0 - self.curvature_beta)

    def forward(self, x):
        curv = torch.sqrt(self.diag_curvature + 1e-8)
        w_q = late_quantize_exact(self.weight, curv)
        return F.linear(x, w_q, self.bias)


class LATeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, curvature_beta=0.9):
        super().__init__()
        self.conv1 = LATeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                curvature_beta=curvature_beta)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LATeConv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                                curvature_beta=curvature_beta)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                LATeConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride,
                           curvature_beta=curvature_beta),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class LATeResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, curvature_beta=0.9,
                 quantize_first_last=False):
        super().__init__()
        self.in_planes = 64
        self.curvature_beta = curvature_beta

        if quantize_first_last:
            self.conv1 = LATeConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                    curvature_beta=curvature_beta)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, curvature_beta=curvature_beta)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, curvature_beta=curvature_beta)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, curvature_beta=curvature_beta)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, curvature_beta=curvature_beta)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if quantize_first_last:
            self.fc = LATeLinear(512 * block.expansion, num_classes, curvature_beta=curvature_beta)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, curvature_beta):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, curvature_beta=curvature_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def LATeResNet18(num_classes=10, curvature_beta=0.9, quantize_first_last=False):
    return LATeResNetCIFAR(LATeBasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                           curvature_beta=curvature_beta, quantize_first_last=quantize_first_last)


# --- Direct Hard-Projected LATe Architecture (Baseline 3: No Latent FP32 Accumulator) ---
def project_to_late_(weight, diag_curvature=None):
    with torch.no_grad():
        w_flat = weight.view(-1)
        abs_w = w_flat.abs()
        if diag_curvature is not None:
            d_flat = diag_curvature.view(-1)
            d_mean = d_flat.mean().clamp(min=1e-8)
            d_flat = d_flat / d_mean
        else:
            d_flat = torch.ones_like(w_flat)

        sorted_abs_w, sorted_idx = torch.sort(abs_w, descending=True)
        sorted_d = d_flat[sorted_idx]
        weighted_w = sorted_d * sorted_abs_w
        p_k = torch.cumsum(weighted_w, dim=0)
        d_k = torch.cumsum(sorted_d, dim=0).clamp(min=1e-8)
        s_k = (p_k ** 2) / d_k
        best_k_idx = torch.argmax(s_k)
        alpha = p_k[best_k_idx] / d_k[best_k_idx]
        if torch.isnan(alpha) or torch.isinf(alpha) or alpha < 1e-5:
            alpha = abs_w.mean().clamp(min=1e-5)
        delta = 0.5 * alpha

        mask = (abs_w > delta).to(weight.dtype)
        if mask.sum() == 0:
            delta = 0.5 * abs_w.mean().clamp(min=1e-5)
            mask = (abs_w > delta).to(weight.dtype)
            alpha = (abs_w * mask).sum() / mask.sum().clamp(min=1.0)
            if torch.isnan(alpha) or alpha < 1e-5:
                alpha = abs_w.mean().clamp(min=1e-5)

        w_ternary = (alpha * torch.sign(w_flat) * mask).view_as(weight)
        weight.copy_(w_ternary)


class DirectLATeConv2d(nn.Conv2d):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))
        project_to_late_(self.weight.data, self.diag_curvature)

    def update_curvature(self):
        if self.weight.grad is not None:
            grad_sq = self.weight.grad.data.detach() ** 2
            if not torch.isnan(grad_sq).any() and not torch.isinf(grad_sq).any():
                self.diag_curvature.mul_(self.curvature_beta).add_(grad_sq, alpha=1.0 - self.curvature_beta)


class DirectLATeLinear(nn.Linear):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))
        project_to_late_(self.weight.data, self.diag_curvature)

    def update_curvature(self):
        if self.weight.grad is not None:
            grad_sq = self.weight.grad.data.detach() ** 2
            if not torch.isnan(grad_sq).any() and not torch.isinf(grad_sq).any():
                self.diag_curvature.mul_(self.curvature_beta).add_(grad_sq, alpha=1.0 - self.curvature_beta)


def project_all_direct_late_params(model):
    for module in model.modules():
        if isinstance(module, (DirectLATeConv2d, DirectLATeLinear)):
            curv = torch.sqrt(module.diag_curvature + 1e-8)
            project_to_late_(module.weight.data, curv)


class DirectLATeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, curvature_beta=0.9):
        super().__init__()
        self.conv1 = DirectLATeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                      curvature_beta=curvature_beta)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DirectLATeConv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                                      curvature_beta=curvature_beta)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                DirectLATeConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride,
                                 curvature_beta=curvature_beta),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class DirectLATeResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, curvature_beta=0.9,
                 quantize_first_last=False):
        super().__init__()
        self.in_planes = 64
        self.curvature_beta = curvature_beta

        if quantize_first_last:
            self.conv1 = DirectLATeConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                          curvature_beta=curvature_beta)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, curvature_beta=curvature_beta)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, curvature_beta=curvature_beta)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, curvature_beta=curvature_beta)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, curvature_beta=curvature_beta)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if quantize_first_last:
            self.fc = DirectLATeLinear(512 * block.expansion, num_classes, curvature_beta=curvature_beta)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, curvature_beta):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, curvature_beta=curvature_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def DirectLATeResNet18(num_classes=10, curvature_beta=0.9, quantize_first_last=False):
    return DirectLATeResNetCIFAR(DirectLATeBasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                                 curvature_beta=curvature_beta, quantize_first_last=quantize_first_last)


# ==============================================================================
# 5. Losses & Optimizer Setup
# ==============================================================================

def kd_loss(student_logits, teacher_logits, T):
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    with torch.no_grad():
        p_t = F.softmax(teacher_logits / T, dim=1)
    return -(p_t * log_p_s).sum(dim=1).mean()


def combined_loss(student_logits, teacher_logits, labels, T, lam, ce_criterion):
    l_kd = kd_loss(student_logits, teacher_logits, T)
    l_ce = ce_criterion(student_logits, labels)
    return lam * (T ** 2) * l_kd + (1 - lam) * l_ce


def build_optimizer_and_scheduler(model, base_lr=0.1, momentum=0.9, nesterov=True,
                                  weight_decay=5e-4, warmup_epochs=5, total_epochs=200, steps_per_epoch=351):
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ==============================================================================
# 6. Core Epoch Execution & Curvature Tracking
# ==============================================================================

def update_model_curvatures(model):
    for m in model.modules():
        if hasattr(m, "update_curvature") and callable(m.update_curvature):
            m.update_curvature()


def run_one_epoch(model, loader, optimizer, scheduler, scaler, device, train_mode, amp=True,
                  grad_clip=5.0, ce_criterion=None, teacher=None, kd_temp=4.0, kd_lam=0.5,
                  is_direct_late=False):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp):
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    student_logits = model(images)
                    loss = combined_loss(student_logits, teacher_logits, labels, kd_temp, kd_lam, ce_criterion)
                else:
                    outputs = model(images)
                    loss = ce_criterion(outputs, labels)
                    student_logits = outputs

            if train_mode:
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Loss-aware diagonal curvature update from exact gradient information
                update_model_curvatures(model)

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if is_direct_late:
                    project_all_direct_late_params(model)

                scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds = student_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def evaluate_model(model, loader, device, ce_criterion, teacher=None, kd_temp=4.0, kd_lam=0.5, amp=True):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp):
                if teacher is not None:
                    teacher_logits = teacher(images)
                    outputs = model(images)
                    loss = combined_loss(outputs, teacher_logits, labels, kd_temp, kd_lam, ce_criterion)
                else:
                    outputs = model(images)
                    loss = ce_criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# ==============================================================================
# 7. Diagnostics, Plots & Size Calculation
# ==============================================================================

def print_model_size_summary(model, is_ternary=True, ckpt_path=None, title_suffix=""):
    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = 0
    fp32_params = 0
    num_alpha_scalars = 0

    for name, module in model.named_modules():
        if isinstance(module, (LATeConv2d, LATeLinear, DirectLATeConv2d, DirectLATeLinear)):
            ternary_params += module.weight.numel()
            num_alpha_scalars += 1
            if module.bias is not None:
                fp32_params += module.bias.numel()
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            fp32_params += module.weight.numel()
            if module.bias is not None:
                fp32_params += module.bias.numel()
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                fp32_params += module.weight.numel()
            if module.bias is not None:
                fp32_params += module.bias.numel()

    raw_fp32_size_mb = (total_params * 32) / (8 * 1024 * 1024)
    ternary_bits = ternary_params * 2
    fp32_bits = fp32_params * 32
    alpha_bits = num_alpha_scalars * 32
    total_compressed_bits = ternary_bits + fp32_bits + alpha_bits
    theoretical_ternary_size_mb = total_compressed_bits / (8 * 1024 * 1024)
    compression_ratio = raw_fp32_size_mb / max(theoretical_ternary_size_mb, 1e-8)

    print("\n" + "=" * 75)
    print(f"LATe MODEL SIZE & BIT-LEVEL COMPRESSION MATHEMATICS {title_suffix}")
    print("=" * 75)
    print("1. Parameter Count Breakdown:")
    print(f"   - Total Parameters:                     {total_params:>12,}")
    print(f"   - LATe Ternary (2-bit) Parameters:      {ternary_params:>12,} ({ternary_params/total_params*100:6.2f}%)")
    print(f"   - FP32 Unquantized Parameters:          {fp32_params:>12,} ({fp32_params/total_params*100:6.2f}%)")
    print(f"   - Layerwise Alpha Scaling Factors:      {num_alpha_scalars:>12,} (FP32 scalars)")

    print("\n2. Exact Bitstream Storage Calculation:")
    print(f"   - LATe Weight Bits    : {ternary_params:>10,} params x  2 bits = {ternary_bits:>12,} bits")
    print(f"   - FP32 Parameter Bits : {fp32_params:>10,} params x 32 bits = {fp32_bits:>12,} bits")
    print(f"   - Alpha Scaling Bits  : {num_alpha_scalars:>10,} layers x 32 bits = {alpha_bits:>12,} bits")
    print("   " + "-" * 67)
    print(f"   - Total Compressed Bits:                          = {total_compressed_bits:>12,} bits")
    print(f"   - Compressed Size: {total_compressed_bits:,} / (8 x 1024^2)    = {theoretical_ternary_size_mb:>12.2f} MB")

    print("\n3. Baseline Comparison & Compression Summary:")
    print(f"   - Full FP32 Baseline Footprint:                   {raw_fp32_size_mb:>12.2f} MB")
    print(f"   - Theoretical 2-Bit LATe Footprint (Deployment):  {theoretical_ternary_size_mb:>12.2f} MB")
    print(f"   - Mathematical Compression Ratio:                 {compression_ratio:>12.2f}x")
    if ckpt_path and os.path.exists(ckpt_path):
        disk_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"   - Training Checkpoint on Disk (FP32 state_dict):  {disk_size_mb:>12.2f} MB")
    print("=" * 75 + "\n")


def print_sparsity_and_diagnostics(model):
    print("\nPer-layer weight shape, unique quantized weights, and LATe sparsity:")
    for name, module in model.named_modules():
        if isinstance(module, (LATeConv2d, LATeLinear)):
            curv = torch.sqrt(module.diag_curvature + 1e-8)
            stats = late_stats(module.weight, curv)
            wq = late_quantize_exact(module.weight, curv)
            n_uniq = torch.unique(wq).numel()
            print(f"  {name:30s} | shape: {str(tuple(module.weight.shape)):18s} | unique: {n_uniq:2d} | "
                  f"zero={stats['zero_frac']*100:5.1f}% +alpha={stats['pos_frac']*100:5.1f}% -alpha={stats['neg_frac']*100:5.1f}% | alpha={stats['alpha']:.4f}")
        elif isinstance(module, (DirectLATeConv2d, DirectLATeLinear)):
            curv = torch.sqrt(module.diag_curvature + 1e-8)
            stats = late_stats(module.weight, curv)
            n_uniq = torch.unique(module.weight.detach()).numel()
            print(f"  {name:30s} | shape: {str(tuple(module.weight.shape)):18s} | unique: {n_uniq:2d} | "
                  f"zero={stats['zero_frac']*100:5.1f}% +alpha={stats['pos_frac']*100:5.1f}% -alpha={stats['neg_frac']*100:5.1f}% | alpha={stats['alpha']:.4f}")


def run_behavioral_quantization_check(model, test_loader, device):
    print("\nRunning LATe Quantization Behavioral Verification Check...")
    _images, _labels = next(iter(test_loader))
    _images, _labels = _images.to(device), _labels.to(device)

    model.eval()
    with torch.no_grad():
        logits_quantized = model(_images)

    def _forward_conv_raw(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def _forward_linear_raw(self, x):
        return F.linear(x, self.weight, self.bias)

    _orig_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, LATeConv2d):
            _orig_forwards[name] = module.forward
            module.forward = types.MethodType(_forward_conv_raw, module)
        elif isinstance(module, LATeLinear):
            _orig_forwards[name] = module.forward
            module.forward = types.MethodType(_forward_linear_raw, module)

    with torch.no_grad():
        logits_raw = model(_images)

    for name, module in model.named_modules():
        if name in _orig_forwards:
            module.forward = _orig_forwards[name]

    preds_quantized = logits_quantized.argmax(dim=1)
    preds_raw = logits_raw.argmax(dim=1)

    logit_diff = (logits_quantized - logits_raw).abs().mean().item()
    agreement = (preds_quantized == preds_raw).float().mean().item() * 100

    print(f"Mean |logit difference| between LATe quantized and raw latent forward: {logit_diff:.4f}")
    print(f"Prediction agreement between the two passes: {agreement:.2f}%")
    if logit_diff < 1e-3 and agreement > 99.5:
        print("WARNING: Quantization may not be active.")
    else:
        print("LATe Quantization is genuinely constraining the network forward pass.")


def plot_and_save_curves(history, out_path, title_prefix="Training Curves"):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax.plot(epochs_range, history["test_loss"],  label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} - Loss vs. Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, history["train_acc"], label="Train Accuracy")
    ax.plot(epochs_range, history["val_acc"],   label="Val Accuracy")
    ax.plot(epochs_range, history["test_acc"],  label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title_prefix} - Accuracy vs. Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curves to '{out_path}'.")


def plot_and_save_histograms(model, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    latent_chunks = []
    neg_count, zero_count, pos_count, total_t_count = 0, 0, 0, 0

    for name, m in model.named_modules():
        if isinstance(m, (LATeConv2d, LATeLinear)):
            latent_chunks.append(m.weight.detach().cpu().flatten().numpy())
            with torch.no_grad():
                curv = torch.sqrt(m.diag_curvature + 1e-8)
                st = late_stats(m.weight.detach(), curv)
                tot = m.weight.numel()
                pos = int(round(st['pos_frac'] * tot))
                neg = int(round(st['neg_frac'] * tot))
                zero = tot - pos - neg
                neg_count += neg
                zero_count += zero
                pos_count += pos
                total_t_count += tot
        elif isinstance(m, (DirectLATeConv2d, DirectLATeLinear)):
            with torch.no_grad():
                w = m.weight.detach()
                pos = (w > 1e-6).sum().item()
                neg = (w < -1e-6).sum().item()
                tot = w.numel()
                zero = tot - pos - neg
                neg_count += neg
                zero_count += zero
                pos_count += pos
                total_t_count += tot

    if latent_chunks:
        latent_all = np.concatenate(latent_chunks)
        axes[0].hist(latent_all, bins=200, color="steelblue", edgecolor="none")
        axes[0].set_yscale("log")
        axes[0].set_title(f"LATe Latent Continuous FP32 Weights {title_suffix}")
        axes[0].set_xlabel("Latent Weight Value")
        axes[0].set_ylabel("Count (log scale)")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No Latent FP32 Weights\n(Hard-Projected LATe Training)",
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[0].set_axis_off()

    if total_t_count > 0:
        pct_neg = 100.0 * neg_count / total_t_count
        pct_zero = 100.0 * zero_count / total_t_count
        pct_pos = 100.0 * pos_count / total_t_count

        states = ['-1 (-alpha)', '0 (Zero)', '+1 (+alpha)']
        counts = [neg_count, zero_count, pos_count]
        percentages = [pct_neg, pct_zero, pct_pos]
        colors = ['crimson', 'gray', 'forestgreen']

        bars = axes[1].bar(states, counts, color=colors, width=0.5, alpha=0.85, edgecolor='black')
        axes[1].set_yscale("log")
        axes[1].set_title(f"LATe Quantized Discrete Ternary States {{-1, 0, +1}} {title_suffix}")
        axes[1].set_ylabel("Parameter Count (log scale)")
        axes[1].grid(axis='y', alpha=0.3)

        for bar, pct, cnt in zip(bars, percentages, counts):
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2.0, yval * 1.15,
                         f"{pct:.2f}%\n({cnt:,})",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[1].set_ylim(top=max(counts) * 5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved weight histograms to '{out_path}'.")


# ==============================================================================
# 8. Single Baseline Execution Function
# ==============================================================================

def run_single_baseline(mode, args, train_loader, val_loader, test_loader, ce_criterion, teacher, device):
    print("\n" + "=" * 75)
    print(f"RUNNING LATe EXPERIMENT: {mode.upper()}")
    print(f"Configuration: Temp={args.kd_temperature}, KD Lambda={args.kd_lambda}, LabelSmoothing={args.label_smoothing}, QuantizeFirstLast={args.quantize_first_last}")
    print("=" * 75)

    fnl_tag = "_fnl_ternary" if args.quantize_first_last else ""
    temp_tag = f"_temp{int(args.kd_temperature) if args.kd_temperature.is_integer() else args.kd_temperature}" if mode in ["baseline2", "baseline3", "baseline4"] else ""

    if args.ckpt_name and len(args.mode.split(',')) == 1 and args.mode != "baselines_all":
        ckpt_best = args.ckpt_name
        ckpt_final = args.ckpt_name.replace(".pth", "_final.pth")
    else:
        ckpt_best = f"late_{mode}{temp_tag}{fnl_tag}_best.pth"
        ckpt_final = f"late_{mode}{temp_tag}{fnl_tag}_final.pth"

    # --- Mode: BASELINE 1 (LATe Scratch, No KD) ---
    if mode == "baseline1":
        model = LATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                             quantize_first_last=args.quantize_first_last).to(device)

    # --- Mode: BASELINE 2 (Two-Stage Pretrain + LATe Fine-tune) ---
    elif mode == "baseline2":
        epochs_stage1 = args.epochs // 2
        epochs_stage2 = args.epochs - epochs_stage1

        # Stage 1: FP32 KD Pretraining
        print(f"\n--- Baseline 2 Stage 1: FP32 KD Pretraining ({epochs_stage1} epochs) ---")
        student_fp32 = ResNet18(args.num_classes).to(device)
        opt1, sched1 = build_optimizer_and_scheduler(
            student_fp32, base_lr=args.base_lr, momentum=args.momentum,
            nesterov=args.nesterov, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, total_epochs=epochs_stage1,
            steps_per_epoch=len(train_loader)
        )
        scaler1 = GradScaler(enabled=args.amp)
        best_val_acc_s1 = 0.0
        stage1_ckpt = f"late_baseline2_stage1_fp32{temp_tag}_best.pth"

        history_s1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}

        for epoch in range(1, epochs_stage1 + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_one_epoch(
                student_fp32, train_loader, opt1, sched1, scaler1, device,
                train_mode=True, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )
            va_loss, va_acc = run_one_epoch(
                student_fp32, val_loader, opt1, sched1, scaler1, device,
                train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )
            te_loss, te_acc = run_one_epoch(
                student_fp32, test_loader, opt1, sched1, scaler1, device,
                train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )

            history_s1["train_loss"].append(tr_loss); history_s1["train_acc"].append(tr_acc)
            history_s1["val_loss"].append(va_loss);   history_s1["val_acc"].append(va_acc)
            history_s1["test_loss"].append(te_loss);  history_s1["test_acc"].append(te_acc)

            if va_acc > best_val_acc_s1:
                best_val_acc_s1 = va_acc
                torch.save(student_fp32.state_dict(), stage1_ckpt)

            print(f"[LATe Baseline2 - Stage 1 {epoch:3d}/{epochs_stage1}] "
                  f"lr {opt1.param_groups[0]['lr']:.4f} | "
                  f"train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}% | "
                  f"test {te_loss:.4f}/{te_acc:.2f}% | {time.time()-t0:.1f}s")

        # Stage 2: Convert to LATe & Fine-tune with KD
        print(f"\n--- Baseline 2 Stage 2: LATe QAT Fine-tuning ({epochs_stage2} epochs) ---")
        model = LATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                             quantize_first_last=args.quantize_first_last).to(device)
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device), strict=False)

        opt2, sched2 = build_optimizer_and_scheduler(
            model, base_lr=args.base_lr, momentum=args.momentum,
            nesterov=args.nesterov, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, total_epochs=epochs_stage2,
            steps_per_epoch=len(train_loader)
        )
        scaler2 = GradScaler(enabled=args.amp)
        best_val_acc = 0.0
        history_s2 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}

        for epoch in range(1, epochs_stage2 + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_one_epoch(
                model, train_loader, opt2, sched2, scaler2, device,
                train_mode=True, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )
            va_loss, va_acc = run_one_epoch(
                model, val_loader, opt2, sched2, scaler2, device,
                train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )
            te_loss, te_acc = run_one_epoch(
                model, test_loader, opt2, sched2, scaler2, device,
                train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
                ce_criterion=ce_criterion, teacher=teacher,
                kd_temp=args.kd_temperature, kd_lam=args.kd_lambda
            )

            history_s2["train_loss"].append(tr_loss); history_s2["train_acc"].append(tr_acc)
            history_s2["val_loss"].append(va_loss);   history_s2["val_acc"].append(va_acc)
            history_s2["test_loss"].append(te_loss);  history_s2["test_acc"].append(te_acc)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), ckpt_best)

            print(f"[LATe Baseline2 - Stage 2 {epoch:3d}/{epochs_stage2}] "
                  f"lr {opt2.param_groups[0]['lr']:.4f} | "
                  f"train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}% | "
                  f"test {te_loss:.4f}/{te_acc:.2f}% | {time.time()-t0:.1f}s")

        torch.save(model.state_dict(), ckpt_final)
        combined_history = {k: history_s1[k] + history_s2[k] for k in history_s1}
        plot_and_save_curves(combined_history, f"late_{mode}{temp_tag}{fnl_tag}_curves.png", title_prefix=f"LATe {mode.upper()}{temp_tag}{fnl_tag}")
        plot_and_save_histograms(model, f"late_{mode}{temp_tag}{fnl_tag}_histograms.png", title_suffix=f"(LATe {mode}{temp_tag}{fnl_tag})")

        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        final_test_loss, final_test_acc = evaluate_model(model, test_loader, device, ce_criterion, teacher=teacher,
                                                         kd_temp=args.kd_temperature, kd_lam=args.kd_lambda, amp=args.amp)
        print(f"\nFinal Best LATe Checkpoint Test Accuracy: {final_test_acc:.2f}% (Loss: {final_test_loss:.4f})")
        print_sparsity_and_diagnostics(model)
        run_behavioral_quantization_check(model, test_loader, device)
        print_model_size_summary(model, is_ternary=True, ckpt_path=ckpt_best, title_suffix=f"(LATe {mode}{temp_tag}{fnl_tag})")

        return {"mode": mode, "best_val_acc": best_val_acc, "test_acc": final_test_acc, "test_loss": final_test_loss, "ckpt": ckpt_best}

    # --- Mode: BASELINE 3 (Hard-Projected LATe, No Latent FP32) ---
    elif mode == "baseline3":
        model = DirectLATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                                   quantize_first_last=args.quantize_first_last).to(device)

    # --- Mode: BASELINE 4 (Joint LATe QAT + KD from Scratch with STE) ---
    elif mode == "baseline4":
        model = LATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                             quantize_first_last=args.quantize_first_last).to(device)

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

    # --- Continuous 200-Epoch Training Loop for Baselines 1, 3, 4 ---
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, base_lr=args.base_lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, total_epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    scaler = GradScaler(enabled=args.amp)
    best_val_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            train_mode=True, amp=args.amp, grad_clip=args.grad_clip,
            ce_criterion=ce_criterion, teacher=teacher,
            kd_temp=args.kd_temperature, kd_lam=args.kd_lambda,
            is_direct_late=(mode == "baseline3")
        )
        va_loss, va_acc = run_one_epoch(
            model, val_loader, optimizer, scheduler, scaler, device,
            train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
            ce_criterion=ce_criterion, teacher=teacher,
            kd_temp=args.kd_temperature, kd_lam=args.kd_lambda,
            is_direct_late=(mode == "baseline3")
        )
        te_loss, te_acc = run_one_epoch(
            model, test_loader, optimizer, scheduler, scaler, device,
            train_mode=False, amp=args.amp, grad_clip=args.grad_clip,
            ce_criterion=ce_criterion, teacher=teacher,
            kd_temp=args.kd_temperature, kd_lam=args.kd_lambda,
            is_direct_late=(mode == "baseline3")
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["test_loss"].append(te_loss);  history["test_acc"].append(te_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_best)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"lr {current_lr:.4f} | "
              f"train {tr_loss:.4f}/{tr_acc:.2f}% | val {va_loss:.4f}/{va_acc:.2f}% | "
              f"test {te_loss:.4f}/{te_acc:.2f}% | {time.time()-t0:.1f}s")

    total_time = time.time() - start_time
    torch.save(model.state_dict(), ckpt_final)
    print(f"\nLATe {mode.upper()} Training Complete in {total_time/3600:.2f} hours. Best Val Acc: {best_val_acc:.2f}% (Saved: {ckpt_best})")

    plot_and_save_curves(history, f"late_{mode}{temp_tag}{fnl_tag}_curves.png", title_prefix=f"LATe {mode.upper()}{temp_tag}{fnl_tag}")
    plot_and_save_histograms(model, f"late_{mode}{temp_tag}{fnl_tag}_histograms.png", title_suffix=f"(LATe {mode}{temp_tag}{fnl_tag})")

    model.load_state_dict(torch.load(ckpt_best, map_location=device))
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, device, ce_criterion, teacher=teacher,
                                                     kd_temp=args.kd_temperature, kd_lam=args.kd_lambda, amp=args.amp)
    print(f"\nFinal Best LATe Checkpoint Test Accuracy: {final_test_acc:.2f}% (Loss: {final_test_loss:.4f})")

    print_sparsity_and_diagnostics(model)
    if mode != "baseline3":
        run_behavioral_quantization_check(model, test_loader, device)

    print_model_size_summary(model, is_ternary=True, ckpt_path=ckpt_best, title_suffix=f"(LATe {mode}{temp_tag}{fnl_tag})")
    return {"mode": mode, "best_val_acc": best_val_acc, "test_acc": final_test_acc, "test_loss": final_test_loss, "ckpt": ckpt_best}


# ==============================================================================
# 9. Standalone Checkpoint Evaluation Routine
# ==============================================================================

def run_evaluation_only(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("=" * 60)
    print(f"LATe CIFAR-10 Evaluator: Checkpoint = {args.eval_ckpt}")
    print(f"Model Type: {args.model_type}")
    print("=" * 60)

    if not os.path.exists(args.eval_ckpt):
        print(f"ERROR: Checkpoint file '{args.eval_ckpt}' does not exist!")
        sys.exit(1)

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.model_type == "late_twn":
        model = LATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                             quantize_first_last=args.quantize_first_last).to(device)
    elif args.model_type == "late_direct":
        model = DirectLATeResNet18(args.num_classes, curvature_beta=args.curvature_beta,
                                   quantize_first_last=args.quantize_first_last).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.load_state_dict(torch.load(args.eval_ckpt, map_location=device))
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    loss, acc = evaluate_model(model, test_loader, device, ce_criterion, amp=args.amp)
    print(f"\nCheckpoint Test Accuracy: {acc:.2f}% | Test Loss: {loss:.4f}")

    print_sparsity_and_diagnostics(model)
    if args.model_type == "late_twn":
        run_behavioral_quantization_check(model, test_loader, device)

    print_model_size_summary(model, is_ternary=True, ckpt_path=args.eval_ckpt)


# ==============================================================================
# 10. CLI Argument Parser & Orchestration
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ATDL LATe (Loss-Aware Ternarization) Master Runner")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baselines_all", "baseline1", "baseline2", "baseline3", "baseline4", "eval"],
                        help="Experiment mode: 'baselines_all' (runs 4 LATe baselines) or specific individual baseline.")
    parser.add_argument("--model_type", type=str, default="late_twn",
                        choices=["late_twn", "late_direct"],
                        help="Model type (required when mode=eval).")
    parser.add_argument("--eval_ckpt", type=str, default=None,
                        help="Path to checkpoint for evaluation mode.")

    # KD & LATe Hyperparameters
    parser.add_argument("--kd_temperature", "-T", type=float, default=4.0,
                        help="Distillation temperature T (default: 4.0).")
    parser.add_argument("--kd_lambda", type=float, default=0.5,
                        help="Weight lambda on KD loss term (default: 0.5).")
    parser.add_argument("--curvature_beta", type=float, default=0.9,
                        help="EMA momentum beta for diagonal curvature tracking (default: 0.9).")
    parser.add_argument("--quantize_first_last", action="store_true", default=False,
                        help="Quantize first stem conv and final linear layer to ternary (default: False, i.e., kept FP32).")

    # Training Setup
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total training epochs (default: 200).")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size (default: 128).")
    parser.add_argument("--base_lr", type=float, default=0.1,
                        help="Initial base learning rate (default: 0.1).")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (default: 0.9).")
    parser.add_argument("--no_nesterov", dest="nesterov", action="store_false", default=True,
                        help="Disable Nesterov momentum (default: enabled).")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay for SGD (default: 5e-4).")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor for Cross-Entropy (default: 0.0).")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs for scheduler (default: 5).")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Gradient norm clipping max norm (default: 5.0).")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable strict bit-exact deterministic CUDA/cuDNN execution.")

    # Paths & Diagnostics
    parser.add_argument("--teacher_ckpt", type=str, default="teacher_best.pth",
                        help="Path to trained FP32 teacher checkpoint (default: teacher_best.pth).")
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="Custom filename for saving the best checkpoint.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of dataset classes (default: 10).")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to CIFAR-10 data directory.")
    parser.add_argument("--val_size", type=int, default=5000,
                        help="Number of samples reserved for validation split (default: 5000).")
    parser.add_argument("--cutout_length", type=int, default=16,
                        help="Cutout augmentation patch length (default: 16).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers (default: 4).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--no_amp", dest="amp", action="store_false", default=True,
                        help="Disable automatic mixed precision (AMP).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device ('cuda', 'cpu', 'mps').")

    args = parser.parse_args()

    if args.mode == "eval":
        if args.eval_ckpt is None:
            print("ERROR: --eval_ckpt is required when --mode=eval")
            sys.exit(1)
        run_evaluation_only(args)
        return

    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Strict bit-exact determinism enabled.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        cutout_length=args.cutout_length,
        seed=args.seed
    )
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    modes_to_run = ["baseline1", "baseline2", "baseline3", "baseline4"] if args.mode == "baselines_all" else [args.mode]

    # Load frozen Teacher for KD baselines
    teacher = None
    needs_teacher = any(m in ["baseline2", "baseline3", "baseline4"] for m in modes_to_run)
    if needs_teacher:
        if not os.path.exists(args.teacher_ckpt):
            print(f"ERROR: Teacher checkpoint '{args.teacher_ckpt}' not found!")
            print("Please ensure the pretrained ResNet-34 teacher checkpoint is available.")
            sys.exit(1)
        print(f"Loading existing frozen FP32 teacher from '{args.teacher_ckpt}'...")
        teacher = ResNet34(args.num_classes).to(device)
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("Teacher successfully loaded.")

    results_summary = []
    for m in modes_to_run:
        t_obj = teacher if m in ["baseline2", "baseline3", "baseline4"] else None
        res = run_single_baseline(m, args, train_loader, val_loader, test_loader, ce_criterion, t_obj, device)
        results_summary.append(res)

    if len(results_summary) > 1:
        print("\n" + "=" * 80)
        print("CONSOLIDATED LATe EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Experiment Mode':<20} | {'Best Val Acc (%)':<18} | {'Test Acc (%)':<15} | {'Test Loss':<12} | {'Saved Checkpoint':<25}")
        print("-" * 80)
        for r in results_summary:
            print(f"LATe {r['mode'].upper():<15} | {r['best_val_acc']:<18.2f} | {r['test_acc']:<15.2f} | {r['test_loss']:<12.4f} | {r['ckpt']:<25}")
        print("=" * 80)


if __name__ == "__main__":
    main()
