"""
Evaluation Script for Unseen CIFAR-10.1 Dataset.
Evaluates Teacher, FP32 Student Base, Best TWN Student, and Best LATe Student
on the full unseen CIFAR-10.1 benchmark (200 images per class = 2,000 images total) or custom subsets.
Saves sample visualizations (2 images per class) and generates Confusion Matrices.

Usage:
  python eval_unseen_cifar10_1.py
  python eval_unseen_cifar10_1.py --twn_student_ckpt results/01_default_twn_temp4/baseline4_temp4_best.pth --late_student_ckpt results/05_late_loss_aware_ternarization/late_baseline4_temp4_best.pth
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ==============================================================================
# 1. Constants & Dataset Definition
# ==============================================================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_1_DATA_URL   = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy"
CIFAR10_1_LABELS_URL = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy"


class CIFAR10_1_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (32, 32, 3), uint8
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_cifar10_1_subset(data_dir="./data/cifar10_1", samples_per_class=200, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "cifar10.1_v6_data.npy")
    labels_path = os.path.join(data_dir, "cifar10.1_v6_labels.npy")

    if not os.path.exists(data_path):
        print(f"Downloading CIFAR-10.1 data from {CIFAR10_1_DATA_URL}...")
        urllib.request.urlretrieve(CIFAR10_1_DATA_URL, data_path)

    if not os.path.exists(labels_path):
        print(f"Downloading CIFAR-10.1 labels from {CIFAR10_1_LABELS_URL}...")
        urllib.request.urlretrieve(CIFAR10_1_LABELS_URL, labels_path)

    all_images = np.load(data_path)  # (2000, 32, 32, 3)
    all_labels = np.load(labels_path)  # (2000,)

    np.random.seed(seed)
    selected_indices = []

    for c in range(10):
        c_indices = np.where(all_labels == c)[0]
        actual_samples = min(samples_per_class, len(c_indices))
        chosen = np.random.choice(c_indices, size=actual_samples, replace=False)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    subset_images = all_images[selected_indices]
    subset_labels = all_labels[selected_indices]

    print(f"Loaded CIFAR-10.1 unseen subset: {len(subset_labels)} images total ({samples_per_class} per class across 10 classes).")
    return subset_images, subset_labels


# ==============================================================================
# 2. Image Verification Plot (2 Images Per Class)
# ==============================================================================

def plot_sample_images(images, labels, out_path="unseen_sample_images_2_per_class.png", samples_to_show=2):
    fig, axes = plt.subplots(10, samples_to_show, figsize=(2 * samples_to_show, 20))

    for c in range(10):
        c_indices = np.where(labels == c)[0][:samples_to_show]
        for j, idx in enumerate(c_indices):
            img = images[idx]
            ax = axes[c, j] if samples_to_show > 1 else axes[c]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Class {c}: {CLASS_NAMES[c]}", fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved sample verification image grid (2 per class) to '{out_path}'.")


# ==============================================================================
# 3. Model Architectures (Teacher, Student Base, TWN, LATe)
# ==============================================================================

# --- FP32 Architecture (Teacher & Student Base) ---
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


# --- TWN Modules (Standard Ternary Weight Networks) ---
def twn_quantize(w, delta_factor=0.75):
    with torch.no_grad():
        abs_w = w.abs()
        delta = delta_factor * abs_w.mean()
        mask = (abs_w > delta).to(w.dtype)
        num_nonzero = mask.sum()
        alpha = (abs_w * mask).sum() / num_nonzero.clamp(min=1.0)
        w_ternary = alpha * torch.sign(w) * mask
    return w + (w_ternary - w).detach()


class TWNConv2d(nn.Conv2d):
    def __init__(self, *args, delta_factor=0.75, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.delta_factor = delta_factor

    def forward(self, x):
        w_q = twn_quantize(self.weight, self.delta_factor)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TWNLinear(nn.Linear):
    def __init__(self, *args, delta_factor=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_factor = delta_factor

    def forward(self, x):
        w_q = twn_quantize(self.weight, self.delta_factor)
        return F.linear(x, w_q, self.bias)


class TWNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, delta_factor=0.75):
        super().__init__()
        self.conv1 = TWNConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               delta_factor=delta_factor)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = TWNConv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                               delta_factor=delta_factor)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                TWNConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride,
                          delta_factor=delta_factor),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class TWNResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, delta_factor=0.75,
                 quantize_first_last=False):
        super().__init__()
        self.in_planes = 64
        self.delta_factor = delta_factor

        if quantize_first_last:
            self.conv1 = TWNConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   delta_factor=delta_factor)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, delta_factor=delta_factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, delta_factor=delta_factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, delta_factor=delta_factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, delta_factor=delta_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if quantize_first_last:
            self.fc = TWNLinear(512 * block.expansion, num_classes, delta_factor=delta_factor)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, delta_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, delta_factor=delta_factor))
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


def TWNResNet18(num_classes=10, delta_factor=0.75, quantize_first_last=False):
    return TWNResNetCIFAR(TWNBasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                          delta_factor=delta_factor, quantize_first_last=quantize_first_last)


# --- LATe Modules (Loss-Aware Ternarization) ---
def late_quantize_exact(w, diag_curvature=None):
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
        alpha = p_k[best_k_idx] / d_k[best_k_idx]
        if torch.isnan(alpha) or torch.isinf(alpha) or alpha < 1e-5:
            alpha = abs_w.mean().clamp(min=1e-5)
        delta = 0.5 * alpha

        mask = (abs_w > delta).to(w_flat.dtype)
        if mask.sum() == 0:
            delta = 0.5 * abs_w.mean().clamp(min=1e-5)
            mask = (abs_w > delta).to(w_flat.dtype)
            alpha = (abs_w * mask).sum() / mask.sum().clamp(min=1.0)
            if torch.isnan(alpha) or alpha < 1e-5:
                alpha = abs_w.mean().clamp(min=1e-5)

        w_ternary = (alpha * torch.sign(w_flat) * mask).view_as(w)
    return w + (w_ternary - w).detach()


class LATeConv2d(nn.Conv2d):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))

    def forward(self, x):
        curv = torch.sqrt(self.diag_curvature + 1e-8)
        w_q = late_quantize_exact(self.weight, curv)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LATeLinear(nn.Linear):
    def __init__(self, *args, curvature_beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.curvature_beta = curvature_beta
        self.register_buffer("diag_curvature", torch.ones_like(self.weight.data))

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


# --- Direct Ternary Architecture (Hard-Projected) ---
def project_to_ternary_(weight, delta_factor=0.75):
    with torch.no_grad():
        abs_w = weight.abs()
        delta = delta_factor * abs_w.mean()
        mask = (abs_w > delta).to(weight.dtype)
        num_nonzero = mask.sum()
        alpha = (abs_w * mask).sum() / num_nonzero.clamp(min=1.0)
        w_ternary = alpha * torch.sign(weight) * mask
        weight.copy_(w_ternary)


class DirectTernaryConv2d(nn.Conv2d):
    def __init__(self, *args, delta_factor=0.75, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.delta_factor = delta_factor
        project_to_ternary_(self.weight.data, self.delta_factor)


class DirectTernaryLinear(nn.Linear):
    def __init__(self, *args, delta_factor=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_factor = delta_factor
        project_to_ternary_(self.weight.data, self.delta_factor)


class DirectTernaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, delta_factor=0.75):
        super().__init__()
        self.conv1 = DirectTernaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                         delta_factor=delta_factor)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DirectTernaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                                         delta_factor=delta_factor)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                DirectTernaryConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride,
                                    delta_factor=delta_factor),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class DirectTernaryResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, delta_factor=0.75,
                 quantize_first_last=False):
        super().__init__()
        self.in_planes = 64
        self.delta_factor = delta_factor

        if quantize_first_last:
            self.conv1 = DirectTernaryConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                             delta_factor=delta_factor)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, delta_factor=delta_factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, delta_factor=delta_factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, delta_factor=delta_factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, delta_factor=delta_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if quantize_first_last:
            self.fc = DirectTernaryLinear(512 * block.expansion, num_classes, delta_factor=delta_factor)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, delta_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, delta_factor=delta_factor))
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


def DirectTernaryResNet18(num_classes=10, delta_factor=0.75, quantize_first_last=False):
    return DirectTernaryResNetCIFAR(DirectTernaryBasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                                    delta_factor=delta_factor, quantize_first_last=quantize_first_last)


# ==============================================================================
# 4. Evaluation Function and Confusion Matrix Plotting
# ==============================================================================

def evaluate_on_unseen(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = 100.0 * (all_preds == all_targets).mean()

    # Per-class accuracy
    per_class_acc = {}
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    for c in range(10):
        total_c = (all_targets == c).sum()
        correct_c = cm[c, c]
        per_class_acc[CLASS_NAMES[c]] = 100.0 * correct_c / max(1, total_c)

    return acc, per_class_acc, cm


def plot_combined_confusion_matrices(cm_dict, out_path="unseen_confusion_matrices.png"):
    num_models = len(cm_dict)
    if num_models == 0:
        return

    cols = min(2, num_models)
    rows = (num_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))

    if num_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

    for extra in range(num_models, len(axes)):
        axes[extra].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined confusion matrices to '{out_path}'.")


# ==============================================================================
# 5. Main Execution Loop
# ==============================================================================

def resolve_checkpoint(path_str, fallback_paths=None):
    if path_str and os.path.exists(path_str):
        return path_str
    if fallback_paths:
        for p in fallback_paths:
            if p and os.path.exists(p):
                return p
    return path_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 Models on Unseen CIFAR-10.1 Dataset")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10_1",
                        help="Path to store/load CIFAR-10.1 data files.")
    parser.add_argument("--samples_per_class", type=int, default=200,
                        help="Number of images per class for test subset (default: 200 = 2,000 images, i.e., the full benchmark).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection (default: 42).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device ('cuda', 'cpu', 'mps').")

    # Single Model Evaluation Mode
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="Path to evaluate a single model checkpoint directly.")
    parser.add_argument("--model_type", type=str, default="twn",
                        choices=["twn", "late", "teacher", "fp32_student", "direct_ternary"],
                        help="Model type when using --model_ckpt (default: 'twn').")
    parser.add_argument("--quantize_first_last", action="store_true", default=False,
                        help="Enable first stem conv and final linear layer quantization.")

    # Batch Comparison Checkpoint paths
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to trained FP32 ResNet-34 teacher checkpoint.")
    parser.add_argument("--student_base_ckpt", type=str, default=None,
                        help="Path to trained FP32 ResNet-18 student baseline checkpoint.")
    parser.add_argument("--twn_student_ckpt", type=str, default=None,
                        help="Path to trained Best TWN ResNet-18 student checkpoint (e.g. baseline4_temp4_best.pth).")
    parser.add_argument("--late_student_ckpt", type=str, default=None,
                        help="Path to trained Best LATe ResNet-18 student checkpoint (e.g. late_baseline4_temp4_best.pth).")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Resolve default paths dynamically if not explicitly specified
    teacher_ckpt = resolve_checkpoint(args.teacher_ckpt or "teacher_best.pth", [
        "results/03_teacher_and_temp2_twn/teacher_best.pth",
        "teacher_best.pth"
    ])
    student_base_ckpt = resolve_checkpoint(args.student_base_ckpt or "student_base_best.pth", [
        "results/01_default_twn_temp4/student_base_best.pth",
        "student_base_best.pth"
    ])
    twn_student_ckpt = resolve_checkpoint(args.twn_student_ckpt or "baseline4_temp4_best.pth", [
        "results/01_default_twn_temp4/baseline4_temp4_best.pth",
        "baseline4_temp4_best.pth",
        "baseline4_best.pth"
    ])
    late_student_ckpt = resolve_checkpoint(args.late_student_ckpt or "late_baseline4_temp4_best.pth", [
        "results/05_late_loss_aware_ternarization/late_baseline4_temp4_best.pth",
        "late_baseline4_temp4_best.pth",
        "late_baseline4_best.pth"
    ])

    # 1. Load Unseen CIFAR-10.1 Subset
    subset_images, subset_labels = load_cifar10_1_subset(
        data_dir=args.data_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )

    # 2. Save 2-Image per Class Verification Grid
    plot_sample_images(subset_images, subset_labels, out_path="unseen_sample_images_2_per_class.png", samples_to_show=2)

    # 3. Create PyTorch DataLoader for Evaluation
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    dataset = CIFAR10_1_Dataset(subset_images, subset_labels, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 4. Single Model Evaluation Branch
    if args.model_ckpt:
        ckpt = resolve_checkpoint(args.model_ckpt, [
            os.path.join("results/01_default_twn_temp4", args.model_ckpt),
            os.path.join("results/05_late_loss_aware_ternarization", args.model_ckpt)
        ])
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint file '{ckpt}' not found!")
            sys.exit(1)

        if args.model_type == "twn":
            model = TWNResNet18(10, quantize_first_last=args.quantize_first_last).to(device)
            m_name = f"TWN ResNet-18 (fnl_ternary={args.quantize_first_last})"
        elif args.model_type == "late":
            model = LATeResNet18(10, quantize_first_last=args.quantize_first_last).to(device)
            m_name = f"LATe ResNet-18 (fnl_ternary={args.quantize_first_last})"
        elif args.model_type == "teacher":
            model = ResNet34(10).to(device)
            m_name = "FP32 ResNet-34 Teacher"
        elif args.model_type == "fp32_student":
            model = ResNet18(10).to(device)
            m_name = "FP32 ResNet-18 Student"
        elif args.model_type == "direct_ternary":
            model = DirectTernaryResNet18(10, quantize_first_last=args.quantize_first_last).to(device)
            m_name = f"Direct Ternary ResNet-18 (fnl_ternary={args.quantize_first_last})"
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")

        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        acc, per_class_acc, cm = evaluate_on_unseen(model, dataloader, device)

        print("\n" + "=" * 80)
        print(f"EVALUATION RESULT FOR {m_name.upper()} ON UNSEEN CIFAR-10.1 ({len(dataset)} IMAGES)")
        print("=" * 80)
        print(f"Overall Unseen Accuracy: {acc:.2f}% | Checkpoint: {ckpt}")
        print("-" * 80)
        print("Per-Class Accuracy Breakdown:")
        for c in range(10):
            print(f"  Class {c:1d} ({CLASS_NAMES[c]:12s}): {per_class_acc[CLASS_NAMES[c]]:6.2f}%")
        print("=" * 80)

        out_cm = f"unseen_cm_{args.model_type}_{os.path.basename(ckpt).replace('.pth', '')}.png"
        plot_combined_confusion_matrices({m_name: cm}, out_path=out_cm)
        return

    # 5. Multi-Model Batch Comparison Branch
    models_to_test = [
        {"name": "FP32 ResNet-34 Teacher",  "model_fn": lambda: ResNet34(10),     "ckpt": teacher_ckpt},
        {"name": "FP32 ResNet-18 Student",  "model_fn": lambda: ResNet18(10),     "ckpt": student_base_ckpt},
        {"name": "Best TWN ResNet-18 (KD)", "model_fn": lambda: TWNResNet18(10, quantize_first_last=args.quantize_first_last),  "ckpt": twn_student_ckpt},
        {"name": "Best LATe ResNet-18 (KD)","model_fn": lambda: LATeResNet18(10, quantize_first_last=args.quantize_first_last), "ckpt": late_student_ckpt},
    ]

    results_table = []
    cm_dict = {}

    print("\n" + "=" * 80)
    print(f"EVALUATION ON UNSEEN CIFAR-10.1 BENCHMARK ({len(dataset)} IMAGES TOTAL)")
    print("=" * 80)

    for item in models_to_test:
        name = item["name"]
        ckpt = item["ckpt"]

        if not os.path.exists(ckpt):
            print(f"[-] {name:30s} | Checkpoint NOT FOUND at '{ckpt}' (Skipping)")
            continue

        model = item["model_fn"]().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        acc, per_class_acc, cm = evaluate_on_unseen(model, dataloader, device)
        cm_dict[name] = cm
        results_table.append({"name": name, "acc": acc, "ckpt": ckpt, "per_class": per_class_acc})

        print(f"[+] {name:30s} | Unseen Accuracy: {acc:6.2f}% | Checkpoint: {ckpt}")

    if results_table:
        print("\n" + "=" * 80)
        print("CONSOLIDATED UNSEEN CIFAR-10.1 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Model Architecture':<30} | {'Unseen Accuracy (%)':<22} | {'Evaluated Checkpoint':<25}")
        print("-" * 80)
        for r in results_table:
            print(f"{r['name']:<30} | {r['acc']:<22.2f} | {r['ckpt']:<25}")
        print("=" * 80)

        # Plot Confusion Matrices
        plot_combined_confusion_matrices(cm_dict, out_path="unseen_confusion_matrices.png")

    else:
        print("\nNo checkpoints were found to evaluate. Please provide valid paths via CLI arguments.")


if __name__ == "__main__":
    main()
