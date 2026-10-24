# run_baselines.py — Architecture Baselines + Published Method Comparisons
# Run from: (agroonto) C:\Users\hasee\Desktop\AgroOnto-VLM>
# Command:  python run_baselines.py [--method B1|B2|B3|B4|P1|P2|P3|P4|P5] [--dataset PVT|DW|IP]
#
# ═══════════════════════════════════════════════════════════════════════
#  GROUP A — Architecture Baselines (all 3 datasets)
# ═══════════════════════════════════════════════════════════════════════
#   B1 = ResNet-50          (He et al., ImageNet pretrained)
#   B2 = Swin-T             (Liu et al., ImageNet pretrained)
#   B3 = ConvNeXt-B         (Liu et al., ImageNet pretrained)
#   B4 = EfficientNet-B7    (Tan & Le, ImageNet pretrained)
#
# ═══════════════════════════════════════════════════════════════════════
#  GROUP B — Published Method Comparisons (target dataset only)
# ═══════════════════════════════════════════════════════════════════════
#   P1 = DINOv2-B + LoRA    → DeepWeeds    (Espejo-Garcia et al. 2025)
#   P2 = Swin-AARNet        → IP102        (Wang et al. 2025)
#   P3 = FFM+MAM            → IP102        (Qian et al. 2025)
#   P4 = CT-PestNet          → IP102        (Li et al. 2026)
#   P5 = CNN-Stacking        → PVT          (Khan et al. 2024)
#
# References:
#   B1-B2: Wang et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1619551
#   B3:    Qian et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1500571
#   B4:    Wang et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1619551
#   P1:    Espejo-Garcia et al. (2025), Comput. Electron. Agric., DOI:10.1016/j.compag.2025.110900
#   P2:    Wang et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1619551
#   P3:    Qian et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1500571
#   P4:    Li et al. (2026), PLOS ONE, DOI:10.1371/journal.pone.0356061
#   P5:    Khan et al. (2024), Sci. Reports, DOI:10.1038/s41598-024-72237-x
#
# Note: Published methods P2-P3 use smaller backbones than original papers
#       (Swin-T instead of Swin-L, ConvNeXt-B instead of ConvNeXt-L) due to
#       hardware constraints (RTX 3050 Ti 4GB). The architectural contributions
#       (DWAblock/GSA, FFM/MAM, CFDE/ViT-LTI) remain identical to the papers.
#
# Same protocol as M1-M4: 30 epochs, 3-epoch warmup, cosine LR, fp16, AdamW
# Same metrics: accuracy, macro-F1, severity (ontology-based)
# Same datasets & splits: PlantVillageTomato, DeepWeeds, IP102

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from torchvision import transforms

# -- Project imports --------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Desktop", "AgroOnto-VLM"))

from agroonto import config
from agroonto.dataset import (
    PlantVillageDataset,
    DeepWeedsDataset,
    IP102Dataset,
)
from agroonto.ontology import (
    DEEPWEEDS_TAXONOMY,
    IP102_TAXONOMY,
    FACET_WEIGHTS,
    NORM_FACTOR,
    _parse_plantvillage_class,
)

# -- Settings ---------------------------------------------------------------
EPOCHS = 30
WARMUP_EPOCHS = 3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "AgroOnto-VLM")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Batch sizes -- RTX 3050 Ti 4GB VRAM with fp16
BATCH_SIZES = {
    "B1": 16,   # ResNet-50:       ~25M params
    "B2": 16,   # Swin-T:          ~28M params
    "B3": 8,    # ConvNeXt-B:      ~89M params
    "B4": 8,    # EfficientNet-B7: ~66M params (larger input)
    "P1": 16,   # DINOv2-B+LoRA:   ~86M but only LoRA trainable
    "P2": 12,   # Swin-AARNet:     Swin-T + DWA + GSA (~30M)
    "P3": 4,    # FFM+MAM:         ConvNeXt-B + FFM + MAM (~95M)
    "P4": 8,    # CT-PestNet:      CFDE + ViT-LTI (custom, ~25M)
    "P5": 16,   # CNN-Stacking:    4-layer CNN + sklearn stacking (~1M CNN)
}

# Learning rates
LR_FINETUNE = 1e-4
LR_LINEAR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Dataset name mapping
DS_MAP = {
    "PVT": "PlantVillageTomato",
    "DW": "DeepWeeds",
    "IP": "IP102",
    "PlantVillageTomato": "PlantVillageTomato",
    "DeepWeeds": "DeepWeeds",
    "IP102": "IP102",
}
ALL_DATASETS = ["PlantVillageTomato", "DeepWeeds", "IP102"]

# Group A: architecture baselines (run on ALL datasets)
GROUP_A = ["B1", "B2", "B3", "B4"]

# Group B: published methods (run only on target dataset)
GROUP_B = ["P1", "P2", "P3", "P4", "P5"]

# Published method -> target datasets
PUBLISHED_TARGET = {
    "P1": ["DeepWeeds"],
    "P2": ["IP102"],
    "P3": ["IP102"],
    "P4": ["IP102"],
    "P5": ["PlantVillageTomato"],
}

ALL_METHODS = GROUP_A + GROUP_B

METHOD_NAMES = {
    "B1": "B1_ResNet50",
    "B2": "B2_SwinT",
    "B3": "B3_ConvNeXtB",
    "B4": "B4_EfficientNetB7",
    "P1": "P1_DINOv2_LoRA",
    "P2": "P2_SwinAARNet",
    "P3": "P3_FFM_MAM",
    "P4": "P4_CTPestNet",
    "P5": "P5_CNNStacking",
}


# ###########################################################################
#  Utilities
# ###########################################################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_scheduler_with_warmup(optimizer, total_epochs, warmup_epochs):
    """Linear warmup + cosine decay (matches M1-M4 protocol)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cos_epoch = epoch - warmup_epochs
        cos_total = total_epochs - warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * cos_epoch / max(cos_total, 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _ensure_timm():
    """Import timm, installing if needed."""
    try:
        import timm
        return timm
    except ImportError:
        print("  Installing timm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "timm", "--break-system-packages", "-q"])
        import timm
        return timm


# ###########################################################################
#  Dataloaders
# ###########################################################################

def get_imagenet_transforms(img_size=224):
    """Standard ImageNet train/val transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_dinov2_transforms(img_size=224):
    """DINOv2-compatible transforms (ImageNet norm, center crop)."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_cnn_stacking_transforms(img_size=256):
    """CNN-Stacking transforms: 256×256, rescaled [0,1] (no ImageNet norm).
    Matches Khan et al. (2024) preprocessing."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),  # scales to [0,1]
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # scales to [0,1]
    ])
    return train_transform, val_transform


def build_baseline_loaders(dataset_name, train_transform, val_transform,
                           batch_size=16):
    """Build dataloaders using project dataset classes with given transforms."""
    random.seed(SEED)
    np.random.seed(SEED)

    if dataset_name == "PlantVillageTomato":
        train_dir = os.path.join(config.PLANTVILLAGE_DIR, "train")
        all_classes = sorted(os.listdir(train_dir))
        tomato_classes = [c for c in all_classes if c.startswith("Tomato")]
        exclude = set(all_classes) - set(tomato_classes)
        train_ds = PlantVillageDataset(
            config.PLANTVILLAGE_DIR, "train", train_transform,
            exclude_classes=exclude)
        val_ds = PlantVillageDataset(
            config.PLANTVILLAGE_DIR, "val", val_transform,
            exclude_classes=exclude)
        class_names = train_ds.classes
        num_classes = len(class_names)
        print(f"  PlantVillageTomato: {num_classes} classes (no holdout)")

    elif dataset_name == "DeepWeeds":
        train_csv = os.path.join(config.DEEPWEEDS_LABEL_DIR, "train_subset0.csv")
        val_csv = os.path.join(config.DEEPWEEDS_LABEL_DIR, "val_subset0.csv")
        train_ds = DeepWeedsDataset(
            config.DEEPWEEDS_IMG_DIR, train_csv, train_transform)
        val_ds = DeepWeedsDataset(
            config.DEEPWEEDS_IMG_DIR, val_csv, val_transform)
        class_names = [DEEPWEEDS_TAXONOMY[i]["species"] for i in range(9)]
        num_classes = 9
        print(f"  DeepWeeds: {num_classes} classes (no holdout)")

    elif dataset_name == "IP102":
        train_ds = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR, "train"),
            os.path.join(config.IP102_LABEL_DIR, "train"),
            train_transform)
        val_ds = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR, "val"),
            os.path.join(config.IP102_LABEL_DIR, "val"),
            val_transform)
        class_names = [f"pest_{i}" for i in range(102)]
        num_classes = 102
        print(f"  IP102: {num_classes} classes (no holdout)")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, class_names, num_classes


# ###########################################################################
#  Ontology severity computation (identical to evaluate_metrics.py)
# ###########################################################################

def compute_raw_similarity_matrix(dataset_name, class_names=None,
                                   num_classes=None):
    """Build K x K RAW ontology similarity matrix. severity(i->j) = 1 - S_raw(i,j)."""
    if dataset_name in ("PlantVillage", "PlantVillageTomato"):
        assert class_names is not None
        K = len(class_names)
        attrs = [_parse_plantvillage_class(n) for n in class_names]
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                score = 0.0
                if attrs[i]["crop"] == attrs[j]["crop"]:
                    score += FACET_WEIGHTS["crop"]
                if (attrs[i]["pathogen_type"] == attrs[j]["pathogen_type"]
                        and attrs[i]["pathogen_type"] != "healthy"):
                    score += FACET_WEIGHTS["pathogen_type"]
                if (attrs[i]["symptom"] == attrs[j]["symptom"]
                        and attrs[i]["symptom"] != "none"):
                    score += FACET_WEIGHTS["symptom"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)

    elif dataset_name == "DeepWeeds":
        K = num_classes or 9
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                ti = DEEPWEEDS_TAXONOMY.get(i, {})
                tj = DEEPWEEDS_TAXONOMY.get(j, {})
                if ti.get("genus") == "none" or tj.get("genus") == "none":
                    S[i, j] = 0.0
                    continue
                score = 0.0
                if ti["genus"] == tj["genus"]:
                    score += FACET_WEIGHTS["genus"]
                elif ti["family"] == tj["family"]:
                    score += FACET_WEIGHTS["family"]
                elif ti["order"] == tj["order"]:
                    score += FACET_WEIGHTS["order"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)

    elif dataset_name == "IP102":
        K = num_classes or 102
        S = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i == j:
                    S[i, j] = 1.0
                    continue
                ti = IP102_TAXONOMY.get(i, {})
                tj = IP102_TAXONOMY.get(j, {})
                if not ti or not tj:
                    continue
                score = 0.0
                if ti["genus"] == tj["genus"]:
                    score += FACET_WEIGHTS["genus"]
                elif ti["family"] == tj["family"]:
                    score += FACET_WEIGHTS["family"]
                elif ti["order"] == tj["order"]:
                    score += FACET_WEIGHTS["order"]
                S[i, j] = min(score / NORM_FACTOR, 1.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return S


def compute_metrics(all_preds, all_labels, num_classes, raw_sim):
    """Compute accuracy, macro-F1, severity from predictions."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, l in zip(all_preds, all_labels):
        cm[l, p] += 1

    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(round(f1, 4))
        per_class_precision.append(round(prec, 4))
        per_class_recall.append(round(rec, 4))

    macro_f1 = np.mean(per_class_f1)

    severities = []
    for p, l in zip(all_preds, all_labels):
        if p != l:
            sev = 1.0 - raw_sim[l, p]
            severities.append(sev)

    num_errors = len(severities)
    if num_errors > 0:
        severity_mean = float(np.mean(severities))
        severity_std = float(np.std(severities))
    else:
        severity_mean = 0.0
        severity_std = 0.0

    return {
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "macro_f1": round(float(macro_f1) * 100, 2),
        "accuracy": round(100.0 * np.trace(cm) / cm.sum(), 2),
        "severity_mean": round(severity_mean, 4),
        "severity_std": round(severity_std, 4),
        "num_errors": num_errors,
        "num_total": len(all_preds),
    }


# ###########################################################################
#  GROUP A — Architecture Baseline Models
# ###########################################################################

def build_resnet50(num_classes):
    """B1: ResNet-50 pretrained on ImageNet, fine-tune all layers."""
    import torchvision.models as models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_swin_t(num_classes):
    """B2: Swin-T pretrained on ImageNet, fine-tune all layers."""
    timm = _ensure_timm()
    model = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=True, num_classes=num_classes)
    return model


def build_convnext_b(num_classes):
    """B3: ConvNeXt-B pretrained on ImageNet, fine-tune all layers."""
    timm = _ensure_timm()
    model = timm.create_model("convnext_base.fb_in22k_ft_in1k",
                              pretrained=True, num_classes=num_classes)
    return model


def build_efficientnet_b7(num_classes):
    """B4: EfficientNet-B7 pretrained on ImageNet, fine-tune all layers.
    Ref: Wang et al. (2025) use EfficientNet-B7 as architecture baseline.
    Note: We use 224x224 input (not B7's default 600x600) for consistency
    with other baselines and to fit in 4GB VRAM."""
    timm = _ensure_timm()
    model = timm.create_model("tf_efficientnet_b7.ra_in1k",
                              pretrained=True, num_classes=num_classes)
    return model


# ###########################################################################
#  GROUP B — Published Method Implementations
# ###########################################################################

# ── P1: DINOv2-B + LoRA (Espejo-Garcia et al. 2025) ──────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for transformer attention projections."""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


class DINOv2WithLoRA(nn.Module):
    """DINOv2-B with LoRA on the last 4 transformer blocks + linear head.
    Ref: Espejo-Garcia et al. (2025), Comput. Electron. Agric."""
    def __init__(self, num_classes, rank=8, alpha=16, lora_blocks=None):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                                       pretrained=True)
        embed_dim = self.backbone.embed_dim  # 768

        for p in self.backbone.parameters():
            p.requires_grad = False

        if lora_blocks is None:
            lora_blocks = [8, 9, 10, 11]

        for block_idx in lora_blocks:
            block = self.backbone.blocks[block_idx]
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        logits = self.classifier(cls_token)
        return logits


def build_dinov2_lora(num_classes, rank=8, alpha=16):
    """P1: DINOv2-B + LoRA on last 4 blocks."""
    return DINOv2WithLoRA(num_classes, rank=rank, alpha=alpha)


# ── P2: Swin-AARNet (Wang et al. 2025) ───────────────────────────────
# Architecture: Swin Transformer + DWAblock + GSA + MLP
# Original uses Swin-L (~195M); we use Swin-T (~28M) for 4GB VRAM.
# Architectural contributions (DWAblock, GSA) are identical to the paper.
# Ref: Wang et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1619551

class DSWC(nn.Module):
    """Depth-wise Separable Weighted Convolution branch of DWAblock.
    Grouped depthwise conv -> dilated conv (dilation 3) -> channel reorg
    -> attention-weighted modulation."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                      padding=3, dilation=3, groups=channels,
                                      bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        # Attention branch for modulation weights
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.dilated_conv(out)
        out = self.pw_conv(out)
        out = self.bn(out)
        # Channel attention modulation
        w = self.attn_pool(out).flatten(1)
        w = self.attn_fc(w).unsqueeze(-1).unsqueeze(-1)
        return out * w


class DSEC(nn.Module):
    """Depth-wise Separable Excitation Convolution branch of DWAblock.
    Similar to DSWC but uses residual connection only (no modulation)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                      padding=3, dilation=3, groups=channels,
                                      bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.dilated_conv(out)
        out = self.pw_conv(out)
        out = self.bn(out)
        return out + x  # residual


class DWAblock(nn.Module):
    """Depth-wise separable residual Attention block (Wang et al. 2025).
    FC reduce -> split to DSEC + DSWC -> concat -> FC restore.
    Learnable alpha/beta scale the two branches."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // 2
        self.reduce = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
        )
        self.dswc = DSWC(mid, reduction=reduction)
        self.dsec = DSEC(mid, reduction=reduction)
        self.restore = nn.Linear(mid * 2, channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Reduce along channel dim
        x_flat = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_mid = self.reduce(x_flat)  # (B, H, W, C//2)
        x_mid = x_mid.permute(0, 3, 1, 2).contiguous()  # (B, C//2, H, W)
        # Two branches
        out_dswc = self.dswc(x_mid) * self.alpha
        out_dsec = self.dsec(x_mid) * self.beta
        # Concat and restore
        combined = torch.cat([out_dswc, out_dsec], dim=1)  # (B, C, H, W)
        combined = combined.permute(0, 2, 3, 1).contiguous()
        out = self.restore(combined)  # (B, H, W, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return out + x  # residual


class GSA(nn.Module):
    """Global Spatial Attention (Wang et al. 2025).
    7x7 depthwise conv -> channel reduction -> ReLU -> depthwise conv
    -> 7x7 conv restoring channels -> spatial attention weights."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=7, padding=3,
                                   groups=channels, bias=False)
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv2 = nn.Conv2d(mid, mid, kernel_size=7, padding=3,
                                   groups=mid, bias=False)
        self.restore = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.dw_conv1(x)
        attn = self.reduce(attn)
        attn = self.relu(attn)
        attn = self.dw_conv2(attn)
        attn = self.restore(attn)
        attn = self.sigmoid(attn)
        return x * attn


class SwinAARNet(nn.Module):
    """Swin Attention Augmented Residual Network (Wang et al. 2025).
    Swin-T backbone -> DWAblock -> GSA -> GAP -> MLP classifier.
    Original paper uses Swin-L; we use Swin-T for 4GB VRAM constraint."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        timm = _ensure_timm()
        # Swin-T backbone (outputs features, no classifier)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,        # remove classifier
            global_pool="",       # no global pool
        )
        # Swin-T final feature dim = 768, spatial = 7x7
        feat_dim = 768

        self.dwa_block = DWAblock(feat_dim, reduction=4)
        self.gsa = GSA(feat_dim, reduction=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        # Swin-T forward_features returns (B, H*W, C) or (B, C) depending on version
        feat = self.backbone.forward_features(x)
        # timm Swin: feat is (B, 7*7, 768) for 224x224 input
        if feat.dim() == 3:
            B, N, C = feat.shape
            h = w = int(math.sqrt(N))
            feat = feat.permute(0, 2, 1).reshape(B, C, h, w)
        # Now feat: (B, 768, 7, 7)
        feat = self.dwa_block(feat)
        feat = self.gsa(feat)
        feat = self.pool(feat).flatten(1)  # (B, 768)
        logits = self.classifier(feat)
        return logits


def build_swin_aarnet(num_classes):
    """P2: Swin-AARNet (Wang et al. 2025)."""
    return SwinAARNet(num_classes, pretrained=True)


# ── P3: FFM+MAM (Qian et al. 2025) ──────────────────────────────────
# Architecture: ConvNeXt backbone + parallel FFM + MAM branches -> fusion -> MLP
# Original uses ConvNeXt-L (~197M); we use ConvNeXt-B (~89M) for 4GB VRAM.
# Ref: Qian et al. (2025), Front. Plant Sci., DOI:10.3389/fpls.2025.1500571

class MixedAttentionModule(nn.Module):
    """MAM: Mixed Attention Module (Qian et al. 2025).
    SE-like design but replaces FC excitation with self-attention.
    GAP squeeze -> Q,K,V projections -> scaled dot-product attention
    -> sigmoid -> channel-wise rescaling."""
    def __init__(self, channels, dk=384, num_chunks=4):
        super().__init__()
        self.channels = channels
        self.dk = dk
        self.num_chunks = num_chunks
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Self-attention projections (replace SE's FC layers)
        self.q_proj = nn.Linear(channels, dk)
        self.k_proj = nn.Linear(channels, dk)
        self.v_proj = nn.Linear(channels, dk)
        # Chunk-wise linear transforms
        chunk_dim = dk // num_chunks
        self.chunk_fcs = nn.ModuleList([
            nn.Linear(chunk_dim, chunk_dim) for _ in range(num_chunks)
        ])
        self.out_proj = nn.Linear(dk, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # Squeeze: GAP
        squeezed = self.pool(x).flatten(1)  # (B, C)
        # Self-attention excitation
        q = self.q_proj(squeezed)  # (B, dk)
        k = self.k_proj(squeezed)
        v = self.v_proj(squeezed)
        # Scaled dot-product (single-vector attention as channel weighting)
        scale = self.dk ** 0.5
        attn = (q * k) / scale  # element-wise for single vector
        attn = F.softmax(attn, dim=-1)
        out = attn * v  # (B, dk)
        # Chunk-wise transform
        chunk_dim = self.dk // self.num_chunks
        chunks = out.split(chunk_dim, dim=-1)
        transformed = [fc(chunk) for fc, chunk in zip(self.chunk_fcs, chunks)]
        out = torch.cat(transformed, dim=-1)  # (B, dk)
        # Project back to channel dimension
        weights = self.sigmoid(self.out_proj(out))  # (B, C)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weights


class FeatureFusionModule(nn.Module):
    """FFM: Feature Fusion Module (Qian et al. 2025).
    Adaptive filtering: image patches -> conv projection -> class token + pos embed
    -> Transformer blocks (self-attention + MLP).
    Then fuses with MAM output via concatenation and projection."""
    def __init__(self, img_size=224, patch_size=32, embed_dim=768,
                 num_heads=8, num_layers=2, backbone_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 49 for 224/32=7

        # Adaptive filtering: patch projection
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                     stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(0.1)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Fusion projection: combine FFM CLS output + MAM-weighted features
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim + backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, images, mam_features):
        """
        images: (B, 3, H, W) original images
        mam_features: (B, backbone_dim) GAP-pooled MAM output
        Returns: (B, backbone_dim) fused features
        """
        # Patch embedding
        x = self.patch_embed(images)  # (B, embed_dim, H/P, W/P)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, embed_dim) -- CLS token output

        # Fusion with MAM features
        fused = torch.cat([cls_out, mam_features], dim=1)
        fused = self.fusion_proj(fused)  # (B, backbone_dim)
        return fused


class FFMMAMNet(nn.Module):
    """ConvNeXt-B + FFM + MAM (Qian et al. 2025).
    Parallel dual-branch: FFM on image patches + MAM on backbone features
    -> fusion -> MLP classifier.
    Original uses ConvNeXt-L; we use ConvNeXt-B for 4GB VRAM."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        timm = _ensure_timm()
        # ConvNeXt-B backbone (remove classifier, keep features)
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        backbone_dim = 1024  # ConvNeXt-B final channel dim

        # MAM branch: operates on backbone feature maps
        self.mam = MixedAttentionModule(backbone_dim, dk=384, num_chunks=4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # FFM branch: operates on original image patches
        self.ffm = FeatureFusionModule(
            img_size=224, patch_size=32, embed_dim=768,
            num_heads=8, num_layers=2, backbone_dim=backbone_dim,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, num_classes),
        )

    def forward(self, x):
        images = x  # save for FFM branch

        # Backbone features
        feat = self.backbone.forward_features(x)  # (B, C, H, W) or (B, H*W, C)
        if feat.dim() == 3:
            B, N, C = feat.shape
            h = w = int(math.sqrt(N))
            feat = feat.permute(0, 2, 1).reshape(B, C, h, w)

        # MAM branch: channel attention on backbone features
        mam_feat = self.mam(feat)  # (B, C, H, W)
        mam_pooled = self.gap(mam_feat).flatten(1)  # (B, C)

        # FFM branch: patch-level + fusion with MAM
        fused = self.ffm(images, mam_pooled)  # (B, backbone_dim)

        logits = self.classifier(fused)
        return logits


def build_ffm_mam(num_classes):
    """P3: FFM+MAM (Qian et al. 2025)."""
    return FFMMAMNet(num_classes, pretrained=True)


# ── P4: CT-PestNet (Li et al. 2026) ──────────────────────────────────
# Architecture: CFDE (cross-scale feature distillation) + ViT-LTI
# Ref: Li et al. (2026), PLOS ONE, DOI:10.1371/journal.pone.0356061

class CFDE(nn.Module):
    """Cross-scale Feature Distillation Enhancement (Li et al. 2026).
    3 parallel conv branches (kernels 7,5,3) -> differential distillation
    -> concatenate -> SE block for channel recalibration."""
    def __init__(self, in_channels=3, base_channels=64, se_reduction=4):
        super().__init__()
        # Three parallel conv branches at different scales
        self.branch_large = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3,
                      stride=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2,
                      stride=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.branch_small = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # After differential distillation: D1 + D2 + F_large = 3 * base_channels
        combined_channels = base_channels * 3

        # SE block for channel recalibration
        mid = combined_channels // se_reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(combined_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, combined_channels),
            nn.Sigmoid(),
        )
        self.out_channels = combined_channels

    def forward(self, x):
        # Three scales
        f_large = self.branch_large(x)    # (B, C, H/2, W/2)
        f_medium = self.branch_medium(x)  # (B, C, H/2, W/2)
        f_small = self.branch_small(x)    # (B, C, H/2, W/2)

        # Differential distillation: keep only newly added info at each scale
        d1 = f_medium - f_small    # medium minus small
        d2 = f_large - f_medium    # large minus medium

        # Concatenate: distilled features + largest scale
        combined = torch.cat([d1, d2, f_large], dim=1)  # (B, 3C, H/2, W/2)

        # SE recalibration
        B, C_total = combined.shape[:2]
        weights = self.se(combined).unsqueeze(-1).unsqueeze(-1)
        out = combined * weights
        return out


class LTIR(nn.Module):
    """Local Token Interaction Reshape (Li et al. 2026).
    Reshapes tokens to 2D spatial map, applies multi-scale grouped convolutions,
    then reshapes back to token sequence."""
    def __init__(self, embed_dim, spatial_size=3):
        super().__init__()
        self.spatial_size = spatial_size
        group_dim = embed_dim // 3
        remainder = embed_dim - group_dim * 3

        self.conv_small = nn.Conv2d(group_dim, group_dim, kernel_size=1,
                                     groups=1, bias=False)
        self.conv_medium = nn.Conv2d(group_dim, group_dim, kernel_size=3,
                                      padding=1, groups=1, bias=False)
        g3_dim = group_dim + remainder
        self.conv_large = nn.Conv2d(g3_dim, g3_dim, kernel_size=3,
                                     padding=1, groups=1, bias=False)
        self.group_dims = [group_dim, group_dim, g3_dim]

        # Channel reduction branch
        self.reduce_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)

    def forward(self, x, num_tokens):
        """x: (B, N, D) patch tokens (excluding CLS).
        Returns (B, N, D)."""
        B, N, D = x.shape
        h = w = self.spatial_size
        # Reshape to 2D
        x_2d = x.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)

        # Split into 3 channel groups
        g1, g2, g3 = x_2d.split(self.group_dims, dim=1)
        g1 = self.conv_small(g1)
        g2 = self.conv_medium(g2)
        g3 = self.conv_large(g3)

        # Reduced channel branch
        x_reduced = self.reduce_conv(x_2d)

        # Combine
        out = torch.cat([g1, g2, g3], dim=1) + x_reduced
        out = self.bn(out)
        out = out.flatten(2).transpose(1, 2)  # (B, N, D)
        return out


class ViTLTI(nn.Module):
    """Vision Transformer with Local Token Interaction (Li et al. 2026).
    Tokenizes CFDE output into patches, applies transformer encoder with LTIR."""
    def __init__(self, in_channels, num_classes, img_size=112,
                 patch_grid=3, embed_dim=1024, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_grid = patch_grid
        num_patches = patch_grid * patch_grid  # 9
        patch_size = img_size // patch_grid     # ~37

        # Linear projection of flattened patches
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)

        # Positional embeddings + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder layers with LTIR
        self.layers = nn.ModuleList()
        self.ltir_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ))
            self.ltir_layers.append(LTIR(embed_dim, spatial_size=patch_grid))

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """x: (B, C, H, W) from CFDE output."""
        B, C, H, W = x.shape
        # Split into non-overlapping patches
        ph = pw = self.patch_size
        # Pad if needed
        if H % self.patch_grid != 0:
            pad_h = self.patch_grid - (H % self.patch_grid)
            x = F.pad(x, (0, 0, 0, pad_h))
            H = H + pad_h
        if W % self.patch_grid != 0:
            pad_w = self.patch_grid - (W % self.patch_grid)
            x = F.pad(x, (0, pad_w, 0, 0))
            W = W + pad_w
        ph = H // self.patch_grid
        pw = W // self.patch_grid

        # Unfold into patches: (B, num_patches, C*ph*pw)
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # patches: (B, C, grid_h, grid_w, ph, pw)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)
        # (B, grid_h, grid_w, C, ph, pw)
        num_patches = self.patch_grid * self.patch_grid
        patches = patches.reshape(B, num_patches, -1)  # (B, 9, C*ph*pw)

        # Linear projection
        tokens = self.proj(patches)  # (B, 9, embed_dim)

        # Prepend CLS token + positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 10, D)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        tokens = self.pos_drop(tokens)

        # Transformer layers with LTIR
        for layer, ltir in zip(self.layers, self.ltir_layers):
            tokens = layer(tokens)
            # LTIR on patch tokens only (exclude CLS)
            cls_tok = tokens[:, :1]
            patch_tok = tokens[:, 1:]
            patch_tok = ltir(patch_tok, num_patches)
            tokens = torch.cat([cls_tok, patch_tok], dim=1)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]  # (B, embed_dim)
        logits = self.classifier(cls_out)
        return logits


class CTPestNet(nn.Module):
    """CT-PestNet (Li et al. 2026): CFDE -> ViT-LTI -> classifier.
    Cross-scale Feature Distillation Enhancement + ViT with Local Token
    Interaction for fine-grained pest classification.
    Ref: Li et al. (2026), PLOS ONE, DOI:10.1371/journal.pone.0356061"""
    def __init__(self, num_classes, base_channels=64, embed_dim=1024,
                 num_heads=8, num_layers=6):
        super().__init__()
        self.cfde = CFDE(in_channels=3, base_channels=base_channels,
                          se_reduction=4)
        cfde_out_channels = self.cfde.out_channels  # 3 * base_channels = 192
        # CFDE outputs at H/2 x W/2 = 112x112 for 224 input
        self.vit_lti = ViTLTI(
            in_channels=cfde_out_channels,
            num_classes=num_classes,
            img_size=112,
            patch_grid=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, x):
        feat = self.cfde(x)       # (B, 192, 112, 112)
        logits = self.vit_lti(feat)
        return logits


def build_ct_pestnet(num_classes):
    """P4: CT-PestNet (Li et al. 2026)."""
    return CTPestNet(num_classes, base_channels=64, embed_dim=1024,
                     num_heads=8, num_layers=6)


# ###########################################################################
#  P5: CNN-Stacking (Khan et al. 2024)
# ###########################################################################
# "Bayesian optimized multimodal deep hybrid learning approach for
#  tomato leaf disease classification"
# Sci. Reports, DOI:10.1038/s41598-024-72237-x
#
# Architecture: 4-layer CNN feature extractor → Boruta feature selection
# → Stacking ensemble (RF, XGBoost, GNB, SVM, LR, KNN as base learners,
#   Logistic Regression meta-learner).
# Input: 256×256 RGB, rescaled [0,1].
# This is a CNN+sklearn hybrid, NOT end-to-end deep learning.

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor from Khan et al. (2024).
    4 Conv2D layers: 32, 64, 16, 64 filters.
    Kernel 2×2 except 3rd layer 3×3; same padding, stride 1.
    BN + ReLU + MaxPool(2) after each conv block.
    Final: Flatten → Dropout → Dense(128)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32 filters, 2×2 kernel
            nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=0),  # "same" approx
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64 filters, 2×2 kernel
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16 filters, 3×3 kernel (paper specifies 3×3 for this one)
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 64 filters, 2×2 kernel
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Adaptive pool to fixed size so feature dim is deterministic
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(64 * 4 * 4, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.dense(x))
        return x


def _ensure_sklearn_deps():
    """Ensure scikit-learn and xgboost are available."""
    missing = []
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    try:
        import xgboost
    except ImportError:
        missing.append("xgboost")
    try:
        from boruta import BorutaPy
    except ImportError:
        missing.append("boruta")
    if missing:
        import subprocess
        print(f"  Installing {', '.join(missing)}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing +
            ["--break-system-packages", "-q"])


def train_cnn_stacking(method, dataset_name):
    """Train CNN-Stacking (Khan et al. 2024): CNN feature extraction → Boruta
    → Stacking ensemble. Returns metrics dict.

    This is a two-stage pipeline:
    Stage 1: Train CNN feature extractor with CE loss (same 30-epoch protocol)
    Stage 2: Extract features → Boruta selection → fit stacking ensemble
    Evaluation: stacking predictions on test features."""

    print(f"\n{'='*70}")
    print(f"  GROUP B (Published Method)")
    print(f"  TRAIN {METHOD_NAMES[method]}  |  {dataset_name}")
    print(f"  >>> Two-stage CNN + Stacking Ensemble pipeline <<<")
    print(f"{'='*70}")

    _ensure_sklearn_deps()
    from sklearn.ensemble import (RandomForestClassifier,
                                  StackingClassifier,
                                  GradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False
        print("  WARNING: xgboost not available, using GradientBoosting instead")
    try:
        from boruta import BorutaPy
        has_boruta = True
    except ImportError:
        has_boruta = False
        print("  WARNING: Boruta not available, skipping feature selection")

    set_seed(SEED)
    batch_size = BATCH_SIZES[method]

    # Use CNN-Stacking specific transforms: 256×256, [0,1] no ImageNet norm
    train_tf, val_tf = get_cnn_stacking_transforms(256)

    train_loader, val_loader, class_names, num_classes = build_baseline_loaders(
        dataset_name, train_tf, val_tf, batch_size=batch_size)

    # ── Stage 1: Train CNN feature extractor ──
    print(f"\n  ── Stage 1: Training CNN Feature Extractor ──")
    cnn = CNNFeatureExtractor().to(DEVICE)

    # Add a temporary classification head for training
    classifier_head = nn.Linear(128, num_classes).to(DEVICE)

    trainable = sum(p.numel() for p in cnn.parameters()) + \
                sum(p.numel() for p in classifier_head.parameters())
    print(f"  CNN params:  {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"  Head params: {sum(p.numel() for p in classifier_head.parameters()):,}")
    print(f"  Trainable:   {trainable:,}")

    all_params = list(cnn.parameters()) + list(classifier_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR_FINETUNE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)
    scaler = GradScaler()

    best_val = 0.0
    best_epoch = 0
    best_cnn_state = None
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        cnn.train()
        classifier_head.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                feats = cnn(images)
                logits = classifier_head(feats)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        cnn.eval()
        classifier_head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    feats = cnn(images)
                    logits = classifier_head(feats)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_cnn_state = copy.deepcopy(
                {k: v.cpu() for k, v in cnn.state_dict().items()})

        marker = "  *" if val_acc >= best_val else ""
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"[{elapsed:.0f}s]{marker}")

    print(f"  CNN best: {best_val:.2f}% (epoch {best_epoch})")

    # Save CNN checkpoint
    ckpt_path = os.path.join(
        CHECKPOINT_DIR, f"{method}_{dataset_name}_seed{SEED}_cnn.pt")
    if best_cnn_state:
        torch.save(best_cnn_state, ckpt_path)
        print(f"  OK CNN checkpoint: {ckpt_path}")

    # ── Stage 2: Extract features → Boruta → Stacking ──
    print(f"\n  ── Stage 2: Feature Extraction + Stacking Ensemble ──")

    cnn.load_state_dict(best_cnn_state)
    cnn = cnn.to(DEVICE)
    cnn.eval()

    def extract_features(loader):
        all_feats, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                with autocast():
                    feats = cnn(images)
                all_feats.append(feats.cpu().float().numpy())
                all_labels.extend(labels.numpy())
        return np.concatenate(all_feats, axis=0), np.array(all_labels)

    print("  Extracting train features...")
    X_train, y_train = extract_features(train_loader)
    print(f"    Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    print("  Extracting val features...")
    X_val, y_val = extract_features(val_loader)
    print(f"    Val:   {X_val.shape[0]} samples, {X_val.shape[1]} features")

    # Boruta feature selection
    if has_boruta:
        print("  Running Boruta feature selection...")
        rf_boruta = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                           random_state=SEED)
        boruta = BorutaPy(rf_boruta, n_estimators='auto', random_state=SEED,
                          max_iter=50)
        boruta.fit(X_train, y_train)
        n_selected = boruta.support_.sum()
        print(f"    Boruta selected {n_selected}/{X_train.shape[1]} features")
        if n_selected > 10:  # Only apply if reasonable number selected
            X_train = X_train[:, boruta.support_]
            X_val = X_val[:, boruta.support_]
        else:
            print(f"    Too few features selected, using all {X_train.shape[1]}")
    else:
        print("  Skipping Boruta (not installed)")

    # Build stacking ensemble
    print("  Building Stacking ensemble...")
    if has_xgb:
        xgb_clf = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=-1)
    else:
        xgb_clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED)

    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=SEED,
                                      n_jobs=-1)),
        ('xgb', xgb_clf),
        ('gnb', GaussianNB()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=SEED)),
        ('lr', LogisticRegression(max_iter=1000, random_state=SEED,
                                  n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=5,
        n_jobs=-1,
    )

    print("  Fitting stacking ensemble (this may take a few minutes)...")
    fit_start = time.time()
    stacking.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    print(f"  Stacking fit time: {fit_time:.1f}s")

    # Save stacking model
    import pickle
    stack_path = os.path.join(
        CHECKPOINT_DIR, f"{method}_{dataset_name}_seed{SEED}_stacking.pkl")
    with open(stack_path, 'wb') as f:
        pickle.dump(stacking, f)
    print(f"  OK Stacking checkpoint: {stack_path}")

    # ── Evaluate ──
    print(f"\n  Computing metrics...")
    all_preds = stacking.predict(X_val)
    all_labels = y_val

    raw_sim = compute_raw_similarity_matrix(
        dataset_name, class_names=class_names, num_classes=num_classes)
    metrics = compute_metrics(all_preds, all_labels, num_classes, raw_sim)

    total_params = sum(p.numel() for p in cnn.parameters())
    metrics["method"] = METHOD_NAMES[method]
    metrics["dataset"] = dataset_name
    metrics["class_names"] = class_names
    metrics["seed"] = SEED
    metrics["best_epoch"] = best_epoch
    metrics["total_params"] = total_params
    metrics["trainable_params"] = total_params
    metrics["pipeline"] = "CNN-Stacking (CNN + Boruta + sklearn Stacking)"

    print(f"\n  Results for {METHOD_NAMES[method]} on {dataset_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Macro-F1:  {metrics['macro_f1']:.2f}%")
    print(f"  Severity:  {metrics['severity_mean']:.4f} +/- {metrics['severity_std']:.4f}")
    print(f"  Errors:    {metrics['num_errors']}/{metrics['num_total']}")

    total_time = time.time() - t_start
    print(f"  Total time: {total_time/60:.1f} min")

    del cnn, classifier_head, best_cnn_state
    clear_gpu()
    return metrics


def build_cnn_stacking(num_classes):
    """P5: CNN-Stacking (Khan et al. 2024).
    NOTE: This returns only the CNN feature extractor. The full pipeline
    (CNN + Boruta + Stacking) is handled by train_cnn_stacking()."""
    return CNNFeatureExtractor()


# ###########################################################################
#  Model builder dispatcher
# ###########################################################################

MODEL_BUILDERS = {
    "B1": build_resnet50,
    "B2": build_swin_t,
    "B3": build_convnext_b,
    "B4": build_efficientnet_b7,
    "P1": build_dinov2_lora,
    "P2": build_swin_aarnet,
    "P3": build_ffm_mam,
    "P4": build_ct_pestnet,
    "P5": build_cnn_stacking,
}


# ###########################################################################
#  Training loop
# ###########################################################################

def train_baseline(method, dataset_name):
    """Train a single method on a single dataset. Returns metrics dict."""
    print(f"\n{'='*70}")
    group = "A (Architecture Baseline)" if method in GROUP_A else "B (Published Method)"
    print(f"  GROUP {group}")
    print(f"  TRAIN {METHOD_NAMES[method]}  |  {dataset_name}")
    print(f"{'='*70}")

    set_seed(SEED)
    batch_size = BATCH_SIZES[method]

    # Build transforms and loaders
    if method == "P1":
        train_tf, val_tf = get_dinov2_transforms(224)
    elif method == "P5":
        train_tf, val_tf = get_cnn_stacking_transforms(256)
    else:
        train_tf, val_tf = get_imagenet_transforms(224)

    train_loader, val_loader, class_names, num_classes = build_baseline_loaders(
        dataset_name, train_tf, val_tf, batch_size=batch_size)

    # Build model
    builder = MODEL_BUILDERS[method]
    model = builder(num_classes)
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,}")

    # Optimizer setup
    if method == "P1":
        # DINOv2+LoRA: separate LR for LoRA params and classifier
        lora_params = []
        head_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if "lora" in name.lower():
                    lora_params.append(p)
                else:
                    head_params.append(p)
        param_groups = [
            {"params": lora_params, "lr": LR_FINETUNE},
            {"params": head_params, "lr": LR_LINEAR},
        ]
        print(f"  LoRA params: {sum(p.numel() for p in lora_params):,}")
        print(f"  Head params: {sum(p.numel() for p in head_params):,}")
    else:
        # Full fine-tune: backbone at lower LR, head at higher LR
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("fc", "head", "classifier",
                                        "fusion_proj", "out_proj")):
                head_params.append(p)
            else:
                backbone_params.append(p)
        if not head_params:
            # Fallback: everything at fine-tune LR
            backbone_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": LR_FINETUNE},
        ]
        if head_params:
            param_groups.append({"params": head_params, "lr": LR_LINEAR})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)
    scaler = GradScaler()

    best_val = 0.0
    best_epoch = 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]
                 if p.requires_grad],
                GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    logits = model(images)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()})

        marker = "  *" if val_acc >= best_val else ""
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"[{elapsed:.0f}s]{marker}")

    # Save checkpoint
    ckpt_path = os.path.join(
        CHECKPOINT_DIR, f"{method}_{dataset_name}_seed{SEED}_best.pt")
    if best_state:
        torch.save(best_state, ckpt_path)
        print(f"  OK Checkpoint: {ckpt_path}")
        print(f"  OK Best: {best_val:.2f}% (epoch {best_epoch})")

    # Evaluate with full metrics
    print(f"\n  Computing metrics...")
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with autocast():
                logits = model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    raw_sim = compute_raw_similarity_matrix(
        dataset_name, class_names=class_names, num_classes=num_classes)
    metrics = compute_metrics(all_preds, all_labels, num_classes, raw_sim)

    metrics["method"] = METHOD_NAMES[method]
    metrics["dataset"] = dataset_name
    metrics["class_names"] = class_names
    metrics["seed"] = SEED
    metrics["best_epoch"] = best_epoch
    metrics["total_params"] = total_params
    metrics["trainable_params"] = trainable

    print(f"\n  Results for {METHOD_NAMES[method]} on {dataset_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Macro-F1:  {metrics['macro_f1']:.2f}%")
    print(f"  Severity:  {metrics['severity_mean']:.4f} +/- {metrics['severity_std']:.4f}")
    print(f"  Errors:    {metrics['num_errors']}/{metrics['num_total']}")

    total_time = time.time() - t_start
    print(f"  Total time: {total_time/60:.1f} min")

    del model, best_state
    clear_gpu()
    return metrics


# ###########################################################################
#  Results I/O
# ###########################################################################

BASELINE_SUMMARY_PATH = os.path.join(METRICS_DIR, "baseline_summary.json")
BASELINE_ANALYSIS_PATH = os.path.join(METRICS_DIR, "baseline_macro_f1_analysis.json")
BASELINE_CM_PATH = os.path.join(METRICS_DIR, "baseline_confusion_matrices.json")


def load_existing_results():
    """Load previously saved baseline results (for incremental runs)."""
    summary = []
    analysis = {}
    cm_data = {}
    if os.path.exists(BASELINE_SUMMARY_PATH):
        with open(BASELINE_SUMMARY_PATH) as f:
            summary = json.load(f)
    if os.path.exists(BASELINE_ANALYSIS_PATH):
        with open(BASELINE_ANALYSIS_PATH) as f:
            analysis = json.load(f)
    if os.path.exists(BASELINE_CM_PATH):
        with open(BASELINE_CM_PATH) as f:
            cm_data = json.load(f)
    return summary, analysis, cm_data


def save_results(all_metrics):
    """Save results in same format as evaluate_metrics.py."""
    existing_summary, existing_analysis, existing_cm = load_existing_results()
    existing_keys = {f"{e['method']}_{e['dataset']}": i
                     for i, e in enumerate(existing_summary)}
    summary = list(existing_summary)
    analysis = dict(existing_analysis)
    cm_data = dict(existing_cm)

    for m in all_metrics:
        key = f"{m['method']}_{m['dataset']}"
        entry = {k: v for k, v in m.items()
                 if k not in ("confusion_matrix", "class_names",
                              "per_class_f1", "per_class_precision",
                              "per_class_recall")}
        if key in existing_keys:
            summary[existing_keys[key]] = entry
        else:
            summary.append(entry)
        analysis[key] = {k: v for k, v in m.items()
                        if k != "confusion_matrix"}
        cm_data[key] = m.get("confusion_matrix", [])

    with open(BASELINE_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    with open(BASELINE_ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
    with open(BASELINE_CM_PATH, "w") as f:
        json.dump(cm_data, f, indent=2)

    print(f"\n  OK Results saved:")
    print(f"    - {BASELINE_SUMMARY_PATH}")
    print(f"    - {BASELINE_ANALYSIS_PATH}")
    print(f"    - {BASELINE_CM_PATH}")


# ###########################################################################
#  Summary printer
# ###########################################################################

def print_summary(all_metrics):
    print(f"\n{'='*80}")
    print(f"  BASELINE + PUBLISHED METHOD RESULTS")
    print(f"{'='*80}")

    # Group A
    group_a = [m for m in all_metrics if m["method"].startswith("B")]
    group_b = [m for m in all_metrics if m["method"].startswith("P")]

    if group_a:
        print(f"\n  --- Group A: Architecture Baselines ---")
        print(f"  {'Method':<25} {'Dataset':<20} {'Acc%':>7} {'F1%':>7} "
              f"{'Sev':>7} {'Errors':>8}")
        print(f"  {'-'*25} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
        for m in group_a:
            print(f"  {m['method']:<25} {m['dataset']:<20} "
                  f"{m['accuracy']:>7.2f} {m['macro_f1']:>7.2f} "
                  f"{m['severity_mean']:>7.4f} "
                  f"{m['num_errors']:>5}/{m['num_total']}")

    if group_b:
        print(f"\n  --- Group B: Published Method Comparisons ---")
        print(f"  {'Method':<25} {'Dataset':<20} {'Acc%':>7} {'F1%':>7} "
              f"{'Sev':>7} {'Errors':>8}")
        print(f"  {'-'*25} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
        for m in group_b:
            print(f"  {m['method']:<25} {m['dataset']:<20} "
                  f"{m['accuracy']:>7.2f} {m['macro_f1']:>7.2f} "
                  f"{m['severity_mean']:>7.4f} "
                  f"{m['num_errors']:>5}/{m['num_total']}")
    print()


# ###########################################################################
#  Main
# ###########################################################################

def main():
    parser = argparse.ArgumentParser(
        description="AgroOnto-VLM: Architecture Baselines + Published Methods")
    parser.add_argument("--method", type=str, default=None,
                        choices=ALL_METHODS,
                        help="Run one method (B1-B4 or P1-P5)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run on one dataset (PVT/DW/IP or full name)")
    parser.add_argument("--all", action="store_true",
                        help="Run all applicable method+dataset combos")
    parser.add_argument("--group-a", action="store_true",
                        help="Run Group A only (B1-B4 on all datasets)")
    parser.add_argument("--group-b", action="store_true",
                        help="Run Group B only (P1-P5 on target datasets)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos that already have results")
    args = parser.parse_args()

    if not args.method and not args.all and not args.group_a and not args.group_b:
        parser.print_help()
        print("\n  Examples:")
        print("    python run_baselines.py --method B1 --dataset PVT")
        print("    python run_baselines.py --method P2 --dataset IP")
        print("    python run_baselines.py --group-a           # B1-B4 on all datasets")
        print("    python run_baselines.py --group-b           # P1-P4 on target datasets")
        print("    python run_baselines.py --all               # everything")
        print("    python run_baselines.py --all --resume      # skip done")
        print()
        print("  Group A (Architecture Baselines) — all 3 datasets:")
        for m in GROUP_A:
            print(f"    {m}: {METHOD_NAMES[m]}")
        print()
        print("  Group B (Published Methods) — target dataset only:")
        for m in GROUP_B:
            targets = ", ".join(PUBLISHED_TARGET[m])
            print(f"    {m}: {METHOD_NAMES[m]} -> {targets}")
        return

    # Build run list: (method, dataset) pairs
    run_list = []

    if args.method:
        # Single method specified
        method = args.method
        if args.dataset:
            ds_name = DS_MAP.get(args.dataset, args.dataset)
            run_list.append((method, ds_name))
        elif method in GROUP_A:
            # Architecture baseline: run on all datasets
            for ds in ALL_DATASETS:
                run_list.append((method, ds))
        elif method in GROUP_B:
            # Published method: run on target dataset(s) only
            for ds in PUBLISHED_TARGET[method]:
                run_list.append((method, ds))
    else:
        # --all, --group-a, or --group-b
        if args.all or args.group_a:
            for m in GROUP_A:
                if args.dataset:
                    run_list.append((m, DS_MAP.get(args.dataset, args.dataset)))
                else:
                    for ds in ALL_DATASETS:
                        run_list.append((m, ds))
        if args.all or args.group_b:
            for m in GROUP_B:
                if args.dataset:
                    ds_name = DS_MAP.get(args.dataset, args.dataset)
                    if ds_name in PUBLISHED_TARGET[m]:
                        run_list.append((m, ds_name))
                    else:
                        print(f"  SKIP {METHOD_NAMES[m]}: not targeted at {ds_name}")
                else:
                    for ds in PUBLISHED_TARGET[m]:
                        run_list.append((m, ds))

    if not run_list:
        print("  No runs to execute.")
        return

    # Check for resume
    existing_keys = set()
    if args.resume:
        existing_summary, _, _ = load_existing_results()
        existing_keys = {f"{e['method']}_{e['dataset']}"
                         for e in existing_summary}
        if existing_keys:
            print(f"\n  Resume mode: {len(existing_keys)} existing results found")

    # Print plan
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU")
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0)

    print(f"\n  AgroOnto-VLM -- Baseline + Published Method Training")
    print(f"  GPU: {gpu_name}  |  VRAM: {vram:.1f} GB")
    print(f"  Total runs planned: {len(run_list)}")
    print()
    for i, (m, ds) in enumerate(run_list, 1):
        grp = "A" if m in GROUP_A else "B"
        print(f"    {i}. [{grp}] {METHOD_NAMES[m]:25s} | {ds}")

    # Estimate time
    est_per_dataset = {
        "PlantVillageTomato": 15,
        "DeepWeeds": 30,
        "IP102": 50,
    }
    total_est = sum(est_per_dataset.get(ds, 30) for _, ds in run_list)
    print(f"\n  Estimated total: ~{total_est/60:.1f} hours")
    print(f"{'='*70}")

    all_metrics = []
    total_runs = len(run_list)

    for idx, (method, ds) in enumerate(run_list, 1):
        key = f"{METHOD_NAMES[method]}_{ds}"
        if args.resume and key in existing_keys:
            print(f"\n  SKIP {METHOD_NAMES[method]} | {ds} (already done)")
            continue

        print(f"\n\n{'#'*70}")
        print(f"  RUN {idx}/{total_runs}: {METHOD_NAMES[method]} on {ds}")
        print(f"{'#'*70}")

        try:
            if method == "P5":
                metrics = train_cnn_stacking(method, ds)
            else:
                metrics = train_baseline(method, ds)
            if metrics:
                all_metrics.append(metrics)
                save_results([metrics])
        except Exception as e:
            print(f"\n  ERROR: {METHOD_NAMES[method]} on {ds} failed!")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

        clear_gpu()

    if all_metrics:
        print_summary(all_metrics)

    # Final combined summary
    if os.path.exists(BASELINE_SUMMARY_PATH):
        with open(BASELINE_SUMMARY_PATH) as f:
            final = json.load(f)
        print(f"\n{'='*70}")
        print(f"  ALL SAVED RESULTS ({len(final)} entries)")
        print(f"{'='*70}")
        print(f"\n  {'Method':<25} {'Dataset':<20} {'Acc%':>7} {'F1%':>7} "
              f"{'Sev':>7}")
        print(f"  {'-'*25} {'-'*20} {'-'*7} {'-'*7} {'-'*7}")
        for e in final:
            print(f"  {e['method']:<25} {e['dataset']:<20} "
                  f"{e['accuracy']:>7.2f} {e['macro_f1']:>7.2f} "
                  f"{e['severity_mean']:>7.4f}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
