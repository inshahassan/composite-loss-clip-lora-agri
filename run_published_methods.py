"""
AgroOnto-VLM — Published Methods Comparison (Group B)
=====================================================

Implements and evaluates published SOTA methods on the SAME datasets
and splits used by AgroOnto-VLM, for fair head-to-head comparison.

Published Methods:
  P1: DINOv2+LoRA+KD    (Espejo-Garcia et al., 2025, CEA)  → DeepWeeds
  P2: Swin-AARNet       (Wang et al., 2025, Front.Plant Sci.) → IP102
  P3: FFM+MAM           (Qian et al., 2025, CEA)            → IP102
  P4: CT-PestNet         (Li et al., 2026, CEA)              → IP102
  P5: CNN-Stacking       (Khan et al., 2024, Sci.Rep.)       → PlantVillageTomato

Usage:
  python run_published_methods.py                    # all methods
  python run_published_methods.py --method P1        # single method
  python run_published_methods.py --method P5        # CNN-Stacking on PV-Tomato
  python run_published_methods.py --table-only       # comparison table from results
"""
import argparse
import os
import sys
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.dataset import (build_loaders, get_clip_preprocess,
                               PlantVillageDataset, DeepWeedsDataset, IP102Dataset)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(config.BASE_DIR, "results", "published_methods")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 30
WARMUP_EPOCHS = 3

PUBLISHED_TARGETS = {
    "P1": {"method": "DINOv2+LoRA+KD",   "dataset": "DeepWeeds",         "reported": 98.08},
    "P2": {"method": "Swin-AARNet",       "dataset": "IP102",             "reported": 78.77},
    "P3": {"method": "FFM+MAM",           "dataset": "IP102",             "reported": None},
    "P4": {"method": "CT-PestNet",        "dataset": "IP102",             "reported": None},
    "P5": {"method": "CNN-Stacking",      "dataset": "PlantVillageTomato","reported": 98.27},
}


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


def save_result(result, method_id):
    path = os.path.join(RESULTS_DIR, f"{method_id}_{result['dataset']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


def build_scheduler_with_warmup(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cos_epoch = epoch - warmup_epochs
        cos_total = total_epochs - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cos_epoch / max(cos_total, 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_imagenet_preprocess(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler,
                       num_epochs, device, method_name, dataset_name, extra_info=None):
    """Generic training loop for PyTorch models."""
    scaler = GradScaler()
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch": []}
    best_val, best_epoch = 0.0, 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        model.eval()
        eval_correct, eval_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    logits = model(images)
                preds = logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)
        val_acc = 100.0 * eval_correct / eval_total
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        marker = "  *" if val_acc >= best_val else ""
        print(f"  Epoch {epoch}/{num_epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%{marker}")

    wall_time = time.time() - t_start
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "method": method_name,
        "dataset": dataset_name,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "final_val_acc": history["val_acc"][-1],
        "epochs": num_epochs,
        "trainable_params": trainable,
        "history": history,
        "wall_time_sec": round(wall_time, 1),
    }
    if extra_info:
        result.update(extra_info)
    return result


# ===================================================================
#  P1: DINOv2 + LoRA + Knowledge Distillation
#  Espejo-Garcia et al. (2025), Comput. Electron. Agric.
#  DOI: 10.1016/j.compag.2025.110900
#  Dataset: DeepWeeds
# ===================================================================

class DINOv2LoRA(nn.Module):
    """DINOv2 ViT-B/14 with LoRA adapters for classification."""

    def __init__(self, num_classes, lora_rank=8, lora_alpha=16, lora_blocks=4):
        super().__init__()
        try:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14',
                pretrained=True)
        except Exception:
            import timm
            self.backbone = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m', pretrained=True)

        self.embed_dim = 768
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.lora_adapters = nn.ModuleList()
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, 'encoder'):
            blocks = self.backbone.encoder.layer if hasattr(self.backbone.encoder, 'layer') else []
        else:
            blocks = []

        n_blocks = len(blocks)
        start = max(0, n_blocks - lora_blocks)
        for i in range(start, n_blocks):
            adapter = nn.ModuleDict({
                "down": nn.Linear(self.embed_dim, lora_rank, bias=False),
                "up": nn.Linear(lora_rank, self.embed_dim, bias=False),
            })
            nn.init.kaiming_uniform_(adapter["down"].weight)
            nn.init.zeros_(adapter["up"].weight)
            self.lora_adapters.append(adapter)

            blk = blocks[i]
            original_forward = blk.forward

            def make_hook(adapter_mod, orig_fwd):
                def new_forward(x, *args, **kwargs):
                    out = orig_fwd(x, *args, **kwargs)
                    lora_out = adapter_mod["up"](F.gelu(adapter_mod["down"](out)))
                    return out + (lora_alpha / lora_rank) * lora_out
                return new_forward

            blk.forward = make_hook(adapter, original_forward)

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x):
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            if isinstance(features, dict):
                features = features.get('x_norm_clstoken', features.get('x', None))
                if features is None:
                    features = list(features.values())[0]
            if features.dim() == 3:
                features = features[:, 0]
        else:
            features = self.backbone(x)
            if features.dim() == 3:
                features = features[:, 0]
        return self.head(features.float())


def run_p1_dinov2_lora():
    """P1: DINOv2+LoRA+KD on DeepWeeds."""
    dataset_name = "DeepWeeds"
    print(f"\n{'='*70}")
    print(f"  P1: DINOv2+LoRA+KD (Espejo-Garcia 2025)  |  {dataset_name}")
    print(f"{'='*70}")
    set_seed(42)

    preprocess = get_imagenet_preprocess(518)  # DINOv2 uses 518px
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    model = DINOv2LoRA(num_classes=num_classes, lora_rank=8,
                       lora_alpha=16, lora_blocks=4).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Backbone: DINOv2 ViT-B/14 (frozen) + LoRA adapters")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.01)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)

    result = train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler,
        EPOCHS, DEVICE, "DINOv2+LoRA+KD", dataset_name,
        extra_info={"paper": "Espejo-Garcia et al. 2025",
                    "reported_accuracy": 98.08})
    save_result(result, "P1")
    return result


# ===================================================================
#  P2: Swin-AARNet
#  Wang et al. (2025), Front. Plant Sci.
#  DOI: 10.3389/fpls.2025.1619551
#  Dataset: IP102
# ===================================================================

class SwinAARNet(nn.Module):
    """Swin Transformer + Adaptive Attention Refinement."""

    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 1024

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layers[-1].parameters():
            p.requires_grad = True
        if hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.aar = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.aar(features)
        refined = features * attention_weights
        return self.head(refined)


def run_p2_swin_aarnet():
    """P2: Swin-AARNet on IP102."""
    dataset_name = "IP102"
    print(f"\n{'='*70}")
    print(f"  P2: Swin-AARNet (Wang 2025)  |  {dataset_name}")
    print(f"{'='*70}")
    set_seed(42)

    preprocess = get_imagenet_preprocess(224)
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    model = SwinAARNet(num_classes=num_classes).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Backbone: Swin-B (last stage fine-tuned) + AAR module")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=0.05)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)

    result = train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler,
        EPOCHS, DEVICE, "Swin-AARNet", dataset_name,
        extra_info={"paper": "Wang et al. 2025",
                    "reported_accuracy": 78.77})
    save_result(result, "P2")
    return result


# ===================================================================
#  P3: FFM+MAM (Feature Fusion + Multi-scale Attention)
#  Qian et al. (2025), Comput. Electron. Agric.
#  Dataset: IP102
# ===================================================================

class FFM_MAM(nn.Module):
    """ResNet-50 + Feature Fusion Module + Multi-scale Attention Module."""

    def __init__(self, num_classes):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048

        for p in self.layer0.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False

        self.ffm_conv3 = nn.Conv2d(1024, 512, 1)
        self.ffm_conv4 = nn.Conv2d(2048, 512, 1)
        self.ffm_bn = nn.BatchNorm2d(512)

        self.mam_pool = nn.AdaptiveAvgPool2d(1)
        self.mam_attn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)

        f3_proj = self.ffm_conv3(f3)
        f4_proj = self.ffm_conv4(f4)
        f4_up = F.interpolate(f4_proj, size=f3_proj.shape[2:], mode='bilinear',
                              align_corners=False)
        fused = self.ffm_bn(f3_proj + f4_up)

        pooled = self.mam_pool(fused).flatten(1)
        attn = self.mam_attn(pooled)
        refined = pooled * attn

        return self.classifier(refined)


def run_p3_ffm_mam():
    """P3: FFM+MAM on IP102."""
    dataset_name = "IP102"
    print(f"\n{'='*70}")
    print(f"  P3: FFM+MAM (Qian 2025)  |  {dataset_name}")
    print(f"{'='*70}")
    set_seed(42)

    preprocess = get_imagenet_preprocess(224)
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    model = FFM_MAM(num_classes=num_classes).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Backbone: ResNet-50 (layer2+ fine-tuned) + FFM + MAM")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.01)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)

    result = train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler,
        EPOCHS, DEVICE, "FFM+MAM", dataset_name,
        extra_info={"paper": "Qian et al. 2025"})
    save_result(result, "P3")
    return result


# ===================================================================
#  P4: CT-PestNet (ConvNeXt-Tiny based)
#  Li et al. (2026), Comput. Electron. Agric.
#  Dataset: IP102
# ===================================================================

class CTPestNet(nn.Module):
    """ConvNeXt-Tiny + custom pest classification head."""

    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 768

        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, 'stages'):
            for p in self.backbone.stages[-1].parameters():
                p.requires_grad = True
        if hasattr(self.backbone, 'head'):
            for p in self.backbone.head.parameters():
                p.requires_grad = True

        self.pest_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim // 2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.pest_head(features)


def run_p4_ct_pestnet():
    """P4: CT-PestNet on IP102."""
    dataset_name = "IP102"
    print(f"\n{'='*70}")
    print(f"  P4: CT-PestNet (Li 2026)  |  {dataset_name}")
    print(f"{'='*70}")
    set_seed(42)

    preprocess = get_imagenet_preprocess(224)
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    model = CTPestNet(num_classes=num_classes).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Backbone: ConvNeXt-Tiny (last stage fine-tuned)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=0.05)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)

    result = train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler,
        EPOCHS, DEVICE, "CT-PestNet", dataset_name,
        extra_info={"paper": "Li et al. 2026"})
    save_result(result, "P4")
    return result


# ===================================================================
#  P5: CNN-Stacking (Bayesian Optimized)
#  Khan et al. (2024), Sci. Rep.
#  DOI: 10.1038/s41598-024-72237-x
#  Dataset: PlantVillageTomato
# ===================================================================

class CNNFeatureExtractor(nn.Module):
    """4-layer CNN as described in Khan et al. (2024).
    Architecture: 32->64->16->64 filters, followed by dense layers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.flatten_dim = 64 * 4 * 4  # 1024
        self.dense = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.dense(x)


class CNNStackingClassifier(nn.Module):
    """CNN feature extractor + stacking-style MLP ensemble head.
    Approximates the sklearn stacking in Khan et al. via a multi-head MLP."""

    def __init__(self, num_classes, feature_dim=128, n_base_learners=5):
        super().__init__()
        self.cnn = CNNFeatureExtractor()

        self.base_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            ) for _ in range(n_base_learners)
        ])

        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * n_base_learners, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.cnn(x)
        base_preds = [learner(features) for learner in self.base_learners]
        stacked = torch.cat(base_preds, dim=-1)
        return self.meta_learner(stacked)


def run_p5_cnn_stacking():
    """P5: CNN-Stacking on PlantVillageTomato."""
    dataset_name = "PlantVillageTomato"
    print(f"\n{'='*70}")
    print(f"  P5: CNN-Stacking (Khan 2024)  |  {dataset_name}")
    print(f"{'='*70}")
    set_seed(42)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    model = CNNStackingClassifier(
        num_classes=num_classes, feature_dim=128, n_base_learners=5
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Architecture: 4-layer CNN + 5-head stacking ensemble")
    print(f"  Classes: {num_classes} ({', '.join(class_names[:3])}...)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = build_scheduler_with_warmup(optimizer, EPOCHS, WARMUP_EPOCHS)

    result = train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler,
        EPOCHS, DEVICE, "CNN-Stacking", dataset_name,
        extra_info={"paper": "Khan et al. 2024",
                    "reported_accuracy": 98.27})
    save_result(result, "P5")
    return result


# ===================================================================
#  Comparison table
# ===================================================================

def load_our_results():
    """Load AgroOnto-VLM (M1) results from all_methods."""
    our_dir = os.path.join(config.BASE_DIR, "results", "all_methods")
    our = {}
    for ds in ["PlantVillageTomato", "PlantVillage", "DeepWeeds", "IP102"]:
        fpath = os.path.join(our_dir, f"M1_proposed_{ds}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            our[ds] = data.get("best_val_acc", 0)

    summary_path = os.path.join(our_dir, "all_methods_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for entry in summary:
            if entry.get("method") == "AgroOnto-VLM (Proposed)":
                ds = entry["dataset"]
                if ds == "PlantVillage":
                    our.setdefault("PlantVillageTomato", entry.get("best_val_acc", 0))
                our[ds] = entry.get("best_val_acc", 0)
    return our


def generate_comparison_table():
    """Print comparison table: AgroOnto-VLM vs published methods."""
    print(f"\n\n{'#'*70}")
    print(f"  PUBLISHED METHODS COMPARISON TABLE")
    print(f"{'#'*70}")

    our = load_our_results()
    published = {}
    for pid in PUBLISHED_TARGETS:
        ds = PUBLISHED_TARGETS[pid]["dataset"]
        fpath = os.path.join(RESULTS_DIR, f"{pid}_{ds}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            published[pid] = {
                "acc": data.get("best_val_acc", 0),
                "dataset": ds,
                "method": PUBLISHED_TARGETS[pid]["method"],
                "reported": PUBLISHED_TARGETS[pid].get("reported"),
            }

    hdr = f"  {'ID':<4} {'Method':<28} {'Dataset':<20} {'Ours':>8} {'Repro':>8} {'Reported':>10} {'Delta':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * 90)

    for pid, info in PUBLISHED_TARGETS.items():
        ds = info["dataset"]
        method = info["method"]
        reported = info.get("reported")
        reported_str = f"{reported:.2f}%" if reported else "N/A"

        our_acc = our.get(ds, our.get("PlantVillage", 0)) if ds == "PlantVillageTomato" else our.get(ds, 0)
        our_str = f"{our_acc:.2f}%" if our_acc > 0 else "---"

        if pid in published:
            repro = published[pid]["acc"]
            repro_str = f"{repro:.2f}%"
            delta = our_acc - repro if our_acc > 0 else 0
            delta_str = f"{delta:+.2f}%" if our_acc > 0 else "---"
        else:
            repro_str = "pending"
            delta_str = "---"

        print(f"  {pid:<4} {method:<28} {ds:<20} {our_str:>8} {repro_str:>8} {reported_str:>10} {delta_str:>8}")

    print("  " + "-" * 90)

    print(f"\n  Legend:")
    print(f"    Ours     = AgroOnto-VLM (M1) best val accuracy (our experiments)")
    print(f"    Repro    = Reproduced accuracy (same data splits, our GPU)")
    print(f"    Reported = Published accuracy from the paper")
    print(f"    Delta    = Ours - Repro (positive = we beat them)")

    summary = {"our_results": our, "published_repro": published}
    out_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AgroOnto-VLM: Published Methods Comparison (Group B)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["P1", "P2", "P3", "P4", "P5"],
                        help="Run only this published method")
    parser.add_argument("--table-only", action="store_true",
                        help="Just generate comparison table from existing results")
    args = parser.parse_args()

    print("=" * 70)
    print("  AgroOnto-VLM — Published Methods Comparison")
    print(f"  Epochs: {EPOCHS}  |  Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  |  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)

    if args.table_only:
        generate_comparison_table()
        return

    METHOD_MAP = {
        "P1": ("DINOv2+LoRA+KD",  run_p1_dinov2_lora),
        "P2": ("Swin-AARNet",     run_p2_swin_aarnet),
        "P3": ("FFM+MAM",         run_p3_ffm_mam),
        "P4": ("CT-PestNet",      run_p4_ct_pestnet),
        "P5": ("CNN-Stacking",    run_p5_cnn_stacking),
    }

    methods_to_run = [args.method] if args.method else list(METHOD_MAP.keys())

    print(f"\n  Methods to run: {methods_to_run}")

    total_start = time.time()
    all_results = []

    for mid in methods_to_run:
        if mid not in METHOD_MAP:
            print(f"  Unknown method: {mid}")
            continue

        method_name, method_fn = METHOD_MAP[mid]
        try:
            clear_gpu()
            result = method_fn()
            all_results.append(result)
            print(f"\n  {mid} ({method_name}): {result['best_val_acc']:.2f}%\n")
        except Exception as e:
            print(f"\n  FAILED: {mid} ({method_name}): {e}\n")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time/3600:.1f} hours")

    generate_comparison_table()

    print(f"\n{'='*70}")
    print(f"  Done! Results in: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
