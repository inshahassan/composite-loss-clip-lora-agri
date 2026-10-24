"""
Dataset loaders for PlantVillage, DeepWeeds, and IP102.
Each returns (train_loader, val_loader, class_names, num_classes).
"""
import os
import csv
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import clip

from . import config


def get_clip_preprocess():
    """Return the CLIP preprocessing transform."""
    _, preprocess = clip.load(config.CLIP_MODEL, device="cpu")
    return preprocess


# ═══════════════════════════════════════════════════════════════════════
#  PlantVillage  (ImageFolder with train/val split)
# ═══════════════════════════════════════════════════════════════════════
class PlantVillageDataset(Dataset):
    def __init__(self, root, split="train", transform=None, exclude_classes=None):
        self.transform = transform
        split_dir = os.path.join(root, split)
        all_classes = sorted(os.listdir(split_dir))
        if exclude_classes:
            all_classes = [c for c in all_classes if c not in exclude_classes]
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.classes = all_classes
        self.samples = []
        for cls_name in all_classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname),
                                         self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════════════
#  DeepWeeds  (CSV labels + flat image folder, 5-fold splits)
# ═══════════════════════════════════════════════════════════════════════
class DeepWeedsDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, exclude_classes=None,
                 class_remap=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["Filename"]
                label = int(row["Label"])
                if exclude_classes and label in exclude_classes:
                    continue
                if class_remap is not None:
                    label = class_remap.get(label, label)
                path = os.path.join(img_dir, fname)
                if os.path.exists(path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════════════
#  IP102  (YOLO labels → classification)
# ═══════════════════════════════════════════════════════════════════════
class IP102Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None,
                 exclude_classes=None, class_remap=None):
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lbl_file = os.path.join(label_dir,
                                    os.path.splitext(fname)[0] + ".txt")
            if not os.path.exists(lbl_file):
                continue
            with open(lbl_file) as f:
                line = f.readline().strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if exclude_classes and cls_id in exclude_classes:
                continue
            if class_remap is not None:
                cls_id = class_remap.get(cls_id, cls_id)
            self.samples.append((os.path.join(img_dir, fname), cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════════════
#  Unified loader builder
# ═══════════════════════════════════════════════════════════════════════
def build_loaders(dataset_name, preprocess):
    """
    Build train/val DataLoaders for the given dataset.
    Returns (train_loader, val_loader, class_names, num_classes).
    Supported: PlantVillageTomato, PlantVillage, DeepWeeds, IP102.
    """
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    if dataset_name == "PlantVillageTomato":
        # ── Tomato-only subset (no holdout) ──────────────────────────
        train_dir = os.path.join(config.PLANTVILLAGE_DIR, "train")
        all_classes = sorted(os.listdir(train_dir))
        # Keep only classes starting with "Tomato"
        tomato_classes = [c for c in all_classes if c.startswith("Tomato")]
        exclude = set(all_classes) - set(tomato_classes)
        print(f"  PlantVillageTomato: {len(all_classes)} total, "
              f"keeping {len(tomato_classes)} tomato classes (no holdout)")
        train_ds = PlantVillageDataset(config.PLANTVILLAGE_DIR, "train",
                                       preprocess, exclude_classes=exclude)
        val_ds = PlantVillageDataset(config.PLANTVILLAGE_DIR, "val",
                                     preprocess, exclude_classes=exclude)
        class_names = train_ds.classes
        num_classes = len(class_names)

    elif dataset_name == "PlantVillage":
        # ── Full PlantVillage (no holdout) ───────────────────────────
        train_dir = os.path.join(config.PLANTVILLAGE_DIR, "train")
        all_classes = sorted(os.listdir(train_dir))
        print(f"  PlantVillage: {len(all_classes)} total (no holdout)")
        train_ds = PlantVillageDataset(config.PLANTVILLAGE_DIR, "train",
                                       preprocess)
        val_ds = PlantVillageDataset(config.PLANTVILLAGE_DIR, "val",
                                     preprocess)
        class_names = train_ds.classes
        num_classes = len(class_names)

    elif dataset_name == "DeepWeeds":
        # ── Full DeepWeeds — all 9 classes, no holdout ───────────────
        print(f"  DeepWeeds: 9 classes (no holdout)")
        train_csv = os.path.join(config.DEEPWEEDS_LABEL_DIR, "train_subset0.csv")
        val_csv = os.path.join(config.DEEPWEEDS_LABEL_DIR, "val_subset0.csv")
        train_ds = DeepWeedsDataset(config.DEEPWEEDS_IMG_DIR, train_csv,
                                     preprocess)
        val_ds = DeepWeedsDataset(config.DEEPWEEDS_IMG_DIR, val_csv,
                                   preprocess)
        from .ontology import DEEPWEEDS_TAXONOMY
        class_names = [DEEPWEEDS_TAXONOMY[i]["species"] for i in range(9)]
        num_classes = len(class_names)

    elif dataset_name == "IP102":
        # ── Full IP102 — all 102 classes, no holdout ─────────────────
        print(f"  IP102: 102 classes (no holdout)")
        train_ds = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR, "train"),
            os.path.join(config.IP102_LABEL_DIR, "train"),
            preprocess)
        val_ds = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR, "val"),
            os.path.join(config.IP102_LABEL_DIR, "val"),
            preprocess)
        class_names = [f"pest_{i}" for i in range(102)]
        num_classes = len(class_names)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Closed-set classes: {num_classes}, "
          f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)
    return train_loader, val_loader, class_names, num_classes
