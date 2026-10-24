"""
run_extra_experiments.py — Low-data experiments (10% and 25% of TRAIN)
=======================================================================

Methods : M1 (AgroOnto-VLM) and M2 (CLIP-LoRA Baseline)
Datasets: PlantVillageTomato, DeepWeeds, IP102
Fractions: 10%, 25% of TRAIN
Seeds   : 42, 123, 456

Protocol (matches run_final_protocol.py):
  - split_seed = 2024
  - TEST = original val split (full, unchanged)
  - VAL  = 15% stratified holdout from original train (split_seed=2024)
  - TRAIN_FULL = 85% of original train
  - TRAIN_SUBSET = stratified subsample of TRAIN_FULL to frac%
                   (subsample_seed=2024, min 1 per class)
  - Same TRAIN_SUBSET for M1, M2, and every seed
  - 15 epochs, 3-epoch linear warmup → cosine, best epoch on VAL
  - Report accuracy + macro-F1 on TEST at best-val epoch

Lambda_onto: PV=0.05, DW=0.10, IP102=0.125
Output: results/extra_experiments/{M}_{dataset}_frac{10|25}_seed{seed}.json

Usage:
  python run_extra_experiments.py --stage lowdata
  python run_extra_experiments.py --stage lowdata --smoke
"""

import sys as _s
try:
    _s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import copy
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── Project imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.dataset import (DeepWeedsDataset, IP102Dataset,
                               PlantVillageDataset, get_clip_preprocess)
from agroonto.model import AgroOntoVLM
from agroonto.ontology import compute_similarity_matrix
from agroonto.train import train_one_epoch

# ── Constants ──────────────────────────────────────────────────────────
EPOCHS        = 15
WARMUP_EPOCHS = 3
SPLIT_SEED    = 2024       # seed for 85/15 train/val split
SUBSAMP_SEED  = 2024       # seed for fraction subsampling
VAL_FRAC      = 0.15
SEEDS         = [42, 123, 456]
FRACTIONS     = [0.10, 0.25]
DATASETS      = ["PlantVillageTomato", "DeepWeeds", "IP102"]

BEST_LAMBDA = {
    "PlantVillageTomato": 0.05,
    "DeepWeeds":          0.10,
    "IP102":              0.125,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Utility ────────────────────────────────────────────────────────────

def set_seed(seed):
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
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cos_epoch = epoch - warmup_epochs
        cos_total = total_epochs - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cos_epoch / max(cos_total, 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Dataset helpers ────────────────────────────────────────────────────

class ListDataset(Dataset):
    """Wrap a list of (path, label) with a CLIP preprocess transform."""
    def __init__(self, samples, transform):
        from PIL import Image
        self._Image = Image
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def stratified_split(samples, val_frac, seed):
    """
    Split list of (path, label) into (train_items, val_items).
    Stratified by label; min 1 per class in each split.
    """
    rng = random.Random(seed)
    by_class = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)
    train_items, val_items = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac)))
        n_val = min(n_val, len(items) - 1)
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def subsample_train(train_items, fraction, seed, min_per_class=1):
    """
    Stratified subsample of train_items to `fraction` of total.
    Guarantees at least min_per_class per class.
    Same result for every call with the same (train_items, fraction, seed).
    """
    rng = random.Random(seed)
    by_class = {}
    for item in train_items:
        by_class.setdefault(item[1], []).append(item)
    subset = []
    for cls, items in by_class.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        n_take = max(min_per_class, int(round(len(items_copy) * fraction)))
        n_take = min(n_take, len(items_copy))
        subset.extend(items_copy[:n_take])
    rng.shuffle(subset)
    return subset


def get_data(dataset_name, preprocess, smoke=False):
    """
    Load (train_full_items, val_items, test_items, class_names, num_classes)
    following the final-protocol split:
      TEST = original val split  (unchanged)
      VAL  = 15% holdout from original train (stratified, SPLIT_SEED)
      TRAIN_FULL = remaining 85% of original train
    """
    if dataset_name == "PlantVillageTomato":
        import os as _os
        train_dir = _os.path.join(config.PLANTVILLAGE_DIR, "train")
        all_classes = sorted(_os.listdir(train_dir))
        tomato = [c for c in all_classes if c.startswith("Tomato")]
        exclude = set(all_classes) - set(tomato)
        train_ds = PlantVillageDataset(
            config.PLANTVILLAGE_DIR, "train", preprocess, exclude_classes=exclude)
        test_ds  = PlantVillageDataset(
            config.PLANTVILLAGE_DIR, "val",   preprocess, exclude_classes=exclude)
        class_names = train_ds.classes
        all_train = list(train_ds.samples)
        test_items = list(test_ds.samples)

    elif dataset_name == "DeepWeeds":
        train_csv = os.path.join(config.DEEPWEEDS_LABEL_DIR, "train_subset0.csv")
        val_csv   = os.path.join(config.DEEPWEEDS_LABEL_DIR, "val_subset0.csv")
        train_ds  = DeepWeedsDataset(
            config.DEEPWEEDS_IMG_DIR, train_csv, preprocess)
        test_ds   = DeepWeedsDataset(
            config.DEEPWEEDS_IMG_DIR, val_csv,   preprocess)
        from agroonto.ontology import DEEPWEEDS_TAXONOMY
        class_names = [DEEPWEEDS_TAXONOMY[i]["species"] for i in range(9)]
        all_train  = list(train_ds.samples)
        test_items = list(test_ds.samples)

    elif dataset_name == "IP102":
        train_ds = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR,   "train"),
            os.path.join(config.IP102_LABEL_DIR, "train"),
            preprocess)
        test_ds  = IP102Dataset(
            os.path.join(config.IP102_IMG_DIR,   "val"),
            os.path.join(config.IP102_LABEL_DIR, "val"),
            preprocess)
        class_names = [f"pest_{i}" for i in range(102)]
        all_train  = list(train_ds.samples)
        test_items = list(test_ds.samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = len(class_names)

    # Smoke: cap everything to tiny sizes
    if smoke:
        by_cls = {}
        for item in all_train:
            by_cls.setdefault(item[1], []).append(item)
        all_train = [it for items in by_cls.values() for it in items[:8]]
        test_items = all_train[:max(8, num_classes // 5)]

    # 85/15 split of original train
    train_full, val_items = stratified_split(all_train, VAL_FRAC, SPLIT_SEED)

    print(f"  {dataset_name}: TRAIN_FULL={len(train_full)}  "
          f"VAL={len(val_items)}  TEST={len(test_items)}  classes={num_classes}")
    return train_full, val_items, test_items, class_names, num_classes


def make_loaders(train_items, val_items, test_items, preprocess,
                 batch_size, num_workers):
    train_ds = ListDataset(train_items, preprocess)
    val_ds   = ListDataset(val_items,   preprocess)
    test_ds  = ListDataset(test_items,  preprocess)
    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_acc(model, loader):
    """Return top-1 accuracy (%) on loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                logits, _, _ = model(imgs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


def evaluate_full(model, loader):
    """Return (accuracy%, macro-F1%) on loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                logits, _, _ = model(imgs)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc    = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1     = 100.0 * f1_score(all_labels, all_preds,
                               average="macro", zero_division=0)
    return acc, f1


# ── M1: AgroOnto-VLM ──────────────────────────────────────────────────

def make_derangement(n, rng_seed=999):
    """Fixed random permutation of 0..n-1 with no fixed points (same for all seeds)."""
    rng = np.random.RandomState(rng_seed)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not (perm == np.arange(n)).any():
            return perm


def run_m1(dataset_name, train_items, val_items, test_items,
           class_names, num_classes, preprocess,
           seed, epochs, fraction_label, smoke=False, shuffle=False, loss_mode=None):
    """Train M1 (AgroOnto-VLM) on a subsampled train set.
    loss_mode=None   : normal M1 loss.
    loss_mode='noKL' : M1 WITHOUT the ontology-target KL only (CE/uniform/proto weights unchanged).
    loss_mode='M3'   : label smoothing (0.90 CE + 0.10 uniform KL), same as final-protocol M3.
    shuffle=True: control experiment, ontology similarity matrix randomly permuted
    (rows and columns by the same fixed derangement), everything else identical."""
    set_seed(seed)
    config.SEED = seed

    batch_size  = 4  if smoke else config.BATCH_SIZE
    num_workers = 0  if smoke else config.NUM_WORKERS

    train_loader, val_loader, test_loader = make_loaders(
        train_items, val_items, test_items, preprocess, batch_size, num_workers)

    onto_sim = compute_similarity_matrix(
        dataset_name, class_names=class_names,
        num_classes=num_classes).to(DEVICE)

    perm = None
    if shuffle:
        perm = make_derangement(num_classes, 999)
        pt = torch.as_tensor(perm, device=onto_sim.device, dtype=torch.long)
        onto_sim = onto_sim[pt][:, pt].contiguous()   # diagonal stays 1.0
        print(f"  [SHUFFLED ontology] permutation: {perm.tolist()}")

    model = AgroOntoVLM(
        num_classes=num_classes, class_names=class_names,
        clip_model_name=config.CLIP_MODEL,
        lora_rank=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
        lora_blocks=config.LORA_BLOCKS, device=DEVICE).to(DEVICE)

    lambda_onto    = BEST_LAMBDA[dataset_name]
    lambda_ce      = 1.0 - lambda_onto - config.LAMBDA_UNIFORM
    lambda_uniform = config.LAMBDA_UNIFORM
    lambda_proto   = config.LAMBDA_PROTO
    if loss_mode == "noKL":
        lambda_onto = 0.0          # lambda_ce stays at its ORIGINAL M1 value (not renormalised)
    elif loss_mode == "M3":
        lambda_ce, lambda_uniform, lambda_onto, lambda_proto = (
            1.0 - config.LAMBDA_UNIFORM, config.LAMBDA_UNIFORM, 0.0, 0.0)
    print(f"  [LOSS] mode={loss_mode} ce={lambda_ce:.3f} uniform={lambda_uniform:.3f} "
          f"onto={lambda_onto:.3f} proto={lambda_proto:.3f}")

    print(f"  [M1] {dataset_name} frac={fraction_label} seed={seed} "
          f"λ_onto={lambda_onto}  train={len(train_items)}")

    param_groups = model.get_param_groups(
        config.LR_LORA, config.LR_HEAD, config.LR_DELTA, config.LR_LOGIT_SCALE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler_with_warmup(optimizer, epochs, WARMUP_EPOCHS)
    scaler    = GradScaler()

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch": []}
    best_val, best_epoch, best_state = 0.0, 0, None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, onto_sim, num_classes,
            lambda_ce, lambda_uniform, lambda_onto, lambda_proto, DEVICE)
        val_acc = evaluate_acc(model, val_loader)
        scheduler.step()

        if val_acc > best_val:
            best_val   = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()
                 if v.requires_grad or any(
                     t in k.lower() for t in ("lora", "head", "delta", "logit"))})

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc,  2))
        history["val_acc"].append(round(val_acc,       2))
        marker = "★" if epoch == best_epoch else " "
        print(f"  {marker} Ep {epoch:02d}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%  [{time.time()-t0:.1f}s]")

    # Restore best checkpoint → evaluate on TEST
    if best_state:
        model.load_state_dict(best_state, strict=False)
    test_acc, test_f1 = evaluate_full(model, test_loader)
    wall = round(time.time() - t_start, 1)

    print(f"  ✓ M1 {dataset_name} frac={fraction_label} seed={seed}: "
          f"best-val={best_val:.2f}% (ep {best_epoch}) | "
          f"TEST acc={test_acc:.2f}%  F1={test_f1:.2f}%  [{wall/60:.1f} min]")

    return {
        "method": ("M1shuffled_Ontology-Shuffled" if shuffle else
                   "M1noKL_WithoutOntologyKL" if loss_mode == "noKL" else
                   "M3_LabelSmoothing" if loss_mode == "M3" else "M1_AgroOnto-VLM"),
        "loss_weights": {"ce": lambda_ce, "uniform": lambda_uniform, "onto": lambda_onto, "proto": lambda_proto},
        "permutation": perm.tolist() if perm is not None else None,
        "dataset": dataset_name,
        "fraction": fraction_label,
        "seed": seed,
        "lambda_onto": lambda_onto,
        "train_size": len(train_items),
        "val_size":   len(val_items),
        "test_size":  len(test_items),
        "best_val_acc": round(best_val,   4),
        "best_epoch":   best_epoch,
        "test_acc":     round(test_acc,   4),
        "test_macro_f1": round(test_f1,   4),
        "epochs": epochs,
        "history": history,
        "wall_time_sec": wall,
    }


# ── M2: CLIP-LoRA Baseline ────────────────────────────────────────────

def run_m2(dataset_name, train_items, val_items, test_items,
           class_names, num_classes, preprocess,
           seed, epochs, fraction_label, smoke=False):
    """Train M2 (pure CE, no ontology) on a subsampled train set."""
    set_seed(seed)
    config.SEED = seed

    batch_size  = 4  if smoke else config.BATCH_SIZE
    num_workers = 0  if smoke else config.NUM_WORKERS

    train_loader, val_loader, test_loader = make_loaders(
        train_items, val_items, test_items, preprocess, batch_size, num_workers)

    model = AgroOntoVLM(
        num_classes=num_classes, class_names=class_names,
        clip_model_name=config.CLIP_MODEL,
        lora_rank=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
        lora_blocks=config.LORA_BLOCKS, device=DEVICE).to(DEVICE)

    print(f"  [M2] {dataset_name} frac={fraction_label} seed={seed} "
          f"pure-CE  train={len(train_items)}")

    param_groups = model.get_param_groups(
        config.LR_LORA, config.LR_HEAD, config.LR_DELTA, config.LR_LOGIT_SCALE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler_with_warmup(optimizer, epochs, WARMUP_EPOCHS)
    scaler    = GradScaler()

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch": []}
    best_val, best_epoch, best_state = 0.0, 0, None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits_cls, logits_clip, _ = model(imgs)
                loss = F.cross_entropy(logits_cls, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            preds = logits_cls.argmax(dim=-1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            total_loss  += loss.item() * labels.size(0)
        train_loss = total_loss / total
        train_acc  = 100.0 * correct / total
        val_acc    = evaluate_acc(model, val_loader)
        scheduler.step()

        if val_acc > best_val:
            best_val   = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()
                 if v.requires_grad or any(
                     t in k.lower() for t in ("lora", "head", "delta", "logit"))})

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc,  2))
        history["val_acc"].append(round(val_acc,       2))
        marker = "★" if epoch == best_epoch else " "
        print(f"  {marker} Ep {epoch:02d}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%  [{time.time()-t0:.1f}s]")

    if best_state:
        model.load_state_dict(best_state, strict=False)
    test_acc, test_f1 = evaluate_full(model, test_loader)
    wall = round(time.time() - t_start, 1)

    print(f"  ✓ M2 {dataset_name} frac={fraction_label} seed={seed}: "
          f"best-val={best_val:.2f}% (ep {best_epoch}) | "
          f"TEST acc={test_acc:.2f}%  F1={test_f1:.2f}%  [{wall/60:.1f} min]")

    return {
        "method": "M2_Baseline",
        "dataset": dataset_name,
        "fraction": fraction_label,
        "seed": seed,
        "train_size": len(train_items),
        "val_size":   len(val_items),
        "test_size":  len(test_items),
        "best_val_acc": round(best_val,  4),
        "best_epoch":   best_epoch,
        "test_acc":     round(test_acc,  4),
        "test_macro_f1": round(test_f1,  4),
        "epochs": epochs,
        "history": history,
        "wall_time_sec": wall,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",   type=str, default="lowdata",
                        choices=["lowdata", "shuffled", "m3", "nokl"], help="Experiment stage")
    parser.add_argument("--smoke",   action="store_true",
                        help="Smoke test: 1 epoch, tiny data, separate folder")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset")
    parser.add_argument("--method",  type=str, default=None,
                        choices=["M1", "M2"], help="Run only M1 or M2")
    args = parser.parse_args()

    epochs = 1 if args.smoke else EPOCHS
    out_dir = os.path.join(config.BASE_DIR, "results",
                           "extra_experiments_smoke" if args.smoke
                           else "extra_experiments")
    os.makedirs(out_dir, exist_ok=True)

    datasets_to_run = [args.dataset] if args.dataset else DATASETS
    methods_to_run  = [args.method]  if args.method  else ["M1", "M2"]
    if args.stage == "shuffled":
        methods_to_run = ["M1shuffled"]
    elif args.stage == "m3":
        methods_to_run = ["M3"]
    elif args.stage == "nokl":
        methods_to_run = ["M1noKL"]

    print("=" * 70)
    print("  AgroOnto-VLM — Low-data experiments")
    print(f"  Epochs: {epochs}  |  Device: {DEVICE}  |  Smoke: {args.smoke}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  |  VRAM: {props.total_memory/1024**3:.1f} GB")
    print(f"  Output: {out_dir}")
    print(f"  Methods: {methods_to_run}  |  Datasets: {datasets_to_run}")
    print(f"  Fractions: {FRACTIONS}  |  Seeds: {SEEDS}")
    print("=" * 70)

    preprocess = get_clip_preprocess()
    total_start = time.time()
    total_runs = 0

    for dataset_name in datasets_to_run:
        print(f"\n{'─'*70}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'─'*70}")

        # Load once per dataset
        train_full, val_items, test_items, class_names, num_classes = get_data(
            dataset_name, preprocess, smoke=args.smoke)

        for fraction in FRACTIONS:
            frac_pct   = int(round(fraction * 100))
            frac_label = str(frac_pct)  # "10" or "25"

            # Build subset ONCE (same for M1, M2, all seeds)
            if args.smoke:
                # Smoke: just take first few items per class
                by_cls = {}
                for item in train_full:
                    by_cls.setdefault(item[1], []).append(item)
                train_subset = [items[0] for items in by_cls.values()]
            else:
                train_subset = subsample_train(
                    train_full, fraction, SUBSAMP_SEED)

            print(f"\n  Fraction {frac_pct}%: TRAIN_SUBSET={len(train_subset)} samples")

            for method_id in methods_to_run:
                for seed in SEEDS:
                    fname = (f"{method_id}_{dataset_name}_"
                             f"frac{frac_label}_seed{seed}.json")
                    fpath = os.path.join(out_dir, fname)

                    if os.path.exists(fpath):
                        print(f"  ↷ Skip (exists): {fname}")
                        continue

                    print(f"\n  ── {method_id} | {dataset_name} | "
                          f"frac={frac_pct}% | seed={seed} ──")

                    try:
                        clear_gpu()
                        if method_id in ("M1", "M1shuffled", "M3", "M1noKL"):
                            result = run_m1(
                                dataset_name, train_subset, val_items, test_items,
                                class_names, num_classes, preprocess,
                                seed, epochs, frac_label, smoke=args.smoke,
                                shuffle=(method_id == "M1shuffled"),
                                loss_mode=("noKL" if method_id == "M1noKL" else
                                           "M3" if method_id == "M3" else None))
                        else:
                            result = run_m2(
                                dataset_name, train_subset, val_items, test_items,
                                class_names, num_classes, preprocess,
                                seed, epochs, frac_label, smoke=args.smoke)

                        with open(fpath, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"  ✓ Saved: {fpath}")
                        total_runs += 1

                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")
                        import traceback; traceback.print_exc()

                    clear_gpu()

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Done. {total_runs} runs completed in "
          f"{total_time/3600:.1f} h ({total_time/60:.0f} min)")
    print(f"  Results: {out_dir}")
    print(f"{'='*70}")

    print("\nTo aggregate results, run:")
    print(f"  python aggregate_extra.py --results_dir {out_dir}")


if __name__ == "__main__":
    main()
