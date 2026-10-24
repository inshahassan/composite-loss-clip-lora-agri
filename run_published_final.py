"""
AgroOnto-VLM -- published methods re-run on the FINAL protocol (fair comparison)
================================================================================
Same protocol as run_final_protocol.py / run_extra_experiments.py:
  * TEST  = original val split of each dataset (never used for any selection)
  * VAL   = 15% stratified hold-out of the original train (split seed 2024)
  * TRAIN = remaining 85%
  * epoch chosen on VAL only; result = TEST accuracy / macro-F1 at the best-VAL epoch
  * default 15 epochs, seed 42 (single run; say "single run" in the paper)
Architectures, optimisers, learning rates and input pre-processing are taken UNCHANGED
from run_published_methods.py (controlled re-implementations, not exact reproductions).

Usage (from the AgroOnto-VLM folder):
  python -u run_published_final.py --smoke > smoke_pub_log.txt 2>&1
  python -u run_published_final.py --methods P5 P4 P1 > pub_final_log.txt 2>&1
  python -u run_published_final.py > pub_final_log.txt 2>&1          (all five)
Results: results\\published_final\\{P}_{dataset}_seed{seed}.json (finished runs are skipped)
"""
import sys as _s
try:
    _s.stdout.reconfigure(encoding="utf-8"); _s.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.dataset import build_loaders
from run_final_protocol import stratified_split, metrics, set_seed, SPLIT_SEED, VAL_FRAC, warmup_cosine
from run_published_methods import (DINOv2LoRA, SwinAARNet, FFM_MAM, CTPestNet,
                                   CNNStackingClassifier, get_imagenet_preprocess)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 3


def _p5_pre():
    return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


# method id -> (paper, dataset, model builder, preprocess builder, lr, weight_decay)
SPECS = {
    "P5": ("CNN-Stacking (Khan 2024)", "PlantVillageTomato",
           lambda K: CNNStackingClassifier(num_classes=K, feature_dim=128, n_base_learners=5),
           _p5_pre, 1e-3, 0.01),
    "P4": ("CT-PestNet (Li 2026)", "IP102",
           lambda K: CTPestNet(num_classes=K), lambda: get_imagenet_preprocess(224), 5e-5, 0.05),
    "P2": ("Swin-AARNet (Wang 2025)", "IP102",
           lambda K: SwinAARNet(num_classes=K), lambda: get_imagenet_preprocess(224), 5e-5, 0.05),
    "P3": ("FFM+MAM (Qian 2025)", "IP102",
           lambda K: FFM_MAM(num_classes=K), lambda: get_imagenet_preprocess(224), 1e-4, 0.01),
    "P1": ("DINOv2+LoRA+KD (Espejo-Garcia 2025)", "DeepWeeds",
           lambda K: DINOv2LoRA(num_classes=K, lora_rank=8, lora_alpha=16, lora_blocks=4),
           lambda: get_imagenet_preprocess(518), 1e-4, 0.01),
}
DEFAULT_ORDER = ["P5", "P4", "P2", "P3", "P1"]   # cheapest first


def get_data(ds, preprocess, smoke=False):
    train_loader, test_loader, names, K = build_loaders(ds, preprocess)
    base = train_loader.dataset
    labels = [s[1] for s in base.samples]
    tr, va = stratified_split(labels, VAL_FRAC, SPLIT_SEED)
    test_ds = test_loader.dataset
    te = list(range(len(test_ds)))
    if smoke:
        by = {}
        for i in tr:
            by.setdefault(labels[i], []).append(i)
        tr = [i for v in by.values() for i in v[:6]]
        va = va[:64]
        te = te[:64]
    bs = 4 if smoke else config.BATCH_SIZE
    nw = 0 if smoke else config.NUM_WORKERS
    kw = dict(num_workers=nw, pin_memory=True)
    d = dict(
        names=names, K=K,
        train=DataLoader(Subset(base, tr), batch_size=bs, shuffle=True, drop_last=True, **kw),
        val=DataLoader(Subset(base, va), batch_size=bs, shuffle=False, **kw),
        test=DataLoader(Subset(test_ds, te), batch_size=bs, shuffle=False, **kw),
        sizes=dict(train=len(tr), val=len(va), test=len(te)),
    )
    print(f"  [{ds}] classes={K}  train={len(tr)}  val={len(va)}  test={len(te)}")
    return d


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, L = [], []
    for x, y in loader:
        with autocast():
            out = model(x.to(DEVICE))
        P.append(out.argmax(-1).cpu())
        L.append(y)
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def run_one(mid, seed, epochs, smoke):
    paper, ds, build_model, build_pre, lr, wd = SPECS[mid]
    set_seed(seed)
    data = get_data(ds, build_pre(), smoke)
    K = data["K"]
    model = build_model(K).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{mid}] {paper} | {ds} | trainable params {trainable:,} | lr={lr} wd={wd}")
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=wd)
    sched = warmup_cosine(opt, epochs, WARMUP)
    scaler = GradScaler()
    hist = dict(epoch=[], train_loss=[], train_acc=[], val_acc=[])
    best_val, best_ep, test_res = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl_sum, correct, total = 0.0, 0, 0
        for x, y in data["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
            tl_sum += loss.item() * y.size(0)
        sched.step()
        vp, vl = predict(model, data["val"])
        va = 100.0 * float((vp == vl).mean())
        star = ""
        if va > best_val:                      # selection uses VAL only
            best_val, best_ep, star = va, ep, "  *"
            tp, tl_ = predict(model, data["test"])
            test_res = metrics(tp, tl_, K)     # test is only RECORDED at the best-VAL epoch
        hist["epoch"].append(ep); hist["train_loss"].append(round(tl_sum / total, 4))
        hist["train_acc"].append(round(100.0 * correct / total, 2)); hist["val_acc"].append(round(va, 2))
        print(f"    ep {ep:02d}/{epochs}  loss={tl_sum/total:.4f}  train={100.0*correct/total:.2f}  val={va:.2f}{star}")
    wall = round(time.time() - t0, 1)
    peak = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None
    print(f"  DONE {mid} {ds} seed {seed}: best-val {best_val:.2f} (ep {best_ep}) | "
          f"TEST acc {test_res['acc']:.2f} macro-F1 {test_res['macro_f1']:.2f} [{wall/60:.1f} min]")
    return dict(method=mid, paper=paper, dataset=ds, seed=seed, epochs=epochs,
                trainable_params=trainable, lr=lr, weight_decay=wd,
                best_epoch=best_ep, best_val_acc=round(best_val, 2), test=test_res,
                history=hist, split=data["sizes"], wall_time_sec=wall, peak_gpu_gb=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None, choices=list(SPECS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    epochs = 1 if a.smoke else a.epochs
    out = os.path.join(config.BASE_DIR, "results", "published_final_smoke" if a.smoke else "published_final")
    os.makedirs(out, exist_ok=True)
    methods = a.methods or DEFAULT_ORDER
    print("=" * 70)
    print(f"  Published methods on FINAL protocol | epochs {epochs} | seeds {a.seeds} | smoke {a.smoke}")
    print(f"  Methods: {methods} | Output: {out} | Device: {DEVICE}")
    print("=" * 70)
    done = 0
    for mid in methods:
        for seed in a.seeds:
            fp = os.path.join(out, f"{mid}_{SPECS[mid][1]}_seed{seed}.json")
            if os.path.exists(fp):
                print(f"  skip (exists): {fp}")
                continue
            print(f"\n=== {mid} | {SPECS[mid][1]} | seed {seed} ===")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                res = run_one(mid, seed, epochs, a.smoke)
                with open(fp, "w") as f:
                    json.dump(res, f, indent=2)
                done += 1
            except Exception as e:
                print(f"  FAILED {mid}: {e}")
                import traceback; traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\n  Done. {done} runs completed. Results: {out}")


if __name__ == "__main__":
    main()
