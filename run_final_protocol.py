"""
AgroOnto-VLM -- FINAL evaluation protocol (fair, multi-seed, held-out test)
===========================================================================
 * TEST set  = the original "val" folder/csv of each dataset (never used for any selection).
 * VAL set   = 15 % stratified hold-out carved from the original TRAIN set (fixed split seed).
 * Epoch selection (best val) and lambda_onto selection use ONLY the new VAL set.
 * All four methods M1-M4 get the SAME seeds, the SAME splits, NO "search until it wins".
 * Per run we store test accuracy / macro-F1 / weighted-F1 / confusion matrix at the best-VAL epoch.

Usage (run from the AgroOnto-VLM folder):
  python run_final_protocol.py --smoke                      # 2-minute pipeline check
  python run_final_protocol.py --stage sweep                # choose lambda_onto on VAL (M1, seed 42)
  python run_final_protocol.py --stage final                # M1-M4 x 3 datasets x 3 seeds
  python run_final_protocol.py --stage final --datasets IP102 --methods M1 --seeds 42   # a single run
Results: results\\final_protocol\\*.json   (already finished runs are skipped -> safe to stop/restart)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agroonto import config
from agroonto.dataset import build_loaders, get_clip_preprocess
from agroonto.ontology import compute_similarity_matrix
from agroonto.model import AgroOntoVLM
from agroonto.train import train_one_epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = ["PlantVillageTomato", "DeepWeeds", "IP102"]
DEFAULT_LAMBDA = {"PlantVillageTomato": 0.05, "DeepWeeds": 0.10, "IP102": 0.125}
SWEEP_LAMBDAS = [0.05, 0.10, 0.125]
WARMUP = 3
VAL_FRAC = 0.15
SPLIT_SEED = 2024          # fixed: identical train/val split for every method and seed

_cache = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def warmup_cosine(optimizer, total, warmup):
    def f(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(total - warmup, 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def stratified_split(labels, frac, seed):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(frac * len(idx))) if len(idx) >= 2 else 0
        n_val = min(n_val, len(idx) - 1)
        va.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    return sorted(tr), sorted(va)


def get_data(ds, smoke=False):
    key = (ds, smoke)
    if key in _cache:
        return _cache[key]
    pre = get_clip_preprocess()
    train_loader, test_loader, names, K = build_loaders(ds, pre)
    base = train_loader.dataset
    labels = [s[1] for s in base.samples]
    tr, va = stratified_split(labels, VAL_FRAC, SPLIT_SEED)
    test_ds = test_loader.dataset
    te = list(range(len(test_ds)))
    if smoke:
        tr, va, te = tr[:160], va[:64], te[:64]
    bs = config.BATCH_SIZE
    kw = dict(num_workers=config.NUM_WORKERS, pin_memory=True)
    d = dict(
        names=names, K=K,
        train=DataLoader(Subset(base, tr), batch_size=bs, shuffle=True, drop_last=True, **kw),
        val=DataLoader(Subset(base, va), batch_size=bs, shuffle=False, **kw),
        test=DataLoader(Subset(test_ds, te), batch_size=bs, shuffle=False, **kw),
        # un-shuffled copy of train for feature caching (M4)
        train_plain=DataLoader(Subset(base, tr), batch_size=bs, shuffle=False, **kw),
        sizes=dict(train=len(tr), val=len(va), test=len(te)),
    )
    print(f"  [{ds}] classes={K}  train={len(tr)}  val={len(va)}  test={len(te)}")
    _cache[key] = d
    return d


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, L = [], []
    for x, y in loader:
        with autocast():
            out = model(x.to(DEVICE))
        P.append(out[0].argmax(-1).cpu())
        L.append(y)
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def metrics(preds, labels, K):
    present = np.unique(labels)
    return dict(
        acc=round(100.0 * float((preds == labels).mean()), 2),
        macro_f1=round(100.0 * f1_score(labels, preds, labels=present, average="macro", zero_division=0), 2),
        weighted_f1=round(100.0 * f1_score(labels, preds, labels=present, average="weighted", zero_division=0), 2),
        n_classes_in_test=int(len(present)),
        confusion_matrix=confusion_matrix(labels, preds, labels=list(range(K))).tolist(),
    )


def val_acc_of(preds, labels):
    return 100.0 * float((preds == labels).mean())


# ----------------------------------------------------------------------------- M1 / M2 / M3
def run_lora(method, ds, seed, lam, epochs, smoke, eval_test=True):
    data = get_data(ds, smoke)
    K, names = data["K"], data["names"]
    set_seed(seed)
    onto = compute_similarity_matrix(ds, class_names=names, num_classes=K).to(DEVICE)
    model = AgroOntoVLM(num_classes=K, class_names=names, clip_model_name=config.CLIP_MODEL,
                        lora_rank=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
                        lora_blocks=config.LORA_BLOCKS, device=DEVICE).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LU = config.LAMBDA_UNIFORM
    if method == "M1":
        lce, lu, lo, lp = 1.0 - lam - LU, LU, lam, config.LAMBDA_PROTO
    elif method == "M2":
        lce, lu, lo, lp = 1.0, 0.0, 0.0, 0.0
    elif method == "M3":
        lce, lu, lo, lp = 1.0 - LU, LU, 0.0, 0.0
    else:
        raise ValueError(method)
    groups = model.get_param_groups(config.LR_LORA, config.LR_HEAD, config.LR_DELTA, config.LR_LOGIT_SCALE)
    opt = torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)
    sched = warmup_cosine(opt, epochs, WARMUP)
    scaler = GradScaler()
    hist = dict(epoch=[], train_loss=[], train_acc=[], val_acc=[])
    best_val, best_ep, test_res = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, data["train"], opt, scaler, onto, K, lce, lu, lo, lp, DEVICE)
        sched.step()
        vp, vl = predict(model, data["val"])
        va = val_acc_of(vp, vl)
        star = ""
        if va > best_val:
            best_val, best_ep, star = va, ep, "  *"
            if eval_test:
                tp, tl_ = predict(model, data["test"])
                test_res = metrics(tp, tl_, K)
        hist["epoch"].append(ep); hist["train_loss"].append(round(tl, 4))
        hist["train_acc"].append(round(ta, 2)); hist["val_acc"].append(round(va, 2))
        print(f"    ep {ep:02d}/{epochs}  loss={tl:.4f}  train={ta:.2f}  val={va:.2f}{star}")
    return dict(method=method, dataset=ds, seed=seed, lambda_onto=lam if method == "M1" else None,
                epochs=epochs, trainable_params=trainable, best_epoch=best_ep,
                best_val_acc=round(best_val, 2), test=test_res, history=hist,
                split=data["sizes"], wall_time_sec=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- M4 (linear probe)
def _features(clip_model, loader):
    F_, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            with autocast():
                f = clip_model.encode_image(x.to(DEVICE))
            f = f.float()
            F_.append((f / f.norm(dim=-1, keepdim=True)).cpu())
            Y.append(y)
    return torch.cat(F_), torch.cat(Y)


def run_linear(ds, seed, epochs, smoke):
    import clip
    data = get_data(ds, smoke)
    K = data["K"]
    key = ("feats", ds, smoke)
    if key not in _cache:
        cm, _ = clip.load(config.CLIP_MODEL, device=DEVICE, jit=False)
        cm = cm.float().eval()
        for p in cm.parameters():
            p.requires_grad = False
        print("    extracting frozen CLIP features once ...")
        _cache[key] = {s: _features(cm, data[n]) for s, n in
                       (("train", "train_plain"), ("val", "val"), ("test", "test"))}
        del cm
        torch.cuda.empty_cache()
    ft = _cache[key]
    set_seed(seed)
    head = nn.Linear(512, K).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
    sched = warmup_cosine(opt, epochs, WARMUP)
    Xtr, Ytr = ft["train"][0].to(DEVICE), ft["train"][1].to(DEVICE)
    bs = config.BATCH_SIZE
    hist = dict(epoch=[], train_loss=[], train_acc=[], val_acc=[])
    best_val, best_ep, test_res = -1.0, 0, None
    t0 = time.time()

    def pred(split):
        X, Y = ft[split]
        with torch.no_grad():
            p = head(X.to(DEVICE)).argmax(-1).cpu().numpy()
        return p, Y.numpy()

    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        tot, cor, n = 0.0, 0, 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            logits = head(Xtr[idx])
            loss = F.cross_entropy(logits, Ytr[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), config.GRAD_CLIP)
            opt.step()
            tot += loss.item() * len(idx); cor += (logits.argmax(-1) == Ytr[idx]).sum().item(); n += len(idx)
        sched.step()
        head.eval()
        vp, vl = pred("val")
        va = val_acc_of(vp, vl)
        if va > best_val:
            best_val, best_ep = va, ep
            tp, tl_ = pred("test")
            test_res = metrics(tp, tl_, K)
        hist["epoch"].append(ep); hist["train_loss"].append(round(tot / n, 4))
        hist["train_acc"].append(round(100.0 * cor / n, 2)); hist["val_acc"].append(round(va, 2))
    print(f"    best val {best_val:.2f} @ep{best_ep}  test {test_res['acc']:.2f}")
    return dict(method="M4", dataset=ds, seed=seed, lambda_onto=None, epochs=epochs,
                trainable_params=sum(p.numel() for p in head.parameters()), best_epoch=best_ep,
                best_val_acc=round(best_val, 2), test=test_res, history=hist,
                split=data["sizes"], wall_time_sec=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "final"], default="final")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--methods", nargs="+", default=["M1", "M2", "M3", "M4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check (1 epoch, few images)")
    ap.add_argument("--force", action="store_true", help="re-run even if the result file exists")
    a = ap.parse_args()

    out = os.path.join(config.BASE_DIR, "results", "final_protocol_smoke" if a.smoke else "final_protocol")
    os.makedirs(out, exist_ok=True)
    epochs = 1 if a.smoke else a.epochs
    lam_file = os.path.join(out, "chosen_lambda.json")
    print(f"Device: {DEVICE} | stage={a.stage} | epochs={epochs} | out={out}")
    if DEVICE == "cpu" and not a.smoke:
        print("WARNING: no GPU detected - this will be extremely slow.")

    if a.stage == "sweep":
        chosen = json.load(open(lam_file)) if os.path.exists(lam_file) else {}
        for ds in a.datasets:
            scores = {}
            for lam in SWEEP_LAMBDAS:
                f = os.path.join(out, f"sweep_{ds}_lam{lam}.json")
                if os.path.exists(f) and not a.force:
                    r = json.load(open(f))
                else:
                    print(f"\n=== SWEEP {ds}  lambda_onto={lam}  seed=42 ===")
                    r = run_lora("M1", ds, 42, lam, epochs, a.smoke, eval_test=False)
                    json.dump(r, open(f, "w"))
                scores[lam] = r["best_val_acc"]
                print(f"  -> lambda {lam}: best VAL acc {scores[lam]:.2f}")
            chosen[ds] = max(scores, key=scores.get)
            print(f"  CHOSEN lambda_onto for {ds}: {chosen[ds]}  (selected on the validation split only)")
            json.dump(chosen, open(lam_file, "w"), indent=2)
        return

    chosen = json.load(open(lam_file)) if os.path.exists(lam_file) else dict(DEFAULT_LAMBDA)
    print("lambda_onto used for M1:", chosen)
    for ds in a.datasets:
        for seed in a.seeds:
            for m in a.methods:
                f = os.path.join(out, f"{m}_{ds}_seed{seed}.json")
                if os.path.exists(f) and not a.force:
                    print(f"skip (done): {os.path.basename(f)}")
                    continue
                print(f"\n=== {m} | {ds} | seed {seed} ===")
                try:
                    r = run_linear(ds, seed, epochs, a.smoke) if m == "M4" else \
                        run_lora(m, ds, seed, chosen.get(ds, DEFAULT_LAMBDA[ds]), epochs, a.smoke)
                except Exception:
                    import traceback; traceback.print_exc()
                    continue
                json.dump(r, open(f, "w"))
                t = r["test"]
                print(f"  DONE {m} {ds} seed {seed}: best-val {r['best_val_acc']:.2f} (ep {r['best_epoch']}) "
                      f"| TEST acc {t['acc']:.2f}  macro-F1 {t['macro_f1']:.2f}  [{r['wall_time_sec']/60:.1f} min]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print("\nAll requested runs finished.")


if __name__ == "__main__":
    main()
