"""
aggregate_extra.py — Summarise low-data experiment results
===========================================================

Reads all JSONs from results/extra_experiments/ and prints:
  1. Mean ± std of test_acc and test_macro_f1 per (method, dataset, fraction)
  2. Paired M1−M2 difference per (dataset, fraction, seed)
  3. Mean ± std of M1−M2 difference per (dataset, fraction)

Usage:
  python aggregate_extra.py
  python aggregate_extra.py --results_dir results/extra_experiments_smoke
  python aggregate_extra.py --csv out.csv
"""

import argparse
import json
import os
import sys

import numpy as np


def load_results(results_dir):
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")
    return records


def group_records(records):
    """Index by (method, dataset, fraction, seed)."""
    idx = {}
    for r in records:
        key = (r["method"], r["dataset"], str(r["fraction"]), int(r["seed"]))
        idx[key] = r
    return idx


def aggregate(records_dir):
    records = load_results(records_dir)
    if not records:
        print(f"No JSON files found in {records_dir}")
        return []

    idx = group_records(records)

    methods   = sorted({r["method"]   for r in records})
    datasets  = sorted({r["dataset"]  for r in records})
    fractions = sorted({str(r["fraction"]) for r in records},
                       key=lambda x: float(x))
    seeds     = sorted({int(r["seed"]) for r in records})

    rows = []  # for CSV export

    print("=" * 80)
    print("  Low-data experiment summary")
    print(f"  Results dir : {records_dir}")
    print(f"  Methods     : {methods}")
    print(f"  Datasets    : {datasets}")
    print(f"  Fractions   : {fractions}")
    print(f"  Seeds       : {seeds}")
    print("=" * 80)

    # ── Table 1: Mean ± std per (method, dataset, fraction) ──────────
    print(f"\n{'─'*80}")
    print("  TABLE 1 — Mean ± Std  (Test Accuracy % | Test Macro-F1 %)")
    print(f"{'─'*80}")
    hdr = (f"  {'Method':<22} {'Dataset':<22} {'Frac':>5}  "
           f"{'Acc mean±std':>16}  {'F1 mean±std':>16}  "
           f"{'N':>3}")
    print(hdr)
    print("  " + "-" * 78)

    for method in methods:
        for dataset in datasets:
            for frac in fractions:
                accs, f1s = [], []
                for seed in seeds:
                    key = (method, dataset, frac, seed)
                    if key in idx:
                        accs.append(idx[key]["test_acc"])
                        f1s.append(idx[key]["test_macro_f1"])
                if not accs:
                    continue
                acc_str = (f"{np.mean(accs):.2f} ± {np.std(accs, ddof=0):.2f}"
                           if len(accs) > 1 else f"{accs[0]:.2f}")
                f1_str  = (f"{np.mean(f1s):.2f} ± {np.std(f1s, ddof=0):.2f}"
                           if len(f1s) > 1 else f"{f1s[0]:.2f}")
                frac_pct = int(round(float(frac) * 100))
                print(f"  {method:<22} {dataset:<22} {frac_pct:>4}%  "
                      f"{acc_str:>16}  {f1_str:>16}  {len(accs):>3}")
                rows.append({
                    "method": method, "dataset": dataset,
                    "fraction_pct": frac_pct,
                    "acc_mean": round(float(np.mean(accs)), 4),
                    "acc_std":  round(float(np.std(accs, ddof=0)), 4),
                    "f1_mean":  round(float(np.mean(f1s)),  4),
                    "f1_std":   round(float(np.std(f1s,  ddof=0)), 4),
                    "n_seeds":  len(accs),
                })

    # ── Table 2: Paired M1−M2 differences ────────────────────────────
    print(f"\n{'─'*80}")
    print("  TABLE 2 — Paired M1 − M2  (per seed, per dataset/fraction)")
    print(f"{'─'*80}")
    hdr2 = (f"  {'Dataset':<22} {'Frac':>5}  {'Seed':>6}  "
            f"{'ΔAcc':>9}  {'ΔF1':>9}  M1 acc  M2 acc")
    print(hdr2)
    print("  " + "-" * 78)

    diff_rows = []
    for dataset in datasets:
        for frac in fractions:
            dacc_list, df1_list = [], []
            for seed in seeds:
                k1 = ("M1_AgroOnto-VLM", dataset, frac, seed)
                k2 = ("M2_Baseline",     dataset, frac, seed)
                if k1 not in idx or k2 not in idx:
                    continue
                da = idx[k1]["test_acc"]     - idx[k2]["test_acc"]
                df = idx[k1]["test_macro_f1"] - idx[k2]["test_macro_f1"]
                dacc_list.append(da)
                df1_list.append(df)
                frac_pct = int(round(float(frac) * 100))
                print(f"  {dataset:<22} {frac_pct:>4}%  {seed:>6}  "
                      f"{da:>+9.2f}  {df:>+9.2f}  "
                      f"{idx[k1]['test_acc']:.2f}  {idx[k2]['test_acc']:.2f}")
                diff_rows.append({
                    "dataset": dataset, "fraction_pct": frac_pct, "seed": seed,
                    "delta_acc": round(da, 4), "delta_f1": round(df, 4),
                })

            # Mean ± std of differences
            if len(dacc_list) > 1:
                frac_pct = int(round(float(frac) * 100))
                print(f"  {'':22} {frac_pct:>4}%  {'MEAN':>6}  "
                      f"{np.mean(dacc_list):>+9.2f}±{np.std(dacc_list,ddof=0):.2f}  "
                      f"{np.mean(df1_list):>+9.2f}±{np.std(df1_list,ddof=0):.2f}")
            if dacc_list:
                print()

    # ── Table 3: M1 (real ontology) vs M1 shuffled ontology ──────────
    shuf = "M1shuffled_Ontology-Shuffled"
    if any(k[0] == shuf for k in idx):
        print(f"\n{'─'*80}")
        print("  TABLE 3 — M1 (real ontology) − M1 (SHUFFLED ontology), per seed")
        print(f"{'─'*80}")
        for dataset in datasets:
            for frac in fractions:
                dl, fl = [], []
                for seed in seeds:
                    k1 = ("M1_AgroOnto-VLM", dataset, frac, seed)
                    k3 = (shuf, dataset, frac, seed)
                    if k1 not in idx or k3 not in idx:
                        continue
                    da = idx[k1]["test_acc"] - idx[k3]["test_acc"]
                    dl.append(da)
                    fl.append(idx[k1]["test_macro_f1"] - idx[k3]["test_macro_f1"])
                    print(f"  {dataset:<22} {int(round(float(frac)*100)):>4}%  seed {seed:>4}  "
                          f"dAcc {da:>+7.2f}  real {idx[k1]['test_acc']:.2f}  shuffled {idx[k3]['test_acc']:.2f}")
                if len(dl) > 1:
                    print(f"  {'':22} {int(round(float(frac)*100)):>4}%  MEAN dAcc {np.mean(dl):>+7.2f} +- {np.std(dl):.2f} "
                          f"| dF1 {np.mean(fl):>+7.2f}  (wins {sum(x > 0 for x in dl)}/{len(dl)})")
                    print()

    # ── Table 4: label smoothing (M3) and M1 without ontology KL ─────
    for vname, vlabel in (("M3_LabelSmoothing", "M3 label smoothing"),
                          ("M1noKL_WithoutOntologyKL", "M1 without ontology-target KL")):
        if not any(k[0] == vname for k in idx):
            continue
        print(f"\n{'─'*80}")
        print(f"  TABLE 4 — {vlabel}: matched-seed difference vs M1 and vs M2 (acc points)")
        print(f"{'─'*80}")
        for dataset in datasets:
            for frac in fractions:
                d1, d2, accs = [], [], []
                for seed in seeds:
                    kv = (vname, dataset, frac, seed)
                    k1 = ("M1_AgroOnto-VLM", dataset, frac, seed)
                    k2 = ("M2_Baseline", dataset, frac, seed)
                    if kv not in idx:
                        continue
                    accs.append(idx[kv]["test_acc"])
                    if k1 in idx: d1.append(idx[kv]["test_acc"] - idx[k1]["test_acc"])
                    if k2 in idx: d2.append(idx[kv]["test_acc"] - idx[k2]["test_acc"])
                if accs:
                    print(f"  {dataset:<22} {int(round(float(frac)*100)):>4}%  acc {np.mean(accs):.2f} +- {np.std(accs):.2f} (n={len(accs)})"
                          f"  vs M1 {np.mean(d1) if d1 else float('nan'):+.2f}  vs M2 {np.mean(d2) if d2 else float('nan'):+.2f}")

    print("=" * 80)
    print(f"  Total runs loaded: {len(records)}")

    return rows, diff_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "extra_experiments"),
        help="Directory containing result JSONs")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to save summary CSV")
    args = parser.parse_args()

    out = aggregate(args.results_dir)
    if out is None:
        return

    rows, diff_rows = out

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        # Also save diffs
        diff_csv = args.csv.replace(".csv", "_diffs.csv")
        if diff_rows:
            with open(diff_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
                writer.writeheader()
                writer.writerows(diff_rows)
        print(f"\n  Saved: {args.csv}")
        print(f"  Saved: {diff_csv}")


if __name__ == "__main__":
    main()
