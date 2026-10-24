"""Summarise results\\final_protocol: mean +/- std over seeds (test split), paired M1-M2 / M1-M3 differences."""
import glob, json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "results", "final_protocol_smoke" if "--smoke" in sys.argv else "final_protocol")
DS = ["PlantVillageTomato", "DeepWeeds", "IP102"]
MS = ["M1", "M2", "M3", "M4"]

runs = {}
for f in glob.glob(os.path.join(D, "M?_*_seed*.json")):
    r = json.load(open(f))
    runs[(r["method"], r["dataset"], r["seed"])] = r

def vals(m, ds, key):
    seeds = sorted(s for (mm, d, s) in runs if mm == m and d == ds)
    return seeds, np.array([runs[(m, ds, s)]["test"][key] for s in seeds])

lines = []
summary = {}
for key, title in (("acc", "TEST top-1 accuracy (%)"), ("macro_f1", "TEST macro-F1 (%)")):
    lines.append(f"\n{title}  -- mean +/- std over seeds (n)")
    lines.append(f"{'Method':8}" + "".join(f"{d:>24}" for d in DS))
    for m in MS:
        row = f"{m:8}"
        for d in DS:
            s, v = vals(m, d, key)
            if len(v):
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                row += f"{v.mean():>14.2f} +/- {sd:4.2f} ({len(v)})"
                summary[f"{m}|{d}|{key}"] = dict(mean=float(v.mean()), std=float(sd), per_seed=dict(zip(map(str, s), v.tolist())))
            else:
                row += f"{'-':>24}"
        lines.append(row)

lines.append("\nPaired differences M1 - baseline on TEST accuracy (pp), per seed [mean, wins/seeds]")
for other in ("M2", "M3"):
    for d in DS:
        s1, v1 = vals("M1", d, "acc"); s2, v2 = vals(other, d, "acc")
        common = sorted(set(s1) & set(s2))
        if not common:
            continue
        diff = np.array([runs[("M1", d, s)]["test"]["acc"] - runs[(other, d, s)]["test"]["acc"] for s in common])
        lines.append(f"  M1-{other} {d:20}: " + "  ".join(f"{x:+.2f}" for x in diff) +
                     f"   mean {diff.mean():+.2f}  wins {(diff > 0).sum()}/{len(diff)}")

lines.append("\nSelected epochs / val acc (M1):")
for d in DS:
    for (m, dd, s), r in sorted(runs.items()):
        if m == "M1" and dd == d:
            lines.append(f"  {d:20} seed {s}: best epoch {r['best_epoch']}, val {r['best_val_acc']:.2f}, "
                         f"test {r['test']['acc']:.2f}, lambda {r['lambda_onto']}")

txt = "\n".join(lines)
print(txt)
open(os.path.join(D, "SUMMARY.txt"), "w", encoding="utf-8").write(txt)
json.dump(summary, open(os.path.join(D, "summary.json"), "w"), indent=2)
print("\nSaved:", os.path.join(D, "SUMMARY.txt"))
