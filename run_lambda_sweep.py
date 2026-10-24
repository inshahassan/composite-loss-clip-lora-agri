"""
Lambda sweep: run all (dataset × λ_onto) combinations.
Usage:  python run_lambda_sweep.py
        python run_lambda_sweep.py --dataset PlantVillage
        python run_lambda_sweep.py --dataset IP102 --lambda_onto 0.05
"""
import argparse
import json
import os
import sys
import time
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agroonto import config
from agroonto.train import run_training


def main():
    parser = argparse.ArgumentParser(description="AgroOnto-VLM λ_onto sweep")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset (PlantVillage, DeepWeeds, IP102)")
    parser.add_argument("--lambda_onto", type=float, default=None,
                        help="Run only this lambda value")
    args = parser.parse_args()

    # GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu}  ({mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected — running on CPU")

    # Determine which runs to do
    datasets = [args.dataset] if args.dataset else ["PlantVillage", "DeepWeeds", "IP102"]
    lambdas = [args.lambda_onto] if args.lambda_onto else config.SWEEP_LAMBDAS

    total_runs = len(datasets) * len(lambdas)
    print(f"\nTotal runs: {total_runs}  "
          f"({len(datasets)} datasets × {len(lambdas)} λ values)")
    print(f"Epochs per run: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Results dir: {config.RESULTS_DIR}\n")

    all_results = []
    run_idx = 0

    for ds_name in datasets:
        for lam in lambdas:
            run_idx += 1
            print(f"\n{'#'*60}")
            print(f"  RUN {run_idx}/{total_runs}: {ds_name} × λ_onto={lam}")
            print(f"{'#'*60}")

            t0 = time.time()
            try:
                result = run_training(ds_name, lam)
                result["wall_time_sec"] = round(time.time() - t0, 1)
                all_results.append(result)
                print(f"\n  Run completed in {result['wall_time_sec']:.0f}s")
            except Exception as e:
                print(f"\n  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "dataset": ds_name,
                    "lambda_onto": lam,
                    "error": str(e),
                })

            # Clear GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save summary
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(config.RESULTS_DIR, "sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*70}")
    print("  SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<15} {'λ_onto':<8} {'Best Val%':<10} {'Epoch':<6} {'Time':<8}")
    print(f"  {'-'*47}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['dataset']:<15} {r['lambda_onto']:<8} FAILED")
        else:
            print(f"  {r['dataset']:<15} {r['lambda_onto']:<8.3f} "
                  f"{r['best_val_acc']:<10.2f} {r['best_epoch']:<6} "
                  f"{r.get('wall_time_sec', 0):<8.0f}s")
    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    main()
