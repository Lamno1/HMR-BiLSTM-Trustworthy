"""
Multi-seed benchmark runner for the modern-architecture comparison.

Runs train_modern.py once per (model, seed) as a subprocess (full isolation —
no shared global/CUDA state between runs), then aggregates test metrics into
mean +/- std per model. cnn_mamba is excluded by default (see README.md: too
slow for multi-seed on this hardware, and showed no early promise vs the
others at matched epoch count).

Usage:
    python run_benchmark.py --seeds 42 43 44 --epochs 45
    python run_benchmark.py --models bilstm cnn_transformer --seeds 42 43 --quick
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

DEFAULT_MODELS = ["bilstm", "cnn_transformer", "pure_transformer", "cnn_xlstm"]
METRICS_OF_INTEREST = ["f1_macro", "f1_N", "f1_S", "f1_V", "f1_F", "f1_Q", "accuracy", "auc_ovr"]


def run_one(model, seed, epochs, quick, out_root):
    out_dir = out_root / f"{model}_s{seed}"
    log_path = out_dir / "log.json"
    if log_path.exists():
        print(f"[skip] {model} seed={seed} already has log.json at {log_path}")
        return log_path

    cmd = [
        sys.executable, "train_modern.py",
        "--model", model, "--seed", str(seed),
        "--epochs", str(epochs), "--out-dir", str(out_dir),
    ]
    if quick:
        cmd.append("--quick")
    print(f"\n[run] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"train_modern.py failed for model={model} seed={seed} (exit {result.returncode})")
    return log_path


def aggregate(logs_by_model):
    summary = {}
    for model, log_paths in logs_by_model.items():
        per_seed = []
        for lp in log_paths:
            with open(lp) as f:
                d = json.load(f)
            per_seed.append(d["test_metrics"])
        summary[model] = {"n_seeds": len(per_seed), "per_seed": per_seed}
        for metric in METRICS_OF_INTEREST:
            vals = [m[metric] for m in per_seed]
            summary[model][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return summary


def print_table(summary):
    header = f"{'model':<18}" + "".join(f"{m:>14}" for m in METRICS_OF_INTEREST)
    print(header)
    print("-" * len(header))
    for model, s in summary.items():
        row = f"{model:<18}"
        for metric in METRICS_OF_INTEREST:
            row += f"{s[metric]['mean']:>8.4f}+/-{s[metric]['std']:<5.3f}"
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=["bilstm", "cnn_transformer", "cnn_mamba", "cnn_xlstm", "pure_transformer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-root", default="results")
    args = p.parse_args()

    if "cnn_mamba" in args.models:
        print("[WARNING] cnn_mamba is ~80x slower/epoch than the other backbones on this "
              "hardware and showed no early promise at matched epoch count (see README.md). "
              "Proceeding anyway since you explicitly requested it.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    logs_by_model = {m: [] for m in args.models}
    for model in args.models:
        for seed in args.seeds:
            log_path = run_one(model, seed, args.epochs, args.quick, out_root)
            logs_by_model[model].append(log_path)

    summary = aggregate(logs_by_model)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (mean±std across seeds)")
    print("=" * 80)
    print_table(summary)

    summary_path = out_root / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    print(f"\n[OK] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
