"""
Selective classification via MC-Dropout predictive-entropy thresholding.

Motivation: Section "Uncertainty Quantification" of the paper already claims
predictive entropy can act as a triage signal ("route high-entropy predictions
to a clinician") but never quantifies it. This script builds the standard
risk-coverage curve for that claim: sort DS2 test predictions by MC-Dropout
predictive entropy, and show how accuracy/macro-F1 on the *retained*
(low-entropy, non-referred) subset improves as the referral rate increases.

Reuses `enable_mc_dropout`, `mc_predict`, `compute_uncertainty` from
uncertainty/mc_dropout.py unchanged (same 50-pass, BN-safe MC Dropout used
for T4) — no modification to any existing evaluation script.

Outputs:
  outputs/<run_id>/uncertainty/mc_selective_classification.json
  results/figures/risk_coverage_curve.png
  results/tables/table_selective_classification.{csv,tex}
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
from report_results import load_hmr_bilstm
from uncertainty.mc_dropout import mc_predict, compute_uncertainty

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
COVERAGE_GRID = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
REFERRAL_REPORT_COVERAGE = 0.90

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")


def risk_coverage_curve(entropy: np.ndarray, y_true: np.ndarray, preds: np.ndarray,
                         coverage_grid=COVERAGE_GRID) -> list:
    """
    At each coverage level c, retain the c*N lowest-entropy predictions
    (i.e. refer the highest-entropy 1-c fraction to a clinician) and compute
    accuracy/macro-F1 on the retained subset.
    """
    order = np.argsort(entropy)  # ascending: most confident first
    n = len(entropy)
    points = []
    for coverage in coverage_grid:
        n_keep = max(1, int(round(coverage * n)))
        keep_idx = order[:n_keep]
        acc = accuracy_score(y_true[keep_idx], preds[keep_idx])
        f1m = f1_score(y_true[keep_idx], preds[keep_idx], labels=[0, 1, 2, 3],
                        average="macro", zero_division=0)
        points.append({
            "coverage": coverage,
            "referral_rate": round(1.0 - coverage, 4),
            "n_retained": int(n_keep),
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1m), 4),
        })
    return points


def per_class_referral_rate(entropy: np.ndarray, y_true: np.ndarray, coverage: float) -> dict:
    """At a given coverage level, what fraction of each true class gets referred?"""
    order = np.argsort(entropy)
    n = len(entropy)
    n_keep = max(1, int(round(coverage * n)))
    retained = np.zeros(n, dtype=bool)
    retained[order[:n_keep]] = True

    out = {}
    for c, name in CLASS_NAMES.items():
        mask = y_true == c
        if mask.sum() > 0:
            referral_rate = float((~retained[mask]).mean())
            out[name] = {"referral_rate": round(referral_rate, 4), "n_samples": int(mask.sum())}
    return out


def plot_risk_coverage(points: list, out_path: Path) -> None:
    coverage = [p["coverage"] * 100 for p in points]
    acc = [p["accuracy"] for p in points]
    f1m = [p["f1_macro"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(coverage, acc, "o-", color="#1565C0", label="Accuracy (retained subset)")
    ax.plot(coverage, f1m, "s-", color="#C62828", label="Macro F1 (retained subset)")
    ax.set_xlabel("Coverage (% of test set NOT referred to clinician)", fontsize=12)
    ax.set_ylabel("Metric on retained subset", fontsize=12)
    ax.set_title("Selective Classification: Risk-Coverage Curve (MC-Dropout entropy)",
                 fontsize=13, fontweight="bold")
    ax.invert_xaxis()  # 100% coverage on the left, decreasing to the right
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path}")


def write_table(points: list, out_stem: str) -> None:
    header = ["Coverage (%)", "Referral (%)", "Accuracy", "Macro F1"]
    rows = [[f"{p['coverage']*100:.0f}", f"{p['referral_rate']*100:.0f}",
             f"{p['accuracy']:.4f}", f"{p['f1_macro']:.4f}"] for p in points]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{out_stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [OK] {csv_path}")

    tex_path = TABLES_DIR / f"{out_stem}.tex"
    lines = [
        "\\begin{table}[htbp]",
        "\\caption{Selective classification on the DS2 test set: accuracy and macro F1 "
        "on the retained (non-referred) subset as the MC-Dropout entropy referral rate increases.}",
        "\\label{tab:selective}",
        "\\centering",
        "\\begin{tabular}{@{}cccc@{}}",
        "\\toprule",
        "\\textbf{Coverage (\\%)} & \\textbf{Referral (\\%)} & \\textbf{Accuracy} & \\textbf{Macro F1} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {tex_path}")


def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths = build_paths(run_id)
    paths["out_uncert"].mkdir(parents=True, exist_ok=True)

    n_samples = cfg["uncertainty"].get("mc_samples", 50)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")

    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)

    print(f"Loading test data: {INTER_TEST}")
    test = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    print(f"  Test set: {len(X_test)} samples")

    print(f"Running {n_samples} MC-Dropout passes...")
    probs_mc = mc_predict(model, X_test, n_samples, device)
    unc = compute_uncertainty(probs_mc)
    entropy, preds = unc["entropy"], unc["preds"]

    print("\nBuilding risk-coverage curve...")
    points = risk_coverage_curve(entropy, y_test, preds)
    for p in points:
        print(f"  coverage={p['coverage']*100:5.1f}%  referral={p['referral_rate']*100:5.1f}%  "
              f"acc={p['accuracy']:.4f}  f1_macro={p['f1_macro']:.4f}")

    referral_by_class = per_class_referral_rate(entropy, y_test, REFERRAL_REPORT_COVERAGE)
    print(f"\nPer-class referral rate at {REFERRAL_REPORT_COVERAGE*100:.0f}% coverage:")
    for name, stats in referral_by_class.items():
        print(f"    {name}: referred={stats['referral_rate']*100:.1f}%  n={stats['n_samples']}")

    print("\nGenerating outputs...")
    plot_risk_coverage(points, FIGURES_DIR / "risk_coverage_curve.png")
    write_table(points, "table_selective_classification")

    results = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "uncertainty_entropy_thresholding",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mc_samples": n_samples,
            "coverage_grid": COVERAGE_GRID,
            "referral_report_coverage": REFERRAL_REPORT_COVERAGE,
        },
        "risk_coverage_curve": points,
        "per_class_referral_at_report_coverage": referral_by_class,
    }
    out_path = paths["out_uncert"] / "mc_selective_classification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] {out_path}")

    print("\n[Selective Classification] Complete.")


if __name__ == "__main__":
    main()
