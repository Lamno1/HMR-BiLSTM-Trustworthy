"""Generate manuscript Phase-3 figures only from frozen final artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "latex" / "images"
RESULT_FIG = ROOT / "results" / "figures"


def _normalize_png(path: Path) -> None:
    """Remove backend metadata and encode the rendered pixels deterministically."""
    with Image.open(path) as image:
        pixels = image.convert("RGBA")
        pixels.save(path, format="PNG", compress_level=9, optimize=False)


def fold_pairs() -> None:
    report = json.loads((ROOT / "results/p1_gate/report.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    colors = {"flat": "#777777", "r2": "#1769aa"}
    for ax, cls in zip(axes, ("F", "S")):
        for item in report["per_fold"]:
            fold = item["fold"]
            flat = np.asarray(item["flat"]["mit"][cls]["per_seed"], float)
            r2 = np.asarray(item["r2"]["mit"][cls]["per_seed"], float)
            for j in range(2):
                ax.plot([fold - .10, fold + .10], [flat[j], r2[j]], color="#bbbbbb", lw=.8, zorder=1)
            ax.scatter(np.full(2, fold - .10), flat, s=22, color=colors["flat"], alpha=.65, zorder=2)
            ax.scatter(np.full(2, fold + .10), r2, s=22, color=colors["r2"], alpha=.65, zorder=2)
            ax.scatter(fold - .10, flat.mean(), s=75, facecolor="white", edgecolor=colors["flat"], lw=2, zorder=3)
            ax.scatter(fold + .10, r2.mean(), s=75, facecolor="white", edgecolor=colors["r2"], lw=2, zorder=3)
        ax.set_title(f"MIT-BIH F1-{cls}")
        ax.set_xlabel("Outer fold")
        ax.set_xticks(range(5))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("F1 score")
    axes[1].scatter([], [], color=colors["flat"], label="Flat (seed)")
    axes[1].scatter([], [], color=colors["r2"], label="R2 (seed)")
    axes[1].scatter([], [], s=75, facecolor="white", edgecolor="#333", lw=2, label="Fold mean")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "lopo_fold_pairs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _normalize_png(out)


def risk_coverage() -> None:
    with (ROOT / "results/p2_final/continuous_s_risk_coverage.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.1), sharey=True)
    colors = {"flat": "#777777", "r2": "#1769aa"}
    for fold, ax in enumerate(axes):
        for model in ("flat", "r2"):
            for seed, ls in (("42", "-"), ("123", "--")):
                rr = [r for r in rows if r["dataset"] == "mitbih" and r["model"] == model
                      and int(r["fold"]) == fold and r["seed"] == seed
                      and r["overall_selective_risk"]]
                rr.sort(key=lambda r: float(r["alpha"]))
                ax.plot([float(r["overall_coverage"]) for r in rr],
                        [float(r["overall_selective_risk"]) for r in rr],
                        color=colors[model], ls=ls, lw=1.25, alpha=.9,
                        label=f"{model.upper()} seed {seed}" if fold == 0 else None)
        ax.axhline(.02, color="#b51f1f", lw=1, ls=":")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Auto-decision coverage")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Overall selective risk")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    RESULT_FIG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    out = RESULT_FIG / "risk_coverage_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "p2_final_risk_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _normalize_png(out)
    _normalize_png(FIG / "p2_final_risk_coverage.png")


if __name__ == "__main__":
    fold_pairs()
    risk_coverage()
    print("wrote latex/images/lopo_fold_pairs.png, latex/images/p2_final_risk_coverage.png, results/figures/risk_coverage_curve.png")
