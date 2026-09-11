"""Generate manuscript figures from the locked E1/E5 evidence record.

This script performs presentation-only rendering.  It neither trains a model nor
recomputes predictions.  Numerical constants below are the rounded values in
results/forensic/E1_notes.md and E5_notes.md; the source files are checked before
rendering so stale or missing evidence cannot silently produce a figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "images"


def require_text(path: Path, fragments: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"Evidence check failed for {path}: {missing}")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def patient_composition() -> None:
    # Locked DS1/DS2 composition statement in the controlled-CV report/manuscript.
    values = [93, 75]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    bars = ax.bar(["F: subject 213", "S: subject 232"], values,
                  color=["#a23b72", "#2e86ab"], width=0.58)
    ax.set_ylabel("Share of DS2 class beats (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="0.55", linestyle="--", linewidth=1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value}%",
                ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "ds2_rare_class_concentration.png")


def f_error_deciles() -> None:
    # E1 headline allocation of all true-F -> N errors by P(F) rank region.
    regions = ["Bottom 3\ndeciles", "Middle 4\ndeciles", "Top 3\ndeciles"]
    values = [48.7, 46.8, 4.5]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    bars = ax.bar(regions, values, color=["#b2182b", "#f4a582", "#2166ac"])
    ax.set_ylabel("Share of all true-F → N errors (%)")
    ax.set_ylim(0, 55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%",
                ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "e1_f_error_rank_regions.png")


def subtype_auc() -> None:
    folds = np.arange(5)
    series = {
        "R2, MIT-BIH A": [0.664, 0.978, 0.956, 0.977, 0.785],
        "Flat, MIT-BIH A": [0.367, 0.882, 0.823, 0.917, 0.437],
        "R2, INCART A": [0.923, 0.989, 0.939, 0.993, 0.991],
    }
    styles = [("#1b9e77", "o", "-"), ("#d95f02", "s", "-"),
              ("#7570b3", "^", "--")]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for (label, values), (color, marker, line) in zip(series.items(), styles):
        ax.plot(folds, values, label=label, color=color, marker=marker,
                linestyle=line, linewidth=2, markersize=6)
    ax.set(xlabel="Outer fold", ylabel="Subtype-A ROC AUC", xticks=folds,
           ylim=(0.3, 1.02))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "e5_subtype_a_auc.png")


def f_geometry() -> None:
    tertiles = np.arange(1, 4)
    # Pre-head embedding, cosine distance to the patient's N centroid.
    pred_n = [96.8, 29.2, 0.5]
    pred_f = [2.4, 30.7, 36.6]
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.bar(tertiles - width / 2, pred_n, width, label="Predicted N", color="#4c78a8")
    ax.bar(tertiles + width / 2, pred_f, width, label="Predicted F", color="#e45756")
    ax.set(xlabel="Tertile of distance to patient N centroid (near → far)",
           ylabel="Fraction of true-F samples (%)", xticks=tertiles, ylim=(0, 105))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "e5_f_patient_geometry.png")


def main() -> None:
    e1 = ROOT / "results" / "forensic" / "E1_notes.md"
    e5 = ROOT / "results" / "forensic" / "E5_notes.md"
    manuscript = ROOT / "latex" / "sn-article.tex"
    require_text(e1, ["48.7%", "4.5%", "0.785 | 0.001"])
    require_text(e5, ["0:0.664", "0:0.367", "0:0.923", "0.968", "0.366"])
    require_text(manuscript, ["93\\% of DS2 F beats", "75\\% of DS2 S beats"])
    patient_composition()
    f_error_deciles()
    subtype_auc()
    f_geometry()


if __name__ == "__main__":
    main()
