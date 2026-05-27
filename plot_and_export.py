"""
plot_and_export.py
------------------
1. Vẽ biểu đồ Ablation Study (FGSM robustness per variant)
2. Xuất Bảng Baseline đầy đủ (CSV + LaTeX): Clean + FGSM + PGD + Calibration
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# PART 1 – ABLATION STUDY CHARTS
# ──────────────────────────────────────────────────────────────────────────────

# Load ablation data
ablation_rows = []
with open(TABLES / "ablation_table_final.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ablation_rows.append(row)

labels   = [r["Variant"] for r in ablation_rows]
f1_clean = [float(r["F1-Clean"]) for r in ablation_rows]
f1_002   = [float(r["F1-Adv (0.02)"]) for r in ablation_rows]
f1_005   = [float(r["F1-Adv (0.05)"]) for r in ablation_rows]

# ── SHORT labels for readability ──
SHORT = {
    "HMR-BiLSTM (full)":              "Full",
    "No-RMC (c_t = c_lstm)":          "No-RMC",
    "No-CNN (raw input)":              "No-CNN",
    "Mean-Pool (no attention)":        "No-Attn",
    "No-Adv-Training":                 "No-Adv",
    "No-Hybrid-Path (c_t = c_rmc)":   "No-Hybrid",
    "No-Smoothness (lambda=0)":        "No-Smooth",
}
short_labels = [SHORT.get(l, l) for l in labels]

# ── COLORS: full = green, others = shades of gray/blue
COLORS_CLEAN = ["#2ca02c" if "Full" in s else "#4c72b0" for s in short_labels]
COLORS_002   = ["#1a7a1a" if "Full" in s else "#2d5886" for s in short_labels]
COLORS_005   = ["#0f4d0f" if "Full" in s else "#1c3652" for s in short_labels]

x = np.arange(len(short_labels))
W = 0.26

# ── Figure 1: Grouped bar — Clean vs FGSM 0.02 vs FGSM 0.05 ──
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars1 = ax.bar(x - W, f1_clean, W, label="F1 Clean",       color=COLORS_CLEAN, alpha=0.95, zorder=3)
bars2 = ax.bar(x,     f1_002,   W, label="F1 Adv (ε=0.02)",color=COLORS_002,   alpha=0.90, zorder=3)
bars3 = ax.bar(x + W, f1_005,   W, label="F1 Adv (ε=0.05)",color=COLORS_005,   alpha=0.85, zorder=3)

for bar, v in zip(bars1, f1_clean):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold", rotation=45)
for bar, v in zip(bars2, f1_002):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color="#ccddff", rotation=45)
for bar, v in zip(bars3, f1_005):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7, color="#aabbdd", rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, color="#cccccc", fontsize=10, fontweight="bold")
ax.set_ylim(0.40, 1.02)
ax.set_ylabel("F1-macro", color="#cccccc", fontsize=11)
ax.set_title("Ablation Study: Clean vs Adversarial F1 (FGSM)", color="white",
             fontsize=13, fontweight="bold", pad=14)
ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values(): sp.set_color("#333344")
ax.yaxis.grid(True, color="#2a2d3a", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

legend = ax.legend(fontsize=9, framealpha=0.25, labelcolor="white",
                   facecolor="#1a1d27", edgecolor="#444455")
fig.suptitle("HMR-BiLSTM — Ablation Study (MIT-BIH ECG)",
             color="white", fontsize=14, fontweight="bold", y=0.99)

plt.tight_layout()
out1 = FIGURES / "ablation_clean_vs_adv.png"
plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[FIG] {out1}")

# ── Figure 2: F1 Drop bar chart ──
drops = [(c - a) / c * 100 for c, a in zip(f1_clean, f1_002)]
DROP_COLORS = ["#e74c3c" if d > 6 else "#f39c12" if d > 4 else "#2ecc71"
               for d in drops]

fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.patch.set_facecolor("#0f1117")
ax2.set_facecolor("#1a1d27")

bars_d = ax2.bar(x, drops, 0.55, color=DROP_COLORS, alpha=0.9, zorder=3,
                 edgecolor="#222233", linewidth=0.8)
for bar, v in zip(bars_d, drops):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.08, f"{v:.2f}%",
             ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, color="#cccccc", fontsize=10, fontweight="bold")
ax2.set_ylabel("F1 Drop (%) under FGSM ε=0.02", color="#cccccc", fontsize=11)
ax2.set_title("F1 Drop under FGSM Attack — Ablation Variants",
              color="white", fontsize=13, fontweight="bold", pad=12)
ax2.tick_params(colors="#aaaaaa")
for sp in ax2.spines.values(): sp.set_color("#333344")
ax2.yaxis.grid(True, color="#2a2d3a", linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)

# Legend for color meaning
patches = [
    mpatches.Patch(color="#2ecc71", label="Robust (drop ≤ 4%)"),
    mpatches.Patch(color="#f39c12", label="Moderate (4%–6%)"),
    mpatches.Patch(color="#e74c3c", label="Vulnerable (drop > 6%)"),
]
ax2.legend(handles=patches, fontsize=9, framealpha=0.25, labelcolor="white",
           facecolor="#1a1d27", edgecolor="#444455")

plt.tight_layout()
out2 = FIGURES / "ablation_f1_drop.png"
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"[FIG] {out2}")


# ──────────────────────────────────────────────────────────────────────────────
# PART 2 – FULL BASELINE TABLE (CSV + LaTeX)
# ──────────────────────────────────────────────────────────────────────────────

# Hard-coded from existing tables (clean + FGSM + PGD + Calib)
# Data verified from generated CSVs
import csv
csv_in = TABLES / "baseline_full_comparison.csv"
BASELINE_DATA = []
if csv_in.exists():
    with open(csv_in, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            f1c = float(row["F1-macro"]) if row["F1-macro"] != "-" else None
            fgsm_d = float(row["FGSM-Drop(0.02)"].replace("%",""))/100 if row["FGSM-Drop(0.02)"] != "-" else None
            pgd_d = float(row["PGD-Drop(0.02)"].replace("%",""))/100 if row["PGD-Drop(0.02)"] != "-" else None
            
            def get_f(key):
                v = row.get(key, "-")
                return float(v) if v != "-" else None
            
            BASELINE_DATA.append((
                row["Model"], get_f("Acc"), get_f("Prec"), get_f("Rec"), get_f("F1-macro"), get_f("F1-w"), get_f("AUC"),
                f1c, get_f("FGSM-F1(0.02)"), fgsm_d, get_f("FGSM-ASR"), None,
                get_f("PGD-F1(0.02)"), pgd_d, get_f("PGD-ASR"), None,
                get_f("ECE↓"), get_f("Brier↓")
            ))

HEADER_FULL = [
    "Model",
    "Acc", "Prec", "Rec", "F1-macro", "F1-w", "AUC",
    "FGSM-F1(0.02)", "FGSM-Drop(0.02)", "FGSM-ASR",
    "PGD-F1(0.02)", "PGD-Drop(0.02)", "PGD-ASR",
    "ECE↓", "Brier↓",
]

def fmt(v, d=4):
    if v is None:
        return "-"
    return f"{v:.{d}f}"

def drop_pct(clean, adv):
    if clean is None or adv is None:
        return "-"
    return f"{(clean - adv)/clean*100:.2f}%"

rows_full = []
for d in BASELINE_DATA:
    (model, acc, prec, rec, f1m, f1w, auc,
     f1c, fgsm002, fgsm_drop, fgsm_asr, _,
     pgd002, pgd_drop, pgd_asr, pgd005,
     ece, brier) = d
    rows_full.append([
        model,
        fmt(acc), fmt(prec), fmt(rec), fmt(f1m), fmt(f1w), fmt(auc),
        fmt(fgsm002), drop_pct(f1c, fgsm002), fmt(fgsm_asr),
        fmt(pgd002),  drop_pct(f1c, pgd002),  fmt(pgd_asr),
        fmt(ece), fmt(brier),
    ])




# ──────────────────────────────────────────────────────────────────────────────
# PART 3 – SUMMARY FIGURE: Baseline vs Ablation side-by-side
# ──────────────────────────────────────────────────────────────────────────────

fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.patch.set_facecolor("#0f1117")

BG = "#1a1d27"
GRID_C = "#2a2d3a"
TC = "#cccccc"

def style_ax(ax, title, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=TC, fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#333344")
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# Panel A: Clean F1 baselines
bmodels = ["LSTM", "BiLSTM", "HMR-BiLSTM\n(no Adv)", "HMR-BiLSTM"]
bclean  = [0.8691, 0.8616, 0.9023, 0.8825]
bcolors = ["#4c72b0", "#55a868", "#dd8452", "#2ca02c"]
ax = axes[0]
bars = ax.bar(range(4), bclean, color=bcolors, alpha=0.9, zorder=3, width=0.6)
for bar, v in zip(bars, bclean):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.4f}",
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(bmodels, color=TC, fontsize=8)
ax.set_ylim(0.75, 1.0)
style_ax(ax, "A  Baseline Clean F1-macro", "F1-macro")

# Panel B: FGSM F1 @ 0.02 — Baselines vs Ablation full
ax = axes[1]
bfgsm = [0.8189, 0.7732, 0.8046, 0.8425]
bcolors2 = ["#4c72b0", "#55a868", "#dd8452", "#2ca02c"]
bars2 = ax.bar(range(4), bfgsm, color=bcolors2, alpha=0.9, zorder=3, width=0.6)
for bar, v in zip(bars2, bfgsm):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.4f}",
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(bmodels, color=TC, fontsize=8)
ax.set_ylim(0.65, 0.95)
style_ax(ax, "B  FGSM-F1 (ε=0.02)", "F1-macro (adversarial)")

# Panel C: Ablation — F1 drop under FGSM 0.02
ax = axes[2]
bars3 = ax.bar(x, drops, 0.55, color=DROP_COLORS, alpha=0.9, zorder=3,
               edgecolor="#222233", linewidth=0.8)
for bar, v in zip(bars3, drops):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, color=TC, fontsize=8, rotation=15, ha="right")
ax.set_ylim(0, 15)
style_ax(ax, "C  Ablation F1-Drop (ε=0.02)", "F1 Drop (%)")

fig3.suptitle("HMR-BiLSTM — Trustworthy AI: Robustness Summary",
              color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out3 = FIGURES / "robustness_summary.png"
plt.savefig(out3, dpi=300, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print(f"[FIG] {out3}")

print("\n=== DONE ===")
print(f"  Figures : {FIGURES}/ablation_clean_vs_adv.png")
print(f"            {FIGURES}/ablation_f1_drop.png")
print(f"            {FIGURES}/robustness_summary.png")
print(f"  Tables  : {TABLES}/baseline_full_comparison.csv")
print(f"            {TABLES}/baseline_full_comparison.tex")
