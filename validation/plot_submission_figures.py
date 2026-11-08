"""Render approved BEL figures from saved summaries; no training or inference.

python -m validation.plot_submission_figures --qa-tools <nature-figure/scripts>
The optional QA directory enables the installed skill's strict alignment gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
OUT = ROOT / "latex/images"
CORES = ["full", "normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]
LABEL = {"full": "ResMemECG", "normc": "No-RMC", "normc_nodrop": "No-RMC, no cell dropout",
         "bilstm": "nn.LSTM", "bilstm_nocw": "nn.LSTM, no class weights",
         "bilstm_ce": "nn.LSTM, cross-entropy", "reslstm": "Residual BiLSTM",
         "normc_lstmlike": "No-RMC, LSTM-like", "normc_torchinit": "No-RMC, nn.LSTM init.",
         "normc_nofb": "No-RMC, forget bias 0", "full_nodrop": "ResMemECG, no cell dropout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-tools", type=Path)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "legend.fontsize": 8})
    require_matplotlib_panel_alignment = None
    if args.qa_tools:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment

    def save(fig, name):
        fig.canvas.draw()
        if require_matplotlib_panel_alignment:
            require_matplotlib_panel_alignment(fig, json_out=args.out_dir / f"{name}.alignment.json", strict=True)
        fig.savefig(args.out_dir / f"{name}.pdf")
        fig.savefig(args.out_dir / f"{name}.svg")
        fig.savefig(args.out_dir / f"{name}.png", dpi=600)
        plt.close(fig)
        print("saved", name)

    # All three seeds and their sum are retained. The sum is not another replicate.
    cms = json.loads((args.tables_dir / "matched_diagnostics.json").read_text())["confusion_full"]
    fig, axes = plt.subplots(2, 2, figsize=(130.74 / 25.4, 130 / 25.4))
    fig.subplots_adjust(left=.10, right=.98, bottom=.08, top=.95, wspace=.26, hspace=.34)
    for ax, seed in zip(axes.flat, ["42", "123", "456", "sum"]):
        counts = np.asarray(cms[seed])
        fractions = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{counts[i,j]}\n{fractions[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if fractions[i,j] > .6 else "black")
        ax.set_xticks(range(5), ["N", "S", "V", "F", "Q"])
        ax.set_yticks(range(5), ["N", "S", "V", "F", "Q"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Seed " + seed if seed != "sum" else "Sum of three seeds")
    save(fig, "confusion_matrix_seeds")

    transfer = json.loads((args.tables_dir / "mechanism_transfer.json").read_text())
    order = ["bilstm", "bilstm_ce", "bilstm_nocw", "reslstm", "normc_lstmlike",
             "normc_torchinit", "normc_nofb", "normc", "normc_nodrop", "full", "full_nodrop"]
    fig, ax = plt.subplots(figsize=(111.13 / 25.4, 108 / 25.4))
    fig.subplots_adjust(left=.46, right=.98, bottom=.15, top=.82)
    for part, name, marker, color, offset in [
        ("train", "DS1 train", "o", ".15", -.2),
        ("val", "DS1 validation", "s", "tab:orange", 0),
        ("test", "DS2 test records", "^", "tab:blue", .2)]:
        for i, core in enumerate(order):
            ax.scatter([r[part]["auc_F_vs_N"] for r in transfer[core]],
                       [i + offset] * len(transfer[core]), s=14, marker=marker,
                       color=color, label=name if i == 0 else None)
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order]); ax.invert_yaxis()
    ax.axvline(.5, color=".7", lw=.6, ls="--"); ax.set_xlim(.45, 1.01)
    ax.set_xlabel("F-versus-N AUC\n(F-vs-rest logit margin)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8, frameon=False)
    save(fig, "transfer_auc")

    ds2 = json.loads((args.tables_dir / "ds2_validation_blindness.json").read_text())
    rows = list(csv.DictReader((args.tables_dir / "sgcv_arch_runs.csv").open()))
    scores = {(r["core"], int(r["fold"]), int(r["seed"])):
              np.array([float(r["inner_val_macro_f1"]), float(r["outer_test_macro_f1"])]) for r in rows}
    fig, (a,b) = plt.subplots(2, 1, figsize=(130.74 / 25.4, 150 / 25.4))
    fig.subplots_adjust(left=.14, right=.97, bottom=.085, top=.94, hspace=.55)
    configs = ds2["configs_by_val"]
    a.scatter([r["val"] for r in configs], [r["test"] for r in configs], c=".35", s=20)
    # Avoid unreadable overlapping labels; full config scores remain in source data.
    a.set_xlabel("DS1 validation macro F1 (three-seed mean)")
    a.set_ylabel("DS2 test macro F1")
    a.set_title(f"(a) DS1/DS2: 20 configurations, Spearman {ds2['spearman_configs']:.2f}")
    for i, core in enumerate(CORES):
        points=[]
        for f in range(5):
            for s in (42,123):
                center=np.mean([scores[c,f,s] for c in CORES],axis=0)
                points.append(scores[core,f,s]-center)
        points=np.asarray(points)
        b.scatter(points[:,0],points[:,1],s=17,label=LABEL[core],marker="os^vDP"[i])
    b.axhline(0,color=".75",lw=.6,zorder=0);b.axvline(0,color=".75",lw=.6,zorder=0)
    b.set_xlabel("Inner-validation macro F1 minus cell mean")
    b.set_ylabel("Outer-test macro F1 minus cell mean")
    b.set_title("(b) SG-CV: 6 cores, mean within-cell Spearman 0.08")
    b.legend(loc="lower right",fontsize=8,frameon=True,facecolor="white",framealpha=1)
    save(fig, "validation_agreement")


if __name__ == "__main__":
    main()
