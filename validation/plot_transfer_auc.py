"""Figure: AUC of the F-vs-rest logit margin between true F and N-class beats on the DS1
training records, the DS1 validation records and DS2, for every saved checkpoint of the
eleven re-scored configurations (results/tables/mechanism_transfer.json; no retraining).

    python -m validation.plot_transfer_auc
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "mechanism_transfer.json"
FIG = ROOT / "latex" / "images" / "transfer_auc.png"
LABEL = {"bilstm": "nn.LSTM", "bilstm_ce": "nn.LSTM, cross-entropy",
         "bilstm_nocw": "nn.LSTM, no class weights", "reslstm": "Residual BiLSTM",
         "normc_lstmlike": "No-RMC, LSTM-like", "normc_torchinit": "No-RMC, nn.LSTM init.",
         "normc_nofb": "No-RMC, forget bias 0", "normc": "No-RMC",
         "normc_nodrop": "No-RMC, no in-cell dropout", "full": "ResMemECG",
         "full_nodrop": "ResMemECG, no in-cell dropout"}
STYLE = {"train": ("DS1 training records", "o", "0.15"), "val": ("DS1 validation records", "s", "tab:orange"),
         "test": ("DS2 (unseen subjects)", "^", "tab:blue")}


def main():
    res = json.load(open(SRC))
    archs = [a for a in LABEL if a in res]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for part, (name, marker, color) in STYLE.items():
        for i, a in enumerate(archs):
            vals = [r[part]["auc_F_vs_N"] for r in res[a]]
            off = {"train": -0.2, "val": 0.0, "test": 0.2}[part]
            ax.scatter(vals, [i + off] * len(vals), marker=marker, color=color, s=22,
                       label=name if i == 0 else None, alpha=0.9)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([LABEL[a] for a in archs], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.5, color="0.8", lw=0.8, ls="--")
    ax.set_xlabel("AUC, true F versus N-class beats (F-vs-rest logit margin)")
    ax.set_xlim(0.45, 1.01)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    print("saved", FIG)


if __name__ == "__main__":
    main()
