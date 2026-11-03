"""Figure: validation versus test macro-F1 for (a) the twenty DS1/DS2 matched-harness
configurations (three-seed means) and (b) the six cores under controlled SG-CV, one point per
fold-seed run, centred on the mean of the six cores in the same fold-seed cell. Also prints the per-core
transfer summary used in the BEL table (three-seed means of the AUC of the F-vs-rest margin).

    python -m validation.plot_validation_agreement
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.sgcv_validation_blindness import ARCHS as SG_ARCHS, load as load_sgcv

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "latex" / "images" / "validation_agreement.png"
LABEL = {"full": "ResMemECG", "normc": "No-RMC", "normc_nodrop": "No-RMC, no in-cell dropout",
         "bilstm": "nn.LSTM", "bilstm_nocw": "nn.LSTM, no class weights",
         "bilstm_ce": "nn.LSTM, cross-entropy", "full_nocw": "ResMemECG, no class weights",
         "normc_lstmlike": "No-RMC, LSTM-like"}
CORES = ["bilstm", "bilstm_ce", "bilstm_nocw", "normc_lstmlike", "normc", "normc_nodrop",
         "full", "full_nodrop"]


def transfer_table():
    tr = json.load(open(TABLES / "mechanism_transfer.json"))
    fl = json.load(open(TABLES / "mechanism_lbbb.json"))
    fl_tr = json.load(open(TABLES / "mechanism_lbbb_train.json"))["train"]
    for a in CORES:
        m = lambda p: np.mean([r[p]["auc_F_vs_N"] for r in tr[a]])
        test_fl = np.mean([r["auc_F_vs_L"] for r in fl[a]])
        train_fl = np.mean([r["auc_F_vs_L"] for r in fl_tr[a]]) if a in fl_tr else float("nan")
        print(f"{a:15s} F/N train {m('train'):.3f} val {m('val'):.3f} test {m('test'):.3f} | "
              f"F/L train {train_fl:.3f} test {test_fl:.3f}")


def main():
    transfer_table()
    ds2 = json.load(open(TABLES / "ds2_validation_blindness.json"))
    sg = load_sgcv()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    rows = ds2["configs_by_val"]
    a.scatter([r["val"] for r in rows], [r["test"] for r in rows], c="0.4", s=22)
    for r in rows:
        if r["arch"] in ("bilstm", "normc", "full", "normc_nodrop", "normc_lstmlike", "full_nocw"):
            a.annotate(LABEL.get(r["arch"], r["arch"]), (r["val"], r["test"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("DS1 validation macro F1 (three-seed mean)")
    a.set_ylabel("DS2 test macro F1 (three-seed mean)")
    a.set_title(f"(a) DS1/DS2, 20 configurations\nSpearman {ds2['spearman_configs']:.2f}",
                fontsize=9)
    cells = sorted({(k[1], k[2]) for k in sg})
    for i, c in enumerate(SG_ARCHS):
        pts = []
        for f, s in cells:
            cell = np.array([sg[(a2, f, s)] for a2 in SG_ARCHS if (a2, f, s) in sg])
            if (c, f, s) in sg:
                pts.append(np.array(sg[(c, f, s)]) - cell.mean(0))
        pts = np.array(pts)
        b.scatter(pts[:, 0], pts[:, 1], s=18, label=LABEL[c], marker="os^vDP"[i])
    b.axhline(0, color="0.8", lw=0.8, zorder=0); b.axvline(0, color="0.8", lw=0.8, zorder=0)
    b.set_xlabel("Inner-validation macro F1 minus fold-seed mean")
    b.set_ylabel("Outer-test macro F1 minus fold-seed mean")
    sj = json.load(open(TABLES / "sgcv_validation_blindness.json"))
    b.set_title(f"(b) SG-CV, 6 cores, centred within fold-seed\nmean within-cell Spearman "
                f"{sj['within_fold_seed']['mean_spearman']:.2f}", fontsize=9)
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    print("saved", FIG)


if __name__ == "__main__":
    main()
