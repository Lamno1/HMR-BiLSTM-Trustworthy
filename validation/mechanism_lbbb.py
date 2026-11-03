"""Why does class-weighted focal loss push LBBB beats into F for nn.LSTM but not for the ResMemECG cell?

Two candidate factors, separated on the saved DS2 test logits (no retraining):
  1. separability: can the model's F score tell true F beats from L beats at all?
     (AUC of the F-vs-rest logit margin, true F against L beats; threshold-free, so
     unaffected by a constant prior shift);
  2. prior shift: class weights act roughly like adding log w_k to the logits. If L beats
     land in F only because of that shift, subtracting a constant from the F logit should
     return them to N without losing the true F beats that the model can rank.
For every arch/seed we report the AUC and the L->F count and F recall after subtracting a
constant delta from the F logit, for a grid of deltas.

    python -m validation.mechanism_lbbb
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from validation.matched_diagnostics import RUNS, SEEDS, SPLITS, ds2_symbols

OUT = Path(__file__).resolve().parents[1] / "results" / "tables" / "mechanism_lbbb.json"
ARCHS = ["bilstm", "bilstm_ce", "bilstm_nocw", "normc", "full", "normc_nodrop", "full_nodrop",
         "normc_lstmlike"]
DELTAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]


def main():
    y = np.load(SPLITS / "inter_test.npz")["y"]
    sym = ds2_symbols()
    isL, isF = sym == "L", y == 3
    res = {}
    for a in ARCHS:
        rows = []
        for s in SEEDS:
            z = np.load(RUNS / f"{a}_seed{s}_test_logits.npz")["logits"].astype(float)
            margin = z[:, 3] - np.delete(z, 3, axis=1).max(1)   # F logit minus best other
            m = isF | isL
            auc = roc_auc_score(isF[m], margin[m])
            shift = {}
            for d in DELTAS:
                zz = z.copy(); zz[:, 3] -= d
                p = zz.argmax(1)
                shift[d] = {"L_to_F": int(((p == 3) & isL).sum()),
                            "F_recall": float((p[isF] == 3).mean()),
                            "n_pred_F": int((p == 3).sum())}
            rows.append({"seed": s, "auc_F_vs_L": float(auc),
                         "median_margin_L": float(np.median(margin[isL])),
                         "median_margin_F": float(np.median(margin[isF])),
                         "shift": shift})
        res[a] = rows
        print(f"{a:15s} AUC(F vs L) " + " ".join(f"{r['auc_F_vs_L']:.3f}" for r in rows)
              + " | med margin L " + " ".join(f"{r['median_margin_L']:+.2f}" for r in rows)
              + " | F " + " ".join(f"{r['median_margin_F']:+.2f}" for r in rows))
        for d in DELTAS:
            print(f"{'':15s} delta {d:3.1f}: L->F " + " ".join(f"{r['shift'][d]['L_to_F']:5d}" for r in rows)
                  + "  F recall " + " ".join(f"{r['shift'][d]['F_recall']:.3f}" for r in rows))
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
