"""Does the F-vs-N ranking learned on the training subjects transfer to unseen subjects, and
can the validation records see the difference? Re-scores the saved best checkpoints (no
retraining) on DS1-train, DS1-validation and DS2, and reports per arch and seed:
  * AUC of the F-vs-rest logit margin, true F against N-class beats, on each partition;
  * number of N-class beats predicted F and F recall on each partition;
  * the per-class F1 contributions to the DS2 macro-F1 gap between two cores.

    python -m validation.mechanism_transfer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from validation.mechanism_lbbb_train import RUNS, SEEDS, SPLITS, logits_of

OUT = Path(__file__).resolve().parents[1] / "results" / "tables" / "mechanism_transfer.json"
MODELS = ["bilstm", "bilstm_ce", "bilstm_nocw", "reslstm", "normc_lstmlike", "normc_torchinit",
          "normc_nofb", "normc", "normc_nodrop", "full", "full_nodrop"]
GAPS = [("normc", "bilstm"), ("full", "bilstm"), ("normc", "normc_lstmlike"),
        ("bilstm_ce", "bilstm"), ("full_nodrop", "normc_nodrop")]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {p: np.load(SPLITS / f"inter_{p}.npz") for p in ("train", "val", "test")}
    X = {p: torch.from_numpy(d["X"]).float() for p, d in data.items()}
    res, f1s = {}, {}
    for a in MODELS:
        rows = []
        for s in SEEDS:
            row = {"seed": s}
            for p in ("train", "val", "test"):
                y = data[p]["y"]
                if p == "test":
                    z = np.load(RUNS / f"{a}_seed{s}_test_logits.npz")["logits"].astype(float)
                else:
                    z = logits_of(a, s, X[p], device).astype(float)
                pred = z.argmax(1)
                margin = z[:, 3] - np.delete(z, 3, axis=1).max(1)
                m = (y == 0) | (y == 3)
                row[p] = {"auc_F_vs_N": float(roc_auc_score(y[m] == 3, margin[m])),
                          "N_to_F": int(((pred == 3) & (y == 0)).sum()),
                          "n_N": int((y == 0).sum()),
                          "F_recall": float((pred[y == 3] == 3).mean())}
                if p == "test":
                    f1s[(a, s)] = f1_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
            rows.append(row)
        res[a] = rows
        fmt = lambda p, k, f: " ".join(f.format(r[p][k]) for r in rows)
        print(f"{a:16s} AUC F/N train {fmt('train','auc_F_vs_N','{:.3f}')} | val {fmt('val','auc_F_vs_N','{:.3f}')}"
              f" | test {fmt('test','auc_F_vs_N','{:.3f}')} || N->F val {fmt('val','N_to_F','{:4d}')}"
              f" test {fmt('test','N_to_F','{:5d}')}")
    gaps = {}
    for a, b in GAPS:
        d = np.mean([f1s[(a, s)] - f1s[(b, s)] for s in SEEDS], axis=0)
        gaps[f"{a}-{b}"] = {"per_class_F1_diff_NSVF": d.round(4).tolist(),
                            "share_of_macro_gap_NSVF": (d / d.sum()).round(3).tolist(),
                            "macro_gap": float(d.mean())}
        print(f"{a}-{b}: dF1 N/S/V/F {np.round(d, 3)}  macro {d.mean():+.3f}  share {np.round(d / d.sum(), 2)}")
    res["gap_decomposition"] = gaps
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
