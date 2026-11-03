"""Is the LBBB->F failure of nn.LSTM learned on the training records or does it appear only on
unseen subjects? Re-scores the saved best checkpoints on the DS1 training and validation beats
(no retraining) and reports, per arch and seed, the number of L beats predicted F and the
AUC of the F-vs-rest logit margin between true F beats and L beats, overall and per record.

    python -m validation.mechanism_lbbb_train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import wfdb
from sklearn.metrics import roc_auc_score

from train_matched_recipe import ARCHS, BUILDERS
from validation.leakage_audit import VAL
from validation.preprocess_aami import AAMI_MAPPING, DS1

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mitdb"
RUNS = ROOT / "results" / "matched_recipe"
SPLITS = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "results" / "tables" / "mechanism_lbbb_train.json"
SEEDS = (42, 123, 456)
MODELS = ["bilstm", "bilstm_ce", "bilstm_nocw", "normc", "full", "normc_nodrop", "full_nodrop"]


def symbols(records):
    """Annotation symbol of every stored beat of `records`, in stored order."""
    out = []
    for r in records:
        rec = wfdb.rdrecord(str(RAW / r))
        ann = wfdb.rdann(str(RAW / r), "atr")
        n = len(signal.resample_poly(rec.p_signal[:, 0], 125, int(round(rec.fs))))
        for samp, sym in zip(ann.sample, ann.symbol):
            peak = int(samp * 125 / rec.fs)
            if sym in AAMI_MAPPING and peak - 90 >= 0 and peak + 97 <= n:
                out.append(sym)
    return np.array(out)


@torch.no_grad()
def logits_of(arch, seed, X, device):
    ck = torch.load(RUNS / f"{arch}_seed{seed}_model.pt", map_location="cpu", weights_only=False)
    model = BUILDERS[ARCHS[arch][0]](ck["config"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    out = [model(X[i:i + 1024].to(device)).float().cpu() for i in range(0, len(X), 1024)]
    return torch.cat(out).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for part, recs in (("train", [r for r in DS1 if r not in VAL]), ("val", VAL)):
        d = np.load(SPLITS / f"inter_{part}.npz")
        y, rec = d["y"], np.load(SPLITS / f"inter_{part}_records.npy")
        sym = symbols(recs)
        assert np.array_equal(np.array([AAMI_MAPPING[s] for s in sym]), y)
        keep = (sym == "L") | (y == 3)          # only L and F beats are needed
        X = torch.from_numpy(d["X"][keep]).float()
        isL, isF, rk = sym[keep] == "L", y[keep] == 3, rec[keep]
        res[part] = {"n_L": int(isL.sum()), "n_F": int(isF.sum()),
                     "L_by_record": {r: int((isL & (rk == r)).sum()) for r in sorted(set(rk[isL]))},
                     "F_by_record": {r: int((isF & (rk == r)).sum()) for r in sorted(set(rk[isF]))}}
        print(part, res[part])
        for a in MODELS:
            rows = []
            for s in SEEDS:
                z = logits_of(a, s, X, device)
                p = z.argmax(1)
                margin = z[:, 3] - np.delete(z, 3, axis=1).max(1)
                auc = roc_auc_score(isF, margin) if isF.any() and isL.any() else None
                rows.append({"seed": s, "L_to_F": int(((p == 3) & isL).sum()),
                             "F_recall": float((p[isF] == 3).mean()) if isF.any() else None,
                             "auc_F_vs_L": auc,
                             "L_to_F_by_record": {r: int(((p == 3) & isL & (rk == r)).sum())
                                                  for r in res[part]["L_by_record"]}})
            res[part][a] = rows
            print(f"  {a:13s} L->F " + " ".join(f"{r['L_to_F']:5d}" for r in rows)
                  + "  AUC " + " ".join("  n/a" if r["auc_F_vs_L"] is None else f"{r['auc_F_vs_L']:.3f}" for r in rows)
                  + "  Frec " + " ".join("n/a" if r["F_recall"] is None else f"{r['F_recall']:.2f}" for r in rows))
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
