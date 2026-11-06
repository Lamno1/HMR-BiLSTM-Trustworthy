"""DS1/DS2 split files with the MLII lead selected by name (sensitivity check for record 114).

validation/preprocess_aami.py takes signal channel 0 of every record. In MIT-BIH record 114
channel 0 is V5 and MLII is channel 1, so the stored files (data/processed/splits) contain V5
for that record. This script repeats the same extraction (125 Hz, 187-sample window, AAMI
mapping, train-only normalisation) but takes the channel named MLII, and writes the result to
data/processed/splits_mlii/ without touching the original files. It checks that every record
other than 114 yields exactly the same raw beats as the channel-0 extraction.

    python -m validation.preprocess_aami_mlii
    python train_matched_recipe.py --arch full --data-dir data/processed/splits_mlii \
        --out-dir results/matched_recipe_mlii
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.signal as signal
import wfdb

from validation.preprocess_aami import AAMI_MAPPING, DS1, DS2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mitdb"
OUT = ROOT / "data" / "processed" / "splits_mlii"
VAL = ["122", "124", "205", "223"]


def extract(records, by_name: bool):
    """Same steps as preprocess_aami.extract_beats; channel 0 or the channel named MLII."""
    beats, recs, ys, used = [], [], [], {}
    for r in records:
        rec = wfdb.rdrecord(str(RAW / r))
        ann = wfdb.rdann(str(RAW / r), "atr")
        ch = rec.sig_name.index("MLII") if by_name else 0
        used[r] = rec.sig_name[ch]
        sig = signal.resample_poly(rec.p_signal[:, ch], 125, int(round(rec.fs)))
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym not in AAMI_MAPPING:
                continue
            peak = int(samp * 125 / rec.fs)
            if peak - 90 >= 0 and peak + 97 <= len(sig):
                beats.append(sig[peak - 90:peak + 97])
                recs.append(r)
                ys.append(AAMI_MAPPING[sym])
    return np.array(beats, dtype=np.float32), np.array(recs), np.array(ys, dtype=np.int64), used


def main():
    X0, rec0, y0, _ = extract(DS1 + DS2, by_name=False)
    X, rec, y, used = extract(DS1 + DS2, by_name=True)
    changed = sorted(r for r, n in used.items() if n != "MLII")
    assert not changed, f"no MLII channel in {changed}"
    assert np.array_equal(rec, rec0) and np.array_equal(y, y0), "beat order or labels differ"
    other = rec != "114"
    assert np.array_equal(X[other], X0[other]), "records other than 114 changed"
    n114 = int((~other).sum())

    parts = {"train": [r for r in DS1 if r not in VAL], "val": VAL, "test": DS2}
    idx = {k: np.where(np.isin(rec, v))[0] for k, v in parts.items()}
    mean, std = X[idx["train"]].mean(), X[idx["train"]].std() + 1e-8
    OUT.mkdir(parents=True, exist_ok=True)
    for k, ii in idx.items():
        np.savez(OUT / f"inter_{k}.npz", X=((X[ii] - mean) / std).reshape(-1, 187, 1), y=y[ii])
        np.save(OUT / f"inter_{k}_records.npy", rec[ii])
    np.save(OUT / "inter_norm_mean.npy", np.array([mean], dtype=np.float32))
    np.save(OUT / "inter_norm_std.npy", np.array([std], dtype=np.float32))
    info = {"record_114_beats": n114, "train_mean": float(mean), "train_std": float(std),
            "counts": {k: np.bincount(y[ii], minlength=5).tolist() for k, ii in idx.items()}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
