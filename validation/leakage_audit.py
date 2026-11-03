"""Leakage audit for the DS1/DS2 split files and the controlled SG-CV protocol.

The stored DS1/DS2 split files (data/processed/splits/inter_*.npz) hold only X and y, so this
script re-extracts every beat from the raw WFDB records with the procedure of
validation/preprocess_aami.py, keeping the record identifier, and checks that

  * the re-extracted, train-normalised beats reproduce the stored files exactly (same order),
    so the record membership of every stored beat is known;
  * training, validation and test records are disjoint, and which records share a subject;
  * no raw beat appears in more than one partition;
  * the stored normalisation statistics are those of the training records only;
  * which signal channel 0 actually is in each record.

For SG-CV it checks the saved protocol manifest: subject groups disjoint within every fold,
each group tested exactly once, and records 201/202 bound to one group.

    python -m validation.leakage_audit
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.signal as signal
import wfdb

from validation.preprocess_aami import AAMI_MAPPING, DS1, DS2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mitdb"
SPLITS = ROOT / "data" / "processed" / "splits"
SGCV = ROOT / "results" / "controlled_mit_subject_cv" / "protocol_manifest.json"
OUT = ROOT / "results" / "tables" / "leakage_audit.json"
VAL = ["122", "124", "205", "223"]
SAME_SUBJECT = [("201", "202")]  # MIT-BIH documentation: records 201 and 202 come from one subject


def extract(records):
    """Same steps as validation/preprocess_aami.extract_beats, with record ids kept."""
    beats, recs, ys, ch0 = [], [], [], {}
    for r in records:
        rec = wfdb.rdrecord(str(RAW / r))
        ann = wfdb.rdann(str(RAW / r), "atr")
        ch0[r] = rec.sig_name[0]
        sig = signal.resample_poly(rec.p_signal[:, 0], 125, int(round(rec.fs)))
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym not in AAMI_MAPPING:
                continue
            peak = int(samp * 125 / rec.fs)
            if peak - 90 >= 0 and peak + 97 <= len(sig):
                beats.append(sig[peak - 90:peak + 97])
                recs.append(r)
                ys.append(AAMI_MAPPING[sym])
    return np.array(beats, dtype=np.float32), np.array(recs), np.array(ys, dtype=np.int64), ch0


def audit_ds1ds2():
    X, rec, y, ch0 = extract(DS1 + DS2)
    parts = {"train": [r for r in DS1 if r not in VAL], "val": VAL, "test": DS2}
    idx = {k: np.where(np.isin(rec, v))[0] for k, v in parts.items()}
    mean, std = X[idx["train"]].mean(), X[idx["train"]].std() + 1e-8

    out = {"reproduces_stored_files": {}}
    for k, ii in idx.items():
        # Record id of every stored beat, in stored order (valid only if the check below passes).
        np.save(SPLITS / f"inter_{k}_records.npy", rec[ii])
        d = np.load(SPLITS / f"inter_{k}.npz")
        Xs, ys = d["X"].reshape(len(d["y"]), -1), d["y"]
        same_shape = Xs.shape == X[ii].shape
        out["reproduces_stored_files"][k] = {
            "n_stored": int(len(ys)), "n_reextracted": int(len(ii)),
            "labels_identical": bool(same_shape and np.array_equal(ys, y[ii])),
            "max_abs_diff_X": float(np.abs(Xs - (X[ii] - mean) / std).max()) if same_shape else None,
            "records": sorted(set(rec[ii].tolist())),
        }

    names = list(parts)
    out["record_overlap"] = {f"{a}&{b}": sorted(set(parts[a]) & set(parts[b]))
                             for i, a in enumerate(names) for b in names[i + 1:]}
    where = {r: k for k, v in parts.items() for r in v}
    out["same_subject_across_partitions"] = [
        {"records": [a, b], "partitions": [where.get(a), where.get(b)]}
        for a, b in SAME_SUBJECT if where.get(a) != where.get(b)]

    h = np.array([hashlib.sha1(b.tobytes()).hexdigest() for b in X])
    dup = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dup[f"{a}&{b}"] = int(len(set(h[idx[a]]) & set(h[idx[b]])))
    out["identical_raw_beats_across_partitions"] = dup

    stored_mean = float(np.load(SPLITS / "inter_norm_mean.npy")[0])
    stored_std = float(np.load(SPLITS / "inter_norm_std.npy")[0])
    out["normalisation"] = {
        "stored_mean": stored_mean, "stored_std": stored_std,
        "train_only_mean": float(mean), "train_only_std": float(std),
        "all_records_mean": float(X.mean()), "all_records_std": float(X.std()),
        "stored_equals_train_only": bool(np.isclose(stored_mean, mean, atol=1e-6)
                                         and np.isclose(stored_std, std, atol=1e-6)),
    }
    out["channel0_not_MLII"] = {r: n for r, n in ch0.items() if n != "MLII"}
    return out


def audit_sgcv():
    m = json.loads(SGCV.read_text())
    folds, n_groups = m["folds"], int(m["n_subject_groups"])
    rows, tested = [], []
    for f in folds:
        tr, va, te = set(f["train_groups"]), set(f["val_groups"]), set(f["test_groups"])
        rows.append({"fold": f["fold"], "train&val": len(tr & va), "train&test": len(tr & te),
                     "val&test": len(va & te), "covers_all_groups": len(tr | va | te) == n_groups})
        tested += f["test_groups"]
    rec_group = {r: g for g, rs in m["group_records"].items() for r in rs}
    return {
        "n_records": int(m["n_mit_records"]), "n_subject_groups": n_groups, "folds": rows,
        "each_group_tested_once": sorted(tested) == list(range(n_groups)),
        "201_202_same_group": rec_group.get("mitdb/201") == rec_group.get("mitdb/202"),
    }


def main():
    res = {"ds1_ds2": audit_ds1ds2(), "sgcv": audit_sgcv()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
