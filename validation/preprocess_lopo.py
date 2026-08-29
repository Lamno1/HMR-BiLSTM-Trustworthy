"""
preprocess_lopo.py  —  pooled RECORD-level 2-lead beat set for leave-one-patient
-out / grouped-K-fold CV of the rare classes (F, S).  See docs/LOPO_PROTOCOL.md.

v2 (2026-08-30): now carries a SECOND lead for the HMR-MultiPath F-specialist.
  - MIT-BIH: primary = MLII (found by name, not index — record 114 has MLII at
    index 1), second = the other lead (V1 for 38/44 records, else V2/V4/V5).
  - INCART: primary = lead II (0.5 Hz high-pass, R-peak re-center, polarity fix,
    mean-N morphology filter), second = lead V1 (same window/offset, own high-pass
    + polarity fix).

Beat geometry identical to validation/preprocess_aami_rr.py (187 samples, −90/+97
around the R-peak at 125 Hz). NO normalization here — fit per fold in
train_lopo_cv.py.

Output:
  data/processed/lopo/pool.npz
      X (N,187,1)  primary lead        y (N)      rr (N,3)
      X2 (N,187,1) second lead         record (N) source (N)  lead2 (N)  <- lead2 name id
  data/processed/lopo/records.json

Run:
    python validation/preprocess_lopo.py [--min-corr 0.4] [--incart-all]
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.signal as sig_sp
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_aami_rr_crossdb import (      # noqa: E402
    DS1, DS2, AAMI, CLASS_NAMES, rr_features_for_record, highpass_zero_phase,
    mitbih_meanN_template,
)

LEAD2_NAMES = ["V1", "V2", "V3", "V4", "V5", "V6", "MLII", "II", "other"]
_L2 = {n: i for i, n in enumerate(LEAD2_NAMES)}


def _l2id(name):
    n = name.strip().upper()
    for k in LEAD2_NAMES:
        if k.upper() == n:
            return _L2[k]
    return _L2["other"]


# ------------------------------------------------------------------ MIT-BIH (2-lead)
def extract_mitbih_2lead(records):
    raw = Path("data/raw/mitdb")
    out = []
    for r in records:
        rec = wfdb.rdrecord(str(raw / r))
        ann = wfdb.rdann(str(raw / r), "atr")
        names = [s.strip().upper() for s in rec.sig_name]
        p_idx = names.index("MLII") if "MLII" in names else 0
        l2_idx = 1 - p_idx if rec.p_signal.shape[1] >= 2 else p_idx
        fs = rec.fs
        s1 = sig_sp.resample_poly(rec.p_signal[:, p_idx], 125, int(round(fs)))
        s2 = sig_sp.resample_poly(rec.p_signal[:, l2_idx], 125, int(round(fs)))
        l2name = rec.sig_name[l2_idx]

        samples, labels = [], []
        for i in range(len(ann.sample)):
            if ann.symbol[i] in AAMI:
                samples.append(ann.sample[i]); labels.append(AAMI[ann.symbol[i]])
        rr = rr_features_for_record(samples, fs)
        for k, samp in enumerate(samples):
            p = int(samp * 125 / fs)
            a, b = p - 90, p + 97
            if a >= 0 and b <= len(s1):
                out.append({"rec": f"mitdb/{r}", "src": 0,
                            "x": s1[a:b].astype(np.float32),
                            "x2": s2[a:b].astype(np.float32),
                            "l2": _l2id(l2name), "y": labels[k], "rr": rr[k]})
    return out


# ------------------------------------------------------------------ INCART (2-lead)
def _lead_index(sig_name, want):
    for i, n in enumerate(sig_name):
        if n.strip().upper() == want.upper():
            return i
    return None


def extract_incart_2lead(min_corr, template):
    raw = Path("data/raw/incartdb")
    out, dropped, corrs = [], [], {}
    for h in sorted(raw.glob("*.hea")):
        recname = h.stem
        hdr = wfdb.rdheader(str(raw / recname))
        li = _lead_index(hdr.sig_name, "II")
        v1 = _lead_index(hdr.sig_name, "V1")
        if li is None:
            dropped.append((recname, "no lead II", None)); continue
        rec = wfdb.rdrecord(str(raw / recname))
        fs = rec.fs
        ann = wfdb.rdann(str(raw / recname), "atr")

        s1 = sig_sp.resample_poly(highpass_zero_phase(rec.p_signal[:, li].astype(float), fs),
                                  125, int(round(fs)))
        s2 = (sig_sp.resample_poly(highpass_zero_phase(rec.p_signal[:, v1].astype(float), fs),
                                   125, int(round(fs))) if v1 is not None else None)

        samples_raw, labels, peaks = [], [], []
        for i in range(len(ann.sample)):
            sym = ann.symbol[i]
            if sym not in AAMI:
                continue
            p0 = int(ann.sample[i] * 125 / fs)
            lo, hi = max(0, p0 - 15), min(len(s1), p0 + 15)
            if hi - lo < 10:
                continue
            p = lo + int(np.argmax(np.abs(s1[lo:hi] - np.median(s1[lo:hi]))))
            samples_raw.append(ann.sample[i]); labels.append(AAMI[sym]); peaks.append(p)
        if not peaks:
            dropped.append((recname, "no AAMI beats", None)); continue

        devs = [s1[p] - np.median(s1[max(0, p - 60):p + 60]) for p in peaks[:500]]
        if np.median(devs) < 0:
            s1 = -s1
        if s2 is not None:
            d2 = [s2[p] - np.median(s2[max(0, p - 60):p + 60]) for p in peaks[:500]]
            if np.median(d2) < 0:
                s2 = -s2

        nb = [(s1[p - 90:p + 97] - s1[p - 90:p + 97].mean()) / (s1[p - 90:p + 97].std() + 1e-8)
              for p, lab in zip(peaks, labels) if lab == 0 and p - 90 >= 0 and p + 97 <= len(s1)]
        corr = float(np.corrcoef(template, np.mean(nb, axis=0))[0, 1]) if len(nb) >= 20 else np.nan
        corrs[recname] = corr
        if np.isnan(corr) or corr < min_corr:
            nf = sum(1 for l in labels if l == 3)
            dropped.append((recname, f"meanN corr {corr:.2f} < {min_corr}", nf)); continue

        rr = rr_features_for_record(samples_raw, fs)
        for k, p in enumerate(peaks):
            a, b = p - 90, p + 97
            if a >= 0 and b <= len(s1):
                x2 = (s2[a:b] if s2 is not None and b <= len(s2)
                      else np.zeros(187, dtype=np.float32))
                out.append({"rec": f"incart/{recname}", "src": 1,
                            "x": s1[a:b].astype(np.float32), "x2": np.asarray(x2, np.float32),
                            "l2": _l2id("V1") if s2 is not None else _l2["other"],
                            "y": labels[k], "rr": rr[k]})
    return out, dropped, corrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-corr", type=float, default=0.4)
    ap.add_argument("--incart-all", action="store_true")
    args = ap.parse_args()

    out_dir = Path("data/processed/lopo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== MIT-BIH (44 records, MLII + 2nd lead) ===")
    mit = extract_mitbih_2lead(DS1 + DS2)
    print(f"  {len(mit)} beats")

    print("=== INCART (lead II + V1, 0.5 Hz high-pass) ===")
    template = mitbih_meanN_template()
    inc, dropped, corrs = extract_incart_2lead(args.min_corr, template)
    print(f"  kept {len(inc)} / {len(set(b['rec'] for b in inc))} records; dropped {len(dropped)}")

    if not args.incart_all:
        by_rec = defaultdict(lambda: np.zeros(5, int))
        for b in inc:
            by_rec[b["rec"]][b["y"]] += 1
        keep = {r for r, c in by_rec.items() if c[1] > 0 or c[3] > 0}
        inc = [b for b in inc if b["rec"] in keep]
        print(f"  F/S-bearing only: {len(inc)} beats / {len(keep)} records")

    beats = mit + inc
    rec_names = sorted(set(b["rec"] for b in beats))
    rid = {n: i for i, n in enumerate(rec_names)}

    X = np.stack([b["x"] for b in beats]).reshape(-1, 187, 1).astype(np.float32)
    X2 = np.stack([b["x2"] for b in beats]).reshape(-1, 187, 1).astype(np.float32)
    y = np.array([b["y"] for b in beats], np.int64)
    rr = np.array([b["rr"] for b in beats], np.float32)
    record = np.array([rid[b["rec"]] for b in beats], np.int64)
    source = np.array([b["src"] for b in beats], np.int64)
    lead2 = np.array([b["l2"] for b in beats], np.int64)

    np.savez(out_dir / "pool.npz", X=X, X2=X2, y=y, rr=rr,
             record=record, source=source, lead2=lead2)

    meta = {}
    for n, i in rid.items():
        m = record == i
        cc = np.bincount(y[m], minlength=5).tolist()
        meta[str(i)] = {"name": n, "source": int(source[m][0]),
                        "lead2": LEAD2_NAMES[int(lead2[m][0])],
                        "n_beats": int(m.sum()),
                        "class_counts": {CLASS_NAMES[k]: cc[k] for k in range(5)}}
    json.dump({"n_records": len(rid), "n_beats": len(y), "records": meta,
               "incart_dropped": [{"rec": r, "why": w} for r, w, _ in dropped]},
              open(out_dir / "records.json", "w"), indent=2)

    def line(mask, tag):
        cc = np.bincount(y[mask], minlength=5)
        print(f"  {tag:10s} beats={mask.sum():6d} records={len(set(record[mask].tolist())):3d} "
              f"N={cc[0]:6d} S={cc[1]:5d} V={cc[2]:5d} F={cc[3]:4d} Q={cc[4]:3d} "
              f"| F-rec={len(set(record[mask&(y==3)].tolist()))} "
              f"S-rec={len(set(record[mask&(y==1)].tolist()))}")
    print("\n================  LOPO POOL v2 (2-lead)  ================")
    line(source == 0, "MIT-BIH"); line(source == 1, "INCART")
    line(np.ones(len(y), bool), "TOTAL")
    z2 = int((np.abs(X2).sum(axis=(1, 2)) == 0).sum())
    print(f"  2nd-lead zero-filled beats: {z2}  ({100*z2/len(y):.1f}%)")
    print(f"\n[OK] {out_dir}/pool.npz + records.json")


if __name__ == "__main__":
    main()
