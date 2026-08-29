"""
preprocess_lopo.py  —  pooled record-level beat set for leave-one-patient-out /
grouped-K-fold cross-validation of the rare classes (F, S).

See docs/RARE_CLASS_PROGRESS.md. The fixed DS1/DS2 split makes F1-F/F1-S a
single-patient -> single-patient estimate (test-F 93% patient 213, test-S 75%
patient 232) => irreducible seed variance. This builds a pool where the fold
unit is the RECORD, so F/S can be estimated as a distribution over held-out
patients with proper error bars.

Sources pooled:
  - MIT-BIH  : all 44 records, lead 0 (MLII), the official window/resample
               (identical to validation/preprocess_aami_rr.py).
  - INCART   : records that carry F and/or S, lead II, 0.5 Hz Butterworth
               high-pass + re-center + polarity fix + mean-N morphology filter
               (identical to validation/preprocess_aami_rr_crossdb.py v2).

NO normalization is applied here — it MUST be fit per-fold on train-fold records
only (done in train_lopo_cv.py). Output:

  data/processed/lopo/pool.npz      X(N,187,1) y(N) rr(N,3) record(N) source(N)
  data/processed/lopo/records.json  record_id -> {name, source, class_counts}

Run:
    python validation/preprocess_lopo.py [--min-corr 0.4] [--incart-all]
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_aami_rr_crossdb import (      # noqa: E402
    DS1, DS2, extract_mitbih, extract_incart, mitbih_meanN_template, CLASS_NAMES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-corr", type=float, default=0.4)
    ap.add_argument("--incart-all", action="store_true",
                    help="keep ALL INCART records, not just F/S-bearing ones")
    args = ap.parse_args()

    out_dir = Path("data/processed/lopo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== MIT-BIH (44 records, lead MLII) ===")
    mit = extract_mitbih(DS1 + DS2)
    print(f"  {len(mit)} beats")

    print("=== INCART (lead II, 0.5 Hz high-pass) ===")
    template = mitbih_meanN_template()
    inc, dropped, corrs = extract_incart(args.min_corr, template)
    print(f"  kept {len(inc)} beats / {len(set(b['rec'] for b in inc))} records; "
          f"dropped {len(dropped)}")

    if not args.incart_all:
        # keep only INCART records that carry F or S (the scarce classes)
        fs_recs = set()
        by_rec_cls = defaultdict(lambda: np.zeros(5, dtype=int))
        for b in inc:
            by_rec_cls[b['rec']][b['y']] += 1
        for r, c in by_rec_cls.items():
            if c[1] > 0 or c[3] > 0:          # S or F present
                fs_recs.add(r)
        inc = [b for b in inc if b['rec'] in fs_recs]
        print(f"  F/S-bearing only: {len(inc)} beats / {len(fs_recs)} records")

    beats = mit + inc

    # ---- record id map ----
    rec_names = sorted(set(b['rec'] for b in beats))
    rid = {name: i for i, name in enumerate(rec_names)}

    X = np.stack([b['x'] for b in beats]).reshape(-1, 187, 1).astype(np.float32)
    y = np.array([b['y'] for b in beats], dtype=np.int64)
    rr = np.array([b['rr'] for b in beats], dtype=np.float32)
    record = np.array([rid[b['rec']] for b in beats], dtype=np.int64)
    source = np.array([b['src'] for b in beats], dtype=np.int64)

    np.savez(out_dir / "pool.npz", X=X, y=y, rr=rr, record=record, source=source)

    records_meta = {}
    for name, i in rid.items():
        m = record == i
        cc = np.bincount(y[m], minlength=5).tolist()
        records_meta[str(i)] = {
            "name": name, "source": int(source[m][0]),
            "n_beats": int(m.sum()),
            "class_counts": {CLASS_NAMES[k]: cc[k] for k in range(5)},
        }
    with open(out_dir / "records.json", "w") as f:
        json.dump({"n_records": len(rid), "n_beats": len(y),
                   "records": records_meta,
                   "incart_dropped": [{"rec": r, "why": w} for r, w, _ in dropped]}, f, indent=2)

    # ---- report ----
    def line(mask, tag):
        cc = np.bincount(y[mask], minlength=5)
        nrec = len(set(record[mask].tolist()))
        frec = len(set(record[mask & (y == 3)].tolist()))
        srec = len(set(record[mask & (y == 1)].tolist()))
        print(f"  {tag:16s} beats={mask.sum():6d}  records={nrec:3d}  "
              f"N={cc[0]:6d} S={cc[1]:5d} V={cc[2]:5d} F={cc[3]:4d} Q={cc[4]:3d}  "
              f"| F-records={frec}  S-records={srec}")

    print("\n================  LOPO POOL  ================")
    line(source == 0, "MIT-BIH")
    line(source == 1, "INCART")
    line(np.ones(len(y), bool), "TOTAL")
    print(f"\n[OK] {out_dir}/pool.npz  +  records.json")
    print("  Next: python train_lopo_cv.py  (K-fold over records, per-fold normalization)")


if __name__ == "__main__":
    main()
