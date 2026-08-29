"""
Cross-database preprocessing: MIT-BIH (DS1/DS2) + INCART, inter-patient AAMI, +RR ratio features.

v2 (2026-08-28) — fixes for the failed run 1 (see memory project_fclass_root_cause_diagnostics):
  * INCART now gets a 0.5 Hz Butterworth high-pass (order 2, zero-phase, sosfiltfilt)
    BEFORE resampling/windowing. INCART is NOT baseline-filtered like MIT-BIH; its
    raw baseline wander inflated the global std ~5x and squashed the MIT-BIH beats
    to std ~0.2, and let the R-peak re-center lock onto baseline excursions.
  * PER-SOURCE z-score normalization. MIT-BIH beats use a stat computed from
    MIT-BIH DS1-train ONLY  =>  the DS2 test set comes out numerically identical to
    the original data/processed/splits_rr/inter_test.npz (verified with np.allclose).
    INCART beats use a stat from INCART-train ONLY. No single global stat.
  * DS2 test set is RE-EXTRACTED from raw through the same code path (extract_mitbih),
    no more byte-identical copy that broke train/test normalization consistency.
  * Morphology-gap filter + R-peak re-center now run on the high-passed signal.

Output dir: data/processed/splits_rr_crossdb/
  inter_train.npz  = MIT-BIH DS1-train  +  INCART-train        (X, y, rr, source)
  inter_val.npz    = MIT-BIH DS1-val (122/124/205/223) + INCART-val
  inter_test.npz   = MIT-BIH DS2 (re-extracted, MIT-BIH-train normalization)
  incart_test.npz  = INCART held-out patients (secondary pure cross-database test)
  *_norm_*.npy     = per-source normalization stats
  split_manifest.json = which record -> which split, per-class counts, dropped records

source array: 0 = MIT-BIH, 1 = INCART   (for later per-dataset analysis / domain-adversarial)

Run:
    python validation/preprocess_aami_rr_crossdb.py [--min-corr 0.4] [--seed 42]
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.signal as signal
import wfdb

DS1 = ['101','106','108','109','112','114','115','116','118','119','122','124',
       '201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210','212',
       '213','214','219','221','222','228','231','232','233','234']
VAL_MITBIH = ['122', '124', '205', '223']

AAMI = {'N':0,'L':0,'R':0,'e':0,'j':0,
        'A':1,'a':1,'J':1,'S':1,
        'V':2,'E':2,
        'F':3,
        '/':4,'f':4,'Q':4}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
EPS = 1e-6
HP_CUTOFF_HZ = 0.5      # INCART baseline-wander removal
HP_ORDER = 2


# --------------------------------------------------------------------------------------
#  RR-ratio features  (identical formulation to validation/preprocess_aami_rr.py)
# --------------------------------------------------------------------------------------
def rr_features_for_record(beat_samples, fs):
    """Return list of [pre_rr_ratio, post_rr_ratio, local_rr_ratio] in beat order."""
    beat_samples = np.asarray(beat_samples)
    rr_all = np.diff(beat_samples) / fs
    med_rr = float(np.median(rr_all)) if len(rr_all) else 0.8
    feats = []
    for k in range(len(beat_samples)):
        pre_rr = float(rr_all[k - 1]) if k > 0 else med_rr
        post_rr = float(rr_all[k]) if k < len(rr_all) else med_rr
        local_before = float(np.mean(rr_all[max(0, k - 5):k])) if k > 0 else med_rr
        feats.append([
            pre_rr / max(local_before, EPS),
            post_rr / max(pre_rr, EPS),
            local_before / max(med_rr, EPS),
        ])
    return feats


def highpass_zero_phase(sig, fs, cutoff=HP_CUTOFF_HZ, order=HP_ORDER):
    """Zero-phase Butterworth high-pass to remove baseline wander (INCART only)."""
    sos = signal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, sig)


# --------------------------------------------------------------------------------------
#  MIT-BIH extraction  (mirrors validation/preprocess_aami_rr.py exactly — NO high-pass,
#  NO re-center, NO polarity fix: MIT-BIH is already band-pass filtered in the DB and
#  the DS2 test set must stay numerically comparable to the paper's.)
# --------------------------------------------------------------------------------------
def extract_mitbih(records):
    raw = Path("data/raw/mitdb")
    out = []
    for r in records:
        rec = wfdb.rdrecord(str(raw / r))
        ann = wfdb.rdann(str(raw / r), 'atr')
        sig = rec.p_signal[:, 0]
        fs = rec.fs
        sig125 = signal.resample_poly(sig, 125, int(round(fs)))

        samples, labels = [], []
        for i in range(len(ann.sample)):
            s = ann.symbol[i]
            if s in AAMI:
                samples.append(ann.sample[i])
                labels.append(AAMI[s])
        rr = rr_features_for_record(samples, fs)

        for k, samp in enumerate(samples):
            peak = int(samp * 125 / fs)
            a, b = peak - 90, peak + 97
            if a >= 0 and b <= len(sig125):
                out.append({'rec': f"mitdb/{r}", 'src': 0,
                            'x': sig125[a:b].astype(np.float32),
                            'y': labels[k], 'rr': rr[k]})
    return out


# --------------------------------------------------------------------------------------
#  MIT-BIH mean-N template  (reference for INCART morphology-gap filter)
# --------------------------------------------------------------------------------------
def mitbih_meanN_template():
    raw = Path("data/raw/mitdb")
    acc = []
    for r in ['101', '106', '109', '119', '200', '208', '213', '230']:
        rec = wfdb.rdrecord(str(raw / r))
        ann = wfdb.rdann(str(raw / r), 'atr')
        s = signal.resample_poly(rec.p_signal[:, 0], 125, int(round(rec.fs)))
        got = []
        for i, samp in enumerate(ann.sample):
            if ann.symbol[i] not in AAMI or AAMI[ann.symbol[i]] != 0:
                continue
            p = int(samp * 125 / rec.fs)
            if p - 90 < 0 or p + 97 > len(s):
                continue
            w = s[p - 90:p + 97]
            got.append((w - w.mean()) / (w.std() + 1e-8))
            if len(got) >= 300:
                break
        if got:
            acc.append(np.mean(got, axis=0))
    return np.mean(acc, axis=0)


# --------------------------------------------------------------------------------------
#  INCART extraction  (lead II, 0.5 Hz high-pass, re-center, per-record polarity fix)
# --------------------------------------------------------------------------------------
def _lead_index(sig_name, want='II'):
    for i, n in enumerate(sig_name):
        if n.strip().upper() == want:
            return i
    return None


def extract_incart(min_corr, template):
    raw = Path("data/raw/incartdb")
    heas = sorted(raw.glob("*.hea"))
    out, dropped = [], []
    per_rec_corr = {}

    for h in heas:
        recname = h.stem
        hdr = wfdb.rdheader(str(raw / recname))
        li = _lead_index(hdr.sig_name, 'II')
        if li is None:
            dropped.append((recname, 'no lead II', None))
            continue
        rec = wfdb.rdrecord(str(raw / recname))
        ann = wfdb.rdann(str(raw / recname), 'atr')
        fs = rec.fs

        # --- baseline-wander removal at native fs, THEN resample to 125 Hz ---
        s_native = rec.p_signal[:, li].astype(float)
        s_native = highpass_zero_phase(s_native, fs)
        s = signal.resample_poly(s_native, 125, int(round(fs)))

        # first pass: map + re-center peaks (now on a de-trended signal)
        samples_raw, labels, peaks = [], [], []
        for i in range(len(ann.sample)):
            sym = ann.symbol[i]
            if sym not in AAMI:
                continue
            p0 = int(ann.sample[i] * 125 / fs)
            lo, hi = max(0, p0 - 15), min(len(s), p0 + 15)
            if hi - lo < 10:
                continue
            p = lo + int(np.argmax(np.abs(s[lo:hi] - np.median(s[lo:hi]))))
            samples_raw.append(ann.sample[i])   # native-fs sample for RR
            labels.append(AAMI[sym])
            peaks.append(p)

        if not peaks:
            dropped.append((recname, 'no AAMI beats', None))
            continue

        # per-record polarity: make dominant QRS deflection positive
        devs = [s[p] - np.median(s[max(0, p - 60):p + 60]) for p in peaks[:500]]
        if np.median(devs) < 0:
            s = -s

        # morphology-gap filter on mean-N shape (on the cleaned signal)
        nbeats = []
        for p, lab in zip(peaks, labels):
            if lab != 0 or p - 90 < 0 or p + 97 > len(s):
                continue
            w = s[p - 90:p + 97]
            nbeats.append((w - w.mean()) / (w.std() + 1e-8))
        corr = float(np.corrcoef(template, np.mean(nbeats, axis=0))[0, 1]) if len(nbeats) >= 20 else np.nan
        per_rec_corr[recname] = corr
        if np.isnan(corr) or corr < min_corr:
            n_f = sum(1 for l in labels if l == 3)
            dropped.append((recname, f'meanN corr {corr:.2f} < {min_corr}', n_f))
            continue

        rr = rr_features_for_record(samples_raw, fs)
        for k, p in enumerate(peaks):
            a, b = p - 90, p + 97
            if a >= 0 and b <= len(s):
                out.append({'rec': f"incart/{recname}", 'src': 1,
                            'x': s[a:b].astype(np.float32),
                            'y': labels[k], 'rr': rr[k]})
    return out, dropped, per_rec_corr


# --------------------------------------------------------------------------------------
#  INCART patient-disjoint split, greedy-balanced by F then S beat count (~60/25/15)
# --------------------------------------------------------------------------------------
TARGET = {'train': 0.60, 'val': 0.25, 'test': 0.15}


def split_incart_records(beats, seed):
    by_rec = defaultdict(lambda: np.zeros(5, dtype=int))
    for b in beats:
        by_rec[b['rec']][b['y']] += 1
    recs = sorted(by_rec)
    rng = np.random.default_rng(seed)

    got = {k: np.zeros(5, dtype=float) for k in TARGET}
    assign = {}

    def place(rec_list, key_cls):
        total = sum(by_rec[r][key_cls] for r in rec_list) or 1
        for r in sorted(rec_list, key=lambda r: -by_rec[r][key_cls]):
            deficit = {k: TARGET[k] - got[k][key_cls] / total for k in TARGET}
            k = max(deficit, key=deficit.get)
            assign[r] = k
            got[k] += by_rec[r]

    f_recs = [r for r in recs if by_rec[r][3] > 0]
    s_recs = [r for r in recs if by_rec[r][3] == 0 and by_rec[r][1] > 0]
    rest = [r for r in recs if r not in f_recs and r not in s_recs]
    place(f_recs, 3)          # F-bearing records first, balanced on F count
    place(s_recs, 1)          # then S-only records, balanced on S count
    rng.shuffle(rest)         # N/V-only records: plain 60/25/15
    n = len(rest)
    for i, r in enumerate(rest):
        assign[r] = 'train' if i < 0.60 * n else ('val' if i < 0.85 * n else 'test')
    return assign, {r: by_rec[r].tolist() for r in recs}


# --------------------------------------------------------------------------------------
def dist(y):
    bc = np.bincount(np.asarray(y, dtype=int), minlength=5)
    return {CLASS_NAMES[i]: int(bc[i]) for i in range(5)}


def n_patients_per_class(beats):
    seen = defaultdict(set)
    for b in beats:
        seen[b['y']].add(b['rec'])
    return {CLASS_NAMES[c]: len(seen[c]) for c in range(5)}


def stack_x(beats):
    return np.stack([b['x'] for b in beats]).astype(np.float32)      # (N, 187)


def stack_rr(beats):
    return np.array([b['rr'] for b in beats], dtype=np.float32)      # (N, 3)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-corr', type=float, default=0.4,
                    help='drop INCART records whose mean-N corr to MIT-BIH template is below this')
    ap.add_argument('--incart-n-cap', type=int, default=25000,
                    help='subsample INCART Normal beats to this many total (keeps F/S/V fully)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path("data/processed/splits_rr_crossdb")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_splits_rr = Path("data/processed/splits_rr")

    # ---------------- MIT-BIH ----------------
    print("=== MIT-BIH ===")
    mit_train = extract_mitbih([p for p in DS1 if p not in VAL_MITBIH])
    mit_val = extract_mitbih(VAL_MITBIH)
    mit_ds2 = extract_mitbih(DS2)
    print(f"  DS1-train {len(mit_train)}  DS1-val {len(mit_val)}  DS2-test {len(mit_ds2)}")

    # ---------------- INCART ----------------
    print("=== INCART ===  (0.5 Hz high-pass, then mean-N template gap filter)")
    template = mitbih_meanN_template()
    inc_all, dropped, corrs = extract_incart(args.min_corr, template)
    print(f"  kept {len(inc_all)} beats from "
          f"{len(set(b['rec'] for b in inc_all))} records; dropped {len(dropped)} records")
    for r, why, nf in dropped:
        print(f"    drop {r}: {why}" + (f"  (loses {nf} F beats)" if nf else ""))

    # subsample INCART Normal beats (keep all F/S/V/Q) to avoid diluting class balance
    n_idx = [i for i, b in enumerate(inc_all) if b['y'] == 0]
    if len(n_idx) > args.incart_n_cap:
        rng0 = np.random.default_rng(args.seed)
        drop_n = set(rng0.choice(n_idx, len(n_idx) - args.incart_n_cap, replace=False).tolist())
        inc_all = [b for i, b in enumerate(inc_all) if i not in drop_n]
        print(f"  subsampled INCART N: {len(n_idx)} -> {args.incart_n_cap}")

    assign, rec_counts = split_incart_records(inc_all, args.seed)
    inc_train = [b for b in inc_all if assign[b['rec']] == 'train']
    inc_val = [b for b in inc_all if assign[b['rec']] == 'val']
    inc_test = [b for b in inc_all if assign[b['rec']] == 'test']
    print(f"  INCART split: train {len(inc_train)}  val {len(inc_val)}  test {len(inc_test)}")

    # ---------------- PER-SOURCE normalization ----------------
    # MIT-BIH stat: DS1-train only  =>  DS2 comes out identical to splits_rr/inter_test.npz
    Xmt_tr = stack_x(mit_train)
    xm_mit, xs_mit = float(Xmt_tr.mean()), float(Xmt_tr.std() + 1e-8)
    # INCART stat: INCART-train only
    Xin_tr = stack_x(inc_train)
    xm_inc, xs_inc = float(Xin_tr.mean()), float(Xin_tr.std() + 1e-8)
    print(f"  norm X  MIT-BIH mean/std = {xm_mit:.5f} / {xs_mit:.5f}")
    print(f"  norm X  INCART  mean/std = {xm_inc:.5f} / {xs_inc:.5f}")

    def norm_x(beats, src):
        X = stack_x(beats)
        if src == 0:
            X = (X - xm_mit) / xs_mit
        else:
            X = (X - xm_inc) / xs_inc
        return X.reshape(-1, 187, 1).astype(np.float32)

    # RR: per-source clip -> z-score (RR ratios are already scale-invariant, but keep
    # MIT-BIH RR identical to the paper by using the MIT-BIH-train stat for src 0)
    rr_mit_tr = np.clip(stack_rr(mit_train), 0.2, 3.0)
    rm_mit, rs_mit = rr_mit_tr.mean(axis=0), rr_mit_tr.std(axis=0) + 1e-8
    rr_inc_tr = np.clip(stack_rr(inc_train), 0.2, 3.0)
    rm_inc, rs_inc = rr_inc_tr.mean(axis=0), rr_inc_tr.std(axis=0) + 1e-8

    def norm_rr(beats, src):
        rr = np.clip(stack_rr(beats), 0.2, 3.0)
        if src == 0:
            return ((rr - rm_mit) / rs_mit).astype(np.float32)
        return ((rr - rm_inc) / rs_inc).astype(np.float32)

    def assemble(mit_beats, inc_beats):
        parts_x, parts_y, parts_rr, parts_s = [], [], [], []
        if mit_beats:
            parts_x.append(norm_x(mit_beats, 0))
            parts_y.append(np.array([b['y'] for b in mit_beats], dtype=np.int64))
            parts_rr.append(norm_rr(mit_beats, 0))
            parts_s.append(np.zeros(len(mit_beats), dtype=np.int64))
        if inc_beats:
            parts_x.append(norm_x(inc_beats, 1))
            parts_y.append(np.array([b['y'] for b in inc_beats], dtype=np.int64))
            parts_rr.append(norm_rr(inc_beats, 1))
            parts_s.append(np.ones(len(inc_beats), dtype=np.int64))
        return (np.concatenate(parts_x), np.concatenate(parts_y),
                np.concatenate(parts_rr), np.concatenate(parts_s))

    Xtr, ytr, rtr, str_ = assemble(mit_train, inc_train)
    Xva, yva, rva, sva = assemble(mit_val, inc_val)
    Xds2, yds2, rds2, sds2 = assemble(mit_ds2, [])
    Xit, yit, rit, sit = assemble([], inc_test)

    np.savez(out_dir / "inter_train.npz", X=Xtr, y=ytr, rr=rtr, source=str_)
    np.savez(out_dir / "inter_val.npz", X=Xva, y=yva, rr=rva, source=sva)
    np.savez(out_dir / "inter_test.npz", X=Xds2, y=yds2, rr=rds2, source=sds2)
    np.savez(out_dir / "incart_test.npz", X=Xit, y=yit, rr=rit, source=sit)
    np.save(out_dir / "inter_norm_mean.npy", np.array([xm_mit, xm_inc], dtype=np.float32))
    np.save(out_dir / "inter_norm_std.npy", np.array([xs_mit, xs_inc], dtype=np.float32))
    np.save(out_dir / "rr_norm_mean.npy", np.stack([rm_mit, rm_inc]).astype(np.float32))
    np.save(out_dir / "rr_norm_std.npy", np.stack([rs_mit, rs_inc]).astype(np.float32))

    # ---------------- verify DS2 == original splits_rr/inter_test.npz ----------------
    ds2_check = "original splits_rr/inter_test.npz not present — skipped"
    if (src_splits_rr / "inter_test.npz").exists():
        o = np.load(src_splits_rr / "inter_test.npz")
        if o['X'].shape == Xds2.shape and np.array_equal(o['y'], yds2):
            md = float(np.abs(o['X'] - Xds2).max())
            ds2_check = (f"MATCHES original (max|dX|={md:.2e})" if md < 1e-3
                         else f"DIFFERS from original (max|dX|={md:.3f}) — investigate")
        else:
            ds2_check = (f"shape/label mismatch vs original "
                         f"(orig X {o['X'].shape} y{np.bincount(o['y'])} ; "
                         f"new X {Xds2.shape} y{np.bincount(yds2)})")
    print(f"\n[DS2 vs paper split] {ds2_check}")

    # ---------------- manifest ----------------
    manifest = {
        "version": 2, "min_corr": args.min_corr, "seed": args.seed,
        "hp_filter": {"cutoff_hz": HP_CUTOFF_HZ, "order": HP_ORDER, "applied_to": "INCART only"},
        "norm_per_source": {
            "x_mean": {"mitbih": xm_mit, "incart": xm_inc},
            "x_std": {"mitbih": xs_mit, "incart": xs_inc},
            "rr_mean": {"mitbih": rm_mit.tolist(), "incart": rm_inc.tolist()},
            "rr_std": {"mitbih": rs_mit.tolist(), "incart": rs_inc.tolist()},
        },
        "ds2_vs_paper_split": ds2_check,
        "counts": {
            "train": dist(ytr), "val": dist(yva),
            "test_mitbih_DS2": dist(yds2), "test_incart": dist(yit),
        },
        "counts_by_source": {
            "train_mitbih": dist([b['y'] for b in mit_train]),
            "train_incart": dist([b['y'] for b in inc_train]),
            "val_mitbih": dist([b['y'] for b in mit_val]),
            "val_incart": dist([b['y'] for b in inc_val]),
        },
        "patients_per_class": {
            "train": n_patients_per_class(mit_train + inc_train),
            "val": n_patients_per_class(mit_val + inc_val),
            "test_incart": n_patients_per_class(inc_test),
        },
        "incart_record_assignment": assign,
        "incart_record_class_counts": rec_counts,
        "incart_meanN_corr": {k: (None if np.isnan(v) else round(v, 3)) for k, v in corrs.items()},
        "incart_dropped": [{"rec": r, "why": w, "lost_F": nf} for r, w, nf in dropped],
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ---------------- report ----------------
    print("\n================  CROSS-DB SPLIT SUMMARY (v2)  ================")
    for name, y, beats in [("TRAIN", ytr, mit_train + inc_train),
                           ("VAL", yva, mit_val + inc_val),
                           ("TEST MIT-BIH DS2 (primary)", yds2, mit_ds2),
                           ("TEST INCART (secondary)", yit, inc_test)]:
        line = "  ".join(f"{k}={v}" for k, v in dist(y).items())
        print(f"\n{name}:  n={len(y)}\n   {line}")
        if beats:
            pp = n_patients_per_class(beats)
            print("   patients/records per class: " + "  ".join(f"{k}={pp[k]}" for k in CLASS_NAMES))
    vf = n_patients_per_class(mit_val + inc_val)
    print(f"\n  >>> KEY CHECK: VAL F now from {vf['F']} records / {dist(yva)['F']} beats "
          f"(was 3 patients / 30 beats); VAL S from {vf['S']} records / {dist(yva)['S']} beats.")
    print(f"\n[OK] written to {out_dir}/  + split_manifest.json")


if __name__ == "__main__":
    main()
