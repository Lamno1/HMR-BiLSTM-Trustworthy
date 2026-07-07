# validation/preprocess_aami_rr.py
"""
Prototype preprocessing: same inter-patient AAMI pipeline as preprocess_aami.py,
but each beat also carries 3 RR-interval features (pre_rr, post_rr, local_rr).

Writes to a SEPARATE output directory (data/processed/splits_rr/) so the
existing data/processed/splits/ used by the whole rest of the codebase is
left untouched.

Only generates the inter-patient split (that's what the RR prototype needs).

Cach chay:
    python validation/preprocess_aami_rr.py
"""
import numpy as np
import scipy.signal as signal
from pathlib import Path
import wfdb

DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
RECORDS = DS1 + DS2

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}


def extract_beats_with_rr():
    raw_dir = Path("data/raw/mitdb")
    out_dir = Path("data/processed/splits_rr")
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in RECORDS:
        if not (raw_dir / f"{r}.dat").exists() or not (raw_dir / f"{r}.atr").exists():
            print(f"Record {r} files not found.")
            return False

    print("All records found. Processing heartbeats with RR-interval features...")

    all_beats = []

    for r in RECORDS:
        print(f"  Processing record {r}...")
        record_path = str(raw_dir / r)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')

        sig = record.p_signal[:, 0]
        fs = record.fs
        fs_int = int(round(fs))
        sig_125 = signal.resample_poly(sig, 125, fs_int)

        # First pass: collect AAMI-mapped beat samples/labels in annotation order,
        # so RR intervals reflect true physiological beat-to-beat timing
        # (independent of whether the 187-sample window later fits in bounds).
        beat_samples, beat_labels = [], []
        for i in range(len(ann.sample)):
            symbol = ann.symbol[i]
            if symbol in AAMI_MAPPING:
                beat_samples.append(ann.sample[i])
                beat_labels.append(AAMI_MAPPING[symbol])

        beat_samples = np.array(beat_samples)
        rr_all = np.diff(beat_samples) / fs  # seconds; rr_all[k] = interval beat[k] -> beat[k+1]
        record_median_rr = float(np.median(rr_all)) if len(rr_all) > 0 else 0.8  # ~75 bpm fallback
        EPS = 1e-6

        for k, ann_sample in enumerate(beat_samples):
            label = beat_labels[k]
            peak_idx = int(ann_sample * 125 / fs)
            start = peak_idx - 90
            end = peak_idx + 97
            if start >= 0 and end <= len(sig_125):
                beat = sig_125[start:end]

                pre_rr  = float(rr_all[k - 1]) if k > 0 else record_median_rr
                post_rr = float(rr_all[k])     if k < len(rr_all) else record_median_rr
                # Recent local rhythm: mean of up to 5 preceding RR intervals.
                local_avg_before = float(np.mean(rr_all[max(0, k - 5):k])) if k > 0 else record_median_rr

                # Patient-relative RATIOS (unitless, scale-invariant to each patient's own
                # baseline heart rate) instead of absolute seconds + global z-score:
                #   pre_rr_ratio  ~1.0 = normal timing, <1.0 = premature beat (S/V signature)
                #   post_rr_ratio ~1.0 = no pause, >1.0 = compensatory pause after ectopic beat
                #   local_rr_ratio = is the local neighborhood faster/slower than this
                #                    patient's own record-level baseline rhythm
                pre_rr_ratio   = pre_rr / max(local_avg_before, EPS)
                post_rr_ratio  = post_rr / max(pre_rr, EPS)
                local_rr_ratio = local_avg_before / max(record_median_rr, EPS)

                all_beats.append({
                    'patient': r,
                    'x': beat.tolist(),
                    'y': int(label),
                    'rr': [pre_rr_ratio, post_rr_ratio, local_rr_ratio],
                })

    print(f"Extracted {len(all_beats)} beats in total.")

    val_patients = ['122', '124', '205', '223']
    train_patients = [p for p in DS1 if p not in val_patients]
    test_patients = DS2

    train_beats = [b for b in all_beats if b['patient'] in train_patients]
    val_beats   = [b for b in all_beats if b['patient'] in val_patients]
    test_beats  = [b for b in all_beats if b['patient'] in test_patients]

    def to_arrays(beats):
        X  = np.array([b['x'] for b in beats], dtype=np.float32)
        y  = np.array([b['y'] for b in beats], dtype=np.int64)
        rr = np.array([b['rr'] for b in beats], dtype=np.float32)
        return X, y, rr

    X_train, y_train, rr_train = to_arrays(train_beats)
    X_val,   y_val,   rr_val   = to_arrays(val_beats)
    X_test,  y_test,  rr_test  = to_arrays(test_beats)

    print("Normalizing X using Train-Only Statistics...")
    x_mean = X_train.mean()
    x_std  = X_train.std() + 1e-8
    X_train = (X_train - x_mean) / x_std
    X_val   = (X_val   - x_mean) / x_std
    X_test  = (X_test  - x_mean) / x_std

    # Clip patient-relative RR ratios before z-scoring so a rare artifact/missed-beat
    # record (which can produce a huge one-off ratio) doesn't skew the global mean/std
    # used to scale every other beat.
    print("Clipping RR ratios to [0.2, 3.0]...")
    rr_train = np.clip(rr_train, 0.2, 3.0)
    rr_val   = np.clip(rr_val,   0.2, 3.0)
    rr_test  = np.clip(rr_test,  0.2, 3.0)

    print("Normalizing RR ratio features using Train-Only Statistics...")
    rr_mean = rr_train.mean(axis=0)
    rr_std  = rr_train.std(axis=0) + 1e-8
    rr_train = (rr_train - rr_mean) / rr_std
    rr_val   = (rr_val   - rr_mean) / rr_std
    rr_test  = (rr_test  - rr_mean) / rr_std
    print(f"  RR ratio train mean (pre/post/local): {rr_mean}")
    print(f"  RR ratio train std  (pre/post/local): {rr_std}")

    X_train = X_train.reshape(-1, 187, 1)
    X_val   = X_val.reshape(-1, 187, 1)
    X_test  = X_test.reshape(-1, 187, 1)

    np.savez(out_dir / "inter_train.npz", X=X_train, y=y_train, rr=rr_train)
    np.savez(out_dir / "inter_val.npz",   X=X_val,   y=y_val,   rr=rr_val)
    np.savez(out_dir / "inter_test.npz",  X=X_test,  y=y_test,  rr=rr_test)
    np.save(out_dir / "inter_norm_mean.npy", np.array([x_mean], dtype=np.float32))
    np.save(out_dir / "inter_norm_std.npy",  np.array([x_std],  dtype=np.float32))
    np.save(out_dir / "rr_norm_mean.npy", rr_mean.astype(np.float32))
    np.save(out_dir / "rr_norm_std.npy",  rr_std.astype(np.float32))

    print(f"Inter-patient (+RR) splits saved: train={len(train_beats)}, val={len(val_beats)}, test={len(test_beats)}")

    def print_dist(name, y):
        unique, counts = np.unique(y, return_counts=True)
        dist_dict = {u: c for u, c in zip(unique, counts)}
        n_total = len(y) if len(y) > 0 else 1
        print(f"  {name}: ", end="")
        for cls in range(5):
            c = dist_dict.get(cls, 0)
            print(f"Class {cls}: {c:<5} ({c/n_total*100:5.1f}%) | ", end="")
        print()

    print("\nInter-patient Class Distribution:")
    print_dist("Train (DS1)", y_train)
    print_dist("Val (DS1)", y_val)
    print_dist("Test (DS2)", y_test)

    return True


if __name__ == "__main__":
    extract_beats_with_rr()
