# validation/preprocess_aami_rr_2ch.py
"""
Ban 2-KENH cua preprocess_aami_rr.py -- thay vi chi lay p_signal[:,0] (MLII),
lay CA 2 kenh ghi duoc trong MIT-BIH (moi ban ghi co dung 2 kenh). Da kiem
chung truoc (thong ke, khong can GPU): kenh phu (thuong la V1, mot so ban
ghi la V2/V4/V5) tach N-vs-F TOT HON MLII rieng le (benh nhan 213: AUC
0.7856 -> 0.9080; benh nhan 208: 0.8745 -> 0.9242, template rieng tung
benh nhan).

QUAN TRONG: khong phai moi ban ghi co thu tu kenh giong nhau -- ban ghi 114
co thu tu DAO NGUOC (kenh 0=V5, kenh 1=MLII), tat ca ban ghi khac co MLII o
kenh 0. Script nay TIM MLII theo TEN (khong gia dinh vi tri co dinh) de
luon dat MLII o kenh dau ra thu 1, kenh con lai (bat ky ten gi -- V1/V2/V4/
V5) o kenh dau ra thu 2 -- dam bao nhat quan "MLII truoc, kenh phu sau" cho
MOI ban ghi.

Ghi ra data/processed/splits_rr_2ch/ (thu muc rieng, khong dung den
splits_rr/ 1-kenh cu).
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


def get_channel_indices(record):
    """Tra ve (idx_mlii, idx_other) -- tim MLII theo TEN, khong gia dinh vi
    tri. Neu khong tim thay MLII (khong xay ra trong MIT-BIH nhung de an
    toan), fallback ve (0,1)."""
    names = record.sig_name
    if "MLII" in names:
        idx_mlii = names.index("MLII")
        idx_other = 1 - idx_mlii if len(names) == 2 else next(i for i in range(len(names)) if i != idx_mlii)
        return idx_mlii, idx_other
    return 0, 1


def extract_beats_with_rr_2ch():
    raw_dir = Path("data/raw/mitdb")
    out_dir = Path("data/processed/splits_rr_2ch")
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in RECORDS:
        if not (raw_dir / f"{r}.dat").exists() or not (raw_dir / f"{r}.atr").exists():
            print(f"Record {r} files not found.")
            return False

    print("All records found. Processing heartbeats (2-channel) with RR-interval features...")

    all_beats = []

    for r in RECORDS:
        record_path = str(raw_dir / r)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')

        idx_mlii, idx_other = get_channel_indices(record)
        other_name = record.sig_name[idx_other]
        print(f"  Processing record {r}... (kenh 1=MLII, kenh 2={other_name})")

        sig_mlii = record.p_signal[:, idx_mlii]
        sig_other = record.p_signal[:, idx_other]
        fs = record.fs
        fs_int = int(round(fs))
        sig_mlii_125 = signal.resample_poly(sig_mlii, 125, fs_int)
        sig_other_125 = signal.resample_poly(sig_other, 125, fs_int)

        beat_samples, beat_labels = [], []
        for i in range(len(ann.sample)):
            symbol = ann.symbol[i]
            if symbol in AAMI_MAPPING:
                beat_samples.append(ann.sample[i])
                beat_labels.append(AAMI_MAPPING[symbol])

        beat_samples = np.array(beat_samples)
        rr_all = np.diff(beat_samples) / fs
        record_median_rr = float(np.median(rr_all)) if len(rr_all) > 0 else 0.8
        EPS = 1e-6

        for k, ann_sample in enumerate(beat_samples):
            label = beat_labels[k]
            peak_idx = int(ann_sample * 125 / fs)
            start = peak_idx - 90
            end = peak_idx + 97
            if start >= 0 and end <= len(sig_mlii_125):
                beat_mlii = sig_mlii_125[start:end]
                beat_other = sig_other_125[start:end]

                pre_rr = float(rr_all[k - 1]) if k > 0 else record_median_rr
                post_rr = float(rr_all[k]) if k < len(rr_all) else record_median_rr
                local_avg_before = float(np.mean(rr_all[max(0, k - 5):k])) if k > 0 else record_median_rr

                pre_rr_ratio = pre_rr / max(local_avg_before, EPS)
                post_rr_ratio = post_rr / max(pre_rr, EPS)
                local_rr_ratio = local_avg_before / max(record_median_rr, EPS)

                all_beats.append({
                    'patient': r,
                    'x_mlii': beat_mlii.tolist(),
                    'x_other': beat_other.tolist(),
                    'y': int(label),
                    'rr': [pre_rr_ratio, post_rr_ratio, local_rr_ratio],
                })

    print(f"Extracted {len(all_beats)} beats in total.")

    val_patients = ['122', '124', '205', '223']
    train_patients = [p for p in DS1 if p not in val_patients]
    test_patients = DS2

    train_beats = [b for b in all_beats if b['patient'] in train_patients]
    val_beats = [b for b in all_beats if b['patient'] in val_patients]
    test_beats = [b for b in all_beats if b['patient'] in test_patients]

    def to_arrays(beats):
        X_mlii = np.array([b['x_mlii'] for b in beats], dtype=np.float32)
        X_other = np.array([b['x_other'] for b in beats], dtype=np.float32)
        X = np.stack([X_mlii, X_other], axis=-1)  # (N, 187, 2)
        y = np.array([b['y'] for b in beats], dtype=np.int64)
        rr = np.array([b['rr'] for b in beats], dtype=np.float32)
        return X, y, rr

    X_train, y_train, rr_train = to_arrays(train_beats)
    X_val, y_val, rr_val = to_arrays(val_beats)
    X_test, y_test, rr_test = to_arrays(test_beats)

    print("Normalizing X (MOI KENH RIENG) using Train-Only Statistics...")
    x_mean = X_train.mean(axis=(0, 1))  # (2,) -- 1 mean/std cho moi kenh
    x_std = X_train.std(axis=(0, 1)) + 1e-8
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    print("Clipping RR ratios to [0.2, 3.0]...")
    rr_train = np.clip(rr_train, 0.2, 3.0)
    rr_val = np.clip(rr_val, 0.2, 3.0)
    rr_test = np.clip(rr_test, 0.2, 3.0)

    print("Normalizing RR ratio features using Train-Only Statistics...")
    rr_mean = rr_train.mean(axis=0)
    rr_std = rr_train.std(axis=0) + 1e-8
    rr_train = (rr_train - rr_mean) / rr_std
    rr_val = (rr_val - rr_mean) / rr_std
    rr_test = (rr_test - rr_mean) / rr_std

    np.savez(out_dir / "inter_train.npz", X=X_train, y=y_train, rr=rr_train)
    np.savez(out_dir / "inter_val.npz", X=X_val, y=y_val, rr=rr_val)
    np.savez(out_dir / "inter_test.npz", X=X_test, y=y_test, rr=rr_test)
    np.save(out_dir / "inter_norm_mean.npy", x_mean.astype(np.float32))
    np.save(out_dir / "inter_norm_std.npy", x_std.astype(np.float32))
    np.save(out_dir / "rr_norm_mean.npy", rr_mean.astype(np.float32))
    np.save(out_dir / "rr_norm_std.npy", rr_std.astype(np.float32))

    print(f"Inter-patient 2-channel (+RR) splits saved: train={len(train_beats)}, val={len(val_beats)}, test={len(test_beats)}")
    print(f"X shape: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

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
    extract_beats_with_rr_2ch()
