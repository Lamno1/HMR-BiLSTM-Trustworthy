# validation/preprocess_aami.py
import os
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from pathlib import Path
import wfdb
import json

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

def extract_beats():
    raw_dir = Path("data/raw/mitdb")
    out_dir = Path("data/processed/splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if download is complete
    for r in RECORDS:
        if not (raw_dir / f"{r}.dat").exists() or not (raw_dir / f"{r}.atr").exists():
            print(f"Record {r} files not found. Please wait for the download task to finish.")
            return False
            
    print("All records found. Processing heartbeats...")
    
    all_beats = []
    
    for r in RECORDS:
        print(f"  Processing record {r}...")
        # Load signal and annotations
        record_path = str(raw_dir / r)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
        
        # We use channel 0 (usually MLII or Lead II)
        sig = record.p_signal[:, 0]
        fs = record.fs
        
        # Resample signal to 125 Hz using polyphase filtering.
        # Use round() for the denominator so non-integer fs values (e.g. 360.0)
        # produce an exact integer ratio instead of being silently truncated.
        fs_int = int(round(fs))
        sig_125 = signal.resample_poly(sig, 125, fs_int)

        # Map annotations and segment
        for i in range(len(ann.sample)):
            symbol = ann.symbol[i]
            if symbol in AAMI_MAPPING:
                label = AAMI_MAPPING[symbol]
                ann_sample = ann.sample[i]

                # Resampled peak index (use float fs for precision)
                peak_idx = int(ann_sample * 125 / fs)
                
                # Extract window of 187 samples (90 before, 97 after)
                start = peak_idx - 90
                end = peak_idx + 97
                
                if start >= 0 and end <= len(sig_125):
                    beat = sig_125[start:end]
                    all_beats.append({
                        'patient': r,
                        'x': beat.tolist(),
                        'y': int(label)
                    })
                        
    print(f"Extracted {len(all_beats)} beats in total.")
    
    # Save the full extracted beats as JSON (npz with list-of-dicts requires allow_pickle=True)
    beats_json_path = out_dir / "all_extracted_beats.json"
    with open(beats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_beats, f)
    print(f"Full extracted beats saved to {beats_json_path}.")
    
    # --- 1. Generate Inter-patient split ---
    print("Generating Inter-patient split...")
    # Train: DS1 excluding some for Val
    val_patients = ['122', '124', '205', '223']
    train_patients = [p for p in DS1 if p not in val_patients]
    
    assert set(train_patients).isdisjoint(set(val_patients)), "CRITICAL BUG: Patient overlap detected between train and val!"
    test_patients = DS2
    
    train_beats = [b for b in all_beats if b['patient'] in train_patients]
    val_beats = [b for b in all_beats if b['patient'] in val_patients]
    test_beats = [b for b in all_beats if b['patient'] in test_patients]
    
    X_train = np.array([b['x'] for b in train_beats], dtype=np.float32)
    y_train = np.array([b['y'] for b in train_beats], dtype=np.int64)
    X_val = np.array([b['x'] for b in val_beats], dtype=np.float32)
    y_val = np.array([b['y'] for b in val_beats], dtype=np.int64)
    X_test = np.array([b['x'] for b in test_beats], dtype=np.float32)
    y_test = np.array([b['y'] for b in test_beats], dtype=np.int64)
    
    # --- Normalize using Train-Only Statistics ---
    print("Normalizing using Train-Only Statistics...")
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    print(f"  Train Mean: {mean:.6f}")
    print(f"  Train Std:  {std:.6f}")
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Reshape for LSTM: (N, 187, 1)
    X_train = X_train.reshape(-1, 187, 1)
    X_val = X_val.reshape(-1, 187, 1)
    X_test = X_test.reshape(-1, 187, 1)
    
    np.savez(out_dir / "inter_train.npz", X=X_train, y=y_train)
    np.savez(out_dir / "inter_val.npz", X=X_val, y=y_val)
    np.savez(out_dir / "inter_test.npz", X=X_test, y=y_test)
    
    # Save the normalization statistics
    np.save(out_dir / "inter_norm_mean.npy", np.array([mean], dtype=np.float32))
    np.save(out_dir / "inter_norm_std.npy", np.array([std], dtype=np.float32))
    
    print(f"Inter-patient splits saved: train={len(train_beats)}, val={len(val_beats)}, test={len(test_beats)}")
    
    # --- Inter-patient Distribution Report ---
    def print_dist(name, y):
        unique, counts = np.unique(y, return_counts=True)
        # Force padding for 5 classes (0: N, 1: S, 2: V, 3: F, 4: Q) to ensure stable output
        dist_dict = {u: c for u, c in zip(unique, counts)}
        print(f"  {name}: ", end="")
        n_total = len(y) if len(y) > 0 else 1
        for cls in range(5):
            c = dist_dict.get(cls, 0)
            print(f"Class {cls}: {c:<5} ({c/n_total*100:5.1f}%) | ", end="")
        print()
    
    print("\nInter-patient Class Distribution:")
    print_dist("Train (DS1)", y_train)
    print_dist("Val (DS1)", y_val)
    print_dist("Test (DS2)", y_test)
    
    # --- 2. Generate Intra-patient split (stratified) ---
    print("Generating Intra-patient split (stratified)...")
    from sklearn.model_selection import train_test_split as _tts
    y_all_tmp = np.array([b['y'] for b in all_beats], dtype=np.int64)
    all_idx = np.arange(len(all_beats))
    train_idx, tmp_idx = _tts(all_idx, test_size=0.30, stratify=y_all_tmp, random_state=42)
    y_tmp = y_all_tmp[tmp_idx]
    val_idx, test_idx = _tts(tmp_idx, test_size=0.5, stratify=y_tmp, random_state=42)
    
    X_all = np.array([b['x'] for b in all_beats], dtype=np.float32).reshape(-1, 187, 1)
    y_all = np.array([b['y'] for b in all_beats], dtype=np.int64)

    X_intra_train = X_all[train_idx]
    X_intra_val   = X_all[val_idx]
    X_intra_test  = X_all[test_idx]

    # Normalize using intra train-only statistics (same convention as inter-patient splits)
    intra_mean = X_intra_train.mean()
    intra_std  = X_intra_train.std() + 1e-8
    X_intra_train = (X_intra_train - intra_mean) / intra_std
    X_intra_val   = (X_intra_val   - intra_mean) / intra_std
    X_intra_test  = (X_intra_test  - intra_mean) / intra_std
    print(f"  Intra norm — mean: {intra_mean:.6f}  std: {intra_std:.6f}")

    np.savez(out_dir / "intra_train.npz", X=X_intra_train, y=y_all[train_idx])
    np.savez(out_dir / "intra_val.npz",   X=X_intra_val,   y=y_all[val_idx])
    np.savez(out_dir / "intra_test.npz",  X=X_intra_test,  y=y_all[test_idx])
    np.save(out_dir / "intra_norm_mean.npy", np.array([intra_mean], dtype=np.float32))
    np.save(out_dir / "intra_norm_std.npy",  np.array([intra_std],  dtype=np.float32))
    print(f"Intra-patient splits saved: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return True

if __name__ == "__main__":
    extract_beats()
