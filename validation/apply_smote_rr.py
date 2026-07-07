"""
validation/apply_smote_rr.py
==============================
Apply SMOTE oversampling to the minority classes (S, F) of the RR-ratio
prototype training set — and ONLY the training set.

CRITICAL correctness requirement (this is exactly what the earlier,
never-committed attempt at this got wrong per CRITICAL_FIXES_SUMMARY.md
issue #8 — "SMOTE applied before cross-patient split validation"):
SMOTE MUST run strictly AFTER the inter-patient DS1/DS2 split, and MUST
only ever touch inter_train.npz. Synthetic beats are linear interpolations
between two REAL beats of the same class from the training patients only,
so a synthetic sample can never "leak" information from a DS2 (test)
patient. inter_val.npz / inter_test.npz are copied through completely
unchanged (not even re-read beyond the copy) — no resampling, no fitting
of any statistic on them here.

Input:  data/processed/splits_rr/{inter_train,inter_val,inter_test}.npz
        (each has X: (N,187,1), y: (N,), rr: (N,3) — produced by
        validation/preprocess_aami_rr.py, already patient-disjoint)
Output: data/processed/splits_rr_smote/{inter_train,inter_val,inter_test}.npz

SMOTE runs on the concatenated [flattened waveform (187) + rr ratios (3)]
= 190-dim vector per beat, so a synthetic sample's waveform and its RR
context stay mutually consistent (both are the same interpolation weight
applied to the same pair of real neighbor beats).

By default, classes S (label 1) and F (label 3) are oversampled (moderate
boost, not fully balanced to N) — V already trains well in every experiment
so far, and Q has only 8 training beats, too few to synthesize reliably.
Use --oversample-f 0 to isolate whether SMOTE-ing F specifically is what's
been hurting F1(F) across the RR+SMOTE prototype runs, vs. S-only SMOTE.

Cach chay:
    # Default: oversample both S and F
    python validation/apply_smote_rr.py

    # S-only (isolate whether F-oversampling is what's hurting F1(F)):
    python validation/apply_smote_rr.py --oversample-f 0 \\
        --out-dir data/processed/splits_rr_smote_s_only
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE

SRC_DIR = Path("data/processed/splits_rr")

# Default target counts after oversampling (S: 836 -> ~3300, F: 384 -> ~1500).
# N(0), V(2), Q(4) are left untouched -- not passed in sampling_strategy.
DEFAULT_SAMPLING_STRATEGY = {1: 3300, 3: 1500}
K_NEIGHBORS = 5
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Apply SMOTE to the RR-ratio training set only")
    parser.add_argument("--out-dir", default="data/processed/splits_rr_smote",
                        help="Output directory for the oversampled train set (val/test copied unchanged)")
    parser.add_argument("--oversample-s", type=int, default=3300,
                        help="Target S (label 1) count after SMOTE, or 0 to leave S untouched")
    parser.add_argument("--oversample-f", type=int, default=1500,
                        help="Target F (label 3) count after SMOTE, or 0 to leave F untouched")
    args = parser.parse_args()

    dst_dir = Path(args.out_dir)
    sampling_strategy = {}
    if args.oversample_s > 0:
        sampling_strategy[1] = args.oversample_s
    if args.oversample_f > 0:
        sampling_strategy[3] = args.oversample_f
    if not sampling_strategy:
        print("[ERROR] Both --oversample-s and --oversample-f are 0 -- nothing to do.")
        return

    if not (SRC_DIR / "inter_train.npz").exists():
        print(f"[ERROR] {SRC_DIR}/inter_train.npz not found. "
              f"Run: python validation/preprocess_aami_rr.py first.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {dst_dir}")
    print(f"Sampling strategy: {sampling_strategy}")

    print("[Copying val/test UNCHANGED -- SMOTE must never touch them]")
    for split in ["inter_val", "inter_test"]:
        shutil.copy(SRC_DIR / f"{split}.npz", dst_dir / f"{split}.npz")
        print(f"  copied {split}.npz")

    print("\n[Loading training set]")
    d = np.load(SRC_DIR / "inter_train.npz")
    X, y, rr = d["X"], d["y"], d["rr"]
    print(f"  X: {X.shape}  y: {y.shape}  rr: {rr.shape}")

    before = np.bincount(y, minlength=5)
    print(f"  Before SMOTE — N/S/V/F/Q: {before.tolist()}")

    n = X.shape[0]
    X_flat = X.reshape(n, -1)               # (n, 187)
    combined = np.concatenate([X_flat, rr], axis=1)  # (n, 190)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=K_NEIGHBORS,
        random_state=SEED,
    )
    combined_res, y_res = smote.fit_resample(combined, y)

    X_res = combined_res[:, :X_flat.shape[1]].reshape(-1, 187, 1).astype(np.float32)
    rr_res = combined_res[:, X_flat.shape[1]:].astype(np.float32)
    y_res = y_res.astype(np.int64)

    after = np.bincount(y_res, minlength=5)
    print(f"  After SMOTE  — N/S/V/F/Q: {after.tolist()}")
    print(f"  Total beats: {n} -> {len(y_res)}")

    # Shuffle so SMOTE's appended synthetic block at the tail doesn't bias
    # batch composition early in training (DataLoader shuffle=True already
    # handles this per-epoch, but a fixed shuffle here keeps the saved
    # array itself representative if inspected directly).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_res))
    X_res, y_res, rr_res = X_res[perm], y_res[perm], rr_res[perm]

    np.savez(dst_dir / "inter_train.npz", X=X_res, y=y_res, rr=rr_res)
    print(f"\n[OK] Saved -> {dst_dir / 'inter_train.npz'}")
    print(f"[OK] val/test copied unchanged from {SRC_DIR}")


if __name__ == "__main__":
    main()
