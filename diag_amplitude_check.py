"""
diag_amplitude_check.py
=======================
Tính toán biên độ tuyệt đối trung bình (mean absolute amplitude)
của vùng Trung tính [50, 65] so với vùng Onset và vùng QRS.
"""

import sys
import io
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from configs.paths import INTER_TEST

# Load shared indices
SHAP_RESULTS = "outputs/v1.0_20260611_160039/explainability/results.json"
with open(SHAP_RESULTS, "r", encoding="utf-8") as f:
    shap_res = json.load(f)
shared_idx_c = shap_res["metrics"]["shared_idx_c"]
f_shared = shared_idx_c["F"]
s_shared = shared_idx_c["S"]

# Load test data
test = np.load(INTER_TEST)
X_test = test["X"].astype(np.float32).squeeze(-1) # (n, 187)
y_test = test["y"].astype(np.int64)

NEUTRAL_ZONE = (50, 65)

def analyze_amplitudes(cn, cls_id, shared_idxs, onset_zone):
    print(f"\n" + "="*80)
    print(f" AMPLITUDE PROFILE FOR CLASS {cn} (Class ID: {cls_id})")
    print(f"="*80)
    
    idx_all = np.where(y_test == cls_id)[0]
    
    # 1. Đo trên 30 shared beats
    X_shared = X_test[shared_idxs]
    mean_abs_onset_shared = np.mean(np.abs(X_shared[:, onset_zone[0]:onset_zone[1]]))
    mean_abs_neutral_shared = np.mean(np.abs(X_shared[:, NEUTRAL_ZONE[0]:NEUTRAL_ZONE[1]]))
    mean_abs_qrs_shared = np.mean(np.abs(X_shared[:, 85:97]))
    
    # 2. Đo trên toàn bộ quần thể (all beats of this class)
    X_all = X_test[idx_all]
    mean_abs_onset_all = np.mean(np.abs(X_all[:, onset_zone[0]:onset_zone[1]]))
    mean_abs_neutral_all = np.mean(np.abs(X_all[:, NEUTRAL_ZONE[0]:NEUTRAL_ZONE[1]]))
    mean_abs_qrs_all = np.mean(np.abs(X_all[:, 85:97]))
    
    print(f"Trên 30 shared beats:")
    print(f"  Onset {onset_zone}  : {mean_abs_onset_shared:.4f}")
    print(f"  Neutral {NEUTRAL_ZONE}: {mean_abs_neutral_shared:.4f}")
    print(f"  QRS [85, 97]   : {mean_abs_qrs_shared:.4f}")
    print(f"  Tỉ lệ Onset/Neutral  : {mean_abs_onset_shared / mean_abs_neutral_shared:.2f}x")
    print(f"  Tỉ lệ QRS/Neutral    : {mean_abs_qrs_shared / mean_abs_neutral_shared:.2f}x")
    
    print(f"\nTrên toàn bộ {len(idx_all)} beats của test set:")
    print(f"  Onset {onset_zone}  : {mean_abs_onset_all:.4f}")
    print(f"  Neutral {NEUTRAL_ZONE}: {mean_abs_neutral_all:.4f}")
    print(f"  QRS [85, 97]   : {mean_abs_qrs_all:.4f}")
    print(f"  Tỉ lệ Onset/Neutral  : {mean_abs_onset_all / mean_abs_neutral_all:.2f}x")
    print(f"  Tỉ lệ QRS/Neutral    : {mean_abs_qrs_all / mean_abs_neutral_all:.2f}x")

# Run for F (Fusion)
analyze_amplitudes("F", 3, f_shared, (15, 35))

# Run for S (APC)
analyze_amplitudes("S", 1, s_shared, (25, 45))
