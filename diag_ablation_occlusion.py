"""
diag_ablation_occlusion.py
==========================
Measure causal sensitivity of the model to Onset [0, 45], QRS [85, 97],
and a Neutral Control zone [50, 65] on F and S classes using signal ablation/occlusion.
Optimized to run predictions only on the subsets of target classes (100x faster).
"""

import sys
import io
import json
import numpy as np
import torch
from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, INTER_TEST

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load test data
test = np.load(INTER_TEST)
X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
model.eval()

# baseline PR segment
BASELINE_ZONE = (55, 80)
NEUTRAL_ZONE = (50, 65)

def get_predictions(X):
    preds = []
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256]
            preds.append(model(b).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds) if len(preds) > 0 else np.array([])

# Original predictions
print("Running baseline model on unablated signals...")
preds_orig = get_predictions(X_test)
print(f"Overall Test Accuracy: { (preds_orig == y_test).mean():.4f}")

def ablate_zone(X_subset, zone_to_ablate, use_pr_median=True):
    """
    Mask signal zone in specified subset of beats.
    If use_pr_median=True, replace with the median of PR segment [55, 80] for each beat.
    If use_pr_median=False, replace with 0.
    """
    X_new = X_subset.copy()
    s_z, e_z = zone_to_ablate
    for i in range(len(X_new)):
        sig = X_new[i].squeeze()
        if use_pr_median:
            val = np.median(sig[BASELINE_ZONE[0]:BASELINE_ZONE[1]])
        else:
            val = 0.0
        X_new[i, s_z:e_z, 0] = val
    return X_new

def run_experiment(cn, cls_id, onset_zone):
    print(f"\n" + "="*80)
    print(f" ABLATION EXPERIMENT FOR CLASS {cn} (Class ID: {cls_id})")
    print(f"="*80)
    
    idx_true = np.where(y_test == cls_id)[0]
    idx_correct = np.where((y_test == cls_id) & (preds_orig == cls_id))[0]
    n_true = len(idx_true)
    n_correct = len(idx_correct)
    
    print(f"Total true beats of class {cn} in test set: {n_true}")
    print(f"Beats correctly predicted by baseline model: {n_correct} (Recall: {n_correct/n_true:.2%})")
    
    # We only run on the subset of true class beats
    X_subset = X_test[idx_true]
    correct_in_true_mask = np.isin(idx_true, idx_correct)
    
    for method_name, use_pr_med in [("PR Median (Soft)", True), ("Zero Flatline (Hard)", False)]:
        print(f"\n--- Masking method: {method_name} ---")
        
        # 1. Ablate Onset
        X_ab_onset = ablate_zone(X_subset, onset_zone, use_pr_median=use_pr_med)
        preds_ab_onset = get_predictions(X_ab_onset)
        recall_ab_onset = (preds_ab_onset == cls_id).mean()
        n_correct_ab_onset = sum(preds_ab_onset[correct_in_true_mask] == cls_id)
        
        # 2. Ablate QRS
        X_ab_qrs = ablate_zone(X_subset, (85, 97), use_pr_median=use_pr_med)
        preds_ab_qrs = get_predictions(X_ab_qrs)
        recall_ab_qrs = (preds_ab_qrs == cls_id).mean()
        n_correct_ab_qrs = sum(preds_ab_qrs[correct_in_true_mask] == cls_id)

        # 3. Ablate Neutral Control Zone [50, 65]
        X_ab_neutral = ablate_zone(X_subset, NEUTRAL_ZONE, use_pr_median=use_pr_med)
        preds_ab_neutral = get_predictions(X_ab_neutral)
        recall_ab_neutral = (preds_ab_neutral == cls_id).mean()
        n_correct_ab_neutral = sum(preds_ab_neutral[correct_in_true_mask] == cls_id)
        
        print(f"Ablate Onset {onset_zone}:")
        print(f"  Recall: {recall_ab_onset:.2%} ({sum(preds_ab_onset == cls_id)}/{n_true})")
        print(f"  Correctly predicted baseline beats preserved: {n_correct_ab_onset}/{n_correct} ({n_correct_ab_onset/n_correct:.1%})")
        
        print(f"Ablate QRS [85, 97]:")
        print(f"  Recall: {recall_ab_qrs:.2%} ({sum(preds_ab_qrs == cls_id)}/{n_true})")
        print(f"  Correctly predicted baseline beats preserved: {n_correct_ab_qrs}/{n_correct} ({n_correct_ab_qrs/n_correct:.1%})")

        print(f"Ablate Neutral Control Zone {NEUTRAL_ZONE}:")
        print(f"  Recall: {recall_ab_neutral:.2%} ({sum(preds_ab_neutral == cls_id)}/{n_true})")
        print(f"  Correctly predicted baseline beats preserved: {n_correct_ab_neutral}/{n_correct} ({n_correct_ab_neutral/n_correct:.1%})")
        
        print(f"Prediction distribution of baseline-correct beats when Onset is ablated:")
        pred_counts = np.bincount(preds_ab_onset[correct_in_true_mask], minlength=5)
        for i, count in enumerate(pred_counts):
            if count > 0:
                print(f"  -> Predicted as Class {i}: {count} beats ({count/n_correct:.1%})")

# Run for F (Fusion, Class 3)
run_experiment("F", 3, (0, 45))
run_experiment("F (Core)", 3, (15, 35))

# Run for S (APC, Class 1)
run_experiment("S", 1, (0, 45))
run_experiment("S (Core)", 1, (25, 45))
