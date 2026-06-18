"""
diag_compare_shap_ig.py
======================
Chi tiết hóa sự khác biệt giữa SHAP và IG trên 30 shared beats của lớp F.
"""

import json
import numpy as np
import torch
from pathlib import Path
from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, INTER_TEST, INTER_TRAIN
import shap
from captum.attr import IntegratedGradients

# Load shared indices
SHAP_RESULTS = "outputs/v1.0_20260611_160039/explainability/results.json"
with open(SHAP_RESULTS, "r", encoding="utf-8") as f:
    shap_res = json.load(f)
shared_idx_c = shap_res["metrics"]["shared_idx_c"]
f_shared = shared_idx_c["F"]

# Load test data
test = np.load(INTER_TEST)
X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
model.eval()

# Model Wrapper for SHAP
class ModelWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)
wrapper = ModelWrapper(model).eval()

# Prepare background for SHAP
train = np.load(INTER_TRAIN)
X_train = train["X"].astype(np.float32)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train), 50, replace=False)
X_bg = torch.from_numpy(X_train[bg_idx]).to(torch.device("cpu")) # SHAP runs on CPU

# ── Compute SHAP on F shared beats ───────────────────────────────────────────
print("Computing SHAP on F shared beats (CPU)...")
X_f_t_cpu = torch.from_numpy(X_test[f_shared]).to(torch.device("cpu"))
explainer = shap.GradientExplainer(wrapper.to(torch.device("cpu")), X_bg)
sv_raw = explainer.shap_values(X_f_t_cpu)
sv_arr = np.array(sv_raw)
# Normalization logic
if sv_arr.ndim == 4:
    sv_f = sv_arr[:, :, :, 3]
elif sv_arr.ndim == 5:
    sv_f = sv_arr[3]
shap_f = np.abs(sv_f).squeeze(-1) # (30, 187)

# ── Compute IG on F shared beats ─────────────────────────────────────────────
print("Computing IG on F shared beats (CUDA/CPU)...")
model = model.to(device)
ig = IntegratedGradients(model)
ig_f = []
for idx in f_shared:
    x = torch.from_numpy(X_test[idx:idx+1]).to(device)
    baseline = torch.zeros_like(x)
    attr, _ = ig.attribute(x, baseline, target=3, n_steps=50, return_convergence_delta=True)
    ig_f.append(np.abs(attr.squeeze().cpu().detach().numpy()))
ig_f = np.array(ig_f) # (30, 187)

# Print beat-by-beat comparison
print("\n" + "=" * 80)
print(f"{'Beat ID':>8} | {'SHAP_onset_t':>12} | {'SHAP_qrs_t':>10} | {'SHAP_peak':>10} | {'IG_onset_t':>10} | {'IG_qrs_t':>10} | {'IG_peak':>10}")
print("-" * 80)

for i, beat_id in enumerate(f_shared):
    # Timesteps around onset [0, 45] and QRS [85, 97]
    shap_row = shap_f[i]
    ig_row = ig_f[i]
    
    t_onset_sig = np.arange(0, 45)
    t_qrs_sig = np.arange(85, 97)
    
    # Argmax within sub-regions
    t_onset_shap = int(t_onset_sig[np.argmax(shap_row[0:45])])
    t_qrs_shap = int(t_qrs_sig[np.argmax(shap_row[85:97])])
    val_onset_shap = shap_row[t_onset_shap]
    val_qrs_shap = shap_row[t_qrs_shap]
    peak_shap = int(np.argmax(shap_row))
    
    t_onset_ig = int(t_onset_sig[np.argmax(ig_row[0:45])])
    t_qrs_ig = int(t_qrs_sig[np.argmax(ig_row[85:97])])
    val_onset_ig = ig_row[t_onset_ig]
    val_qrs_ig = ig_row[t_qrs_ig]
    peak_ig = int(np.argmax(ig_row))
    
    print(f"{beat_id:>8d} | {t_onset_shap:>2d} ({val_onset_shap:.3f}) | {t_qrs_shap:>2d} ({val_qrs_shap:.3f}) | {peak_shap:>9d} | {t_onset_ig:>2d} ({val_onset_ig:.3f}) | {t_qrs_ig:>2d} ({val_qrs_ig:.3f}) | {peak_ig:>9d}")

print("=" * 80)
