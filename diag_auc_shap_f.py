"""
diag_auc_shap_f.py
===================
Hai việc trong một lượt chạy nhanh:

1. AUC per-class (S, V, F) — OvR, trên inter_test.
   Dùng roc_auc_score với softmax probs từ inter_best_rlstm.pt.

2. SHAP F edge-cluster frequency — phân tích 30 beat F shared.
   Cho mỗi beat F, tính argmax(|SHAP|) trên chiều thời gian,
   rồi đếm bao nhiêu beat có peak ở:
     - Cụm khởi đầu sớm (onset):   t ∈ [0, 40]
     - Cụm R-peak:                  t ∈ [80, 105]
     - Cụm T-wave end (rìa muộn):   t ∈ [145, 187)
   Mục đích: kiểm tra cụm rìa (t≈21, t≈159) có phải artefact 1–2 beat
   kéo lên top-10, hay phổ biến trên nhiều beat F.

Chạy:
    venv\\Scripts\\python.exe diag_auc_shap_f.py
"""

import sys
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
import shap

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, INTER_TEST, INTER_TRAIN

# ── Thông số ──────────────────────────────────────────────────────────────────
SHAP_RESULTS  = "outputs/v1.0_20260611_160039/explainability/results.json"
CLASS_NAMES   = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLINICAL_CLS  = [1, 2, 3]   # S, V, F
F_CLASS       = 3

# Cụm vùng thời gian (T=187, R-peak ≈ t=90)
ONSET_ZONE   = (0,   40)     # khởi đầu sớm / P-wave / onset ectopic
QRS_ZONE     = (80, 105)     # QRS phức bộ / R-peak
T_END_ZONE   = (145, 187)    # T-wave kết thúc / rìa muộn

# SHAP background samples cho lượt tính F
N_BG      = 50   # đủ để pilot, giữ nhanh
SEED      = 42

# ── Load model và dữ liệu ─────────────────────────────────────────────────────
device = torch.device("cpu")   # GradientExplainer cần CPU safer
print(f"Device: {device}")

print("Loading model...")
model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
model.eval()

class ModelWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)
wrapper = ModelWrapper(model).eval()

print("Loading test data...")
test  = np.load(INTER_TEST)
X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

print("Loading train data (for SHAP background)...")
train  = np.load(INTER_TRAIN)
X_train = train["X"].astype(np.float32)

# ════════════════════════════════════════════════════════════════════════════
# PHẦN 1: AUC per-class
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" PHẦN 1: AUC PER-CLASS (OvR) trên inter_test")
print("=" * 62)

print("Running inference on inter_test...")
all_probs = []
X_t = torch.from_numpy(X_test)
with torch.no_grad():
    for i in range(0, len(X_t), 256):
        b = X_t[i:i+256].to(device)
        logits = wrapper(b)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
all_probs = np.concatenate(all_probs)   # (N, 5)

print(f"  Predictions done. Shape: {all_probs.shape}")
print(f"  Accuracy: {(all_probs.argmax(1) == y_test).mean():.4f}")

print("\nPer-class AUC (OvR, macro over samples):")
for cls in CLINICAL_CLS:
    name = CLASS_NAMES[cls]
    n_cls = (y_test == cls).sum()
    try:
        auc = roc_auc_score((y_test == cls).astype(int), all_probs[:, cls])
        print(f"  AUC-{name}: {auc:.4f}  (n_positive={n_cls})")
    except Exception as e:
        print(f"  AUC-{name}: ERROR — {e}")

# Macro AUC tổng thể (5 lớp, loại Q nếu không có mẫu)
valid_cls = [c for c in range(5) if (y_test == c).sum() >= 2]
try:
    auc_macro = roc_auc_score(
        y_test, all_probs[:, valid_cls],
        multi_class="ovr", labels=valid_cls, average="macro"
    )
    print(f"\n  AUC macro (OvR, {len(valid_cls)} classes): {auc_macro:.4f}")
except Exception as e:
    print(f"  AUC macro ERROR: {e}")

# ════════════════════════════════════════════════════════════════════════════
# PHẦN 2: SHAP F edge-cluster frequency
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(" PHẦN 2: SHAP F — tần suất cụm rìa trên 30 beat shared")
print("=" * 62)

# Load shared F indices từ results.json
print(f"Loading shared_idx_c from {SHAP_RESULTS}...")
if not Path(SHAP_RESULTS).exists():
    print(f"  FILE NOT FOUND: {SHAP_RESULTS}")
    print("  Bỏ qua phần 2.")
    sys.exit(0)

with open(SHAP_RESULTS, "r", encoding="utf-8") as f:
    shap_res = json.load(f)

f_shared_idx = shap_res["metrics"]["shared_idx_c"]["F"]
print(f"  Loaded {len(f_shared_idx)} shared F beat indices: {f_shared_idx[:5]}...")

X_f = X_test[f_shared_idx]   # (30, 187, 1)
print(f"  X_f shape: {X_f.shape}")

# Background từ inter_train (seed cố định, giống T2)
rng_bg = np.random.default_rng(SEED)
bg_idx = rng_bg.choice(len(X_train), N_BG, replace=False)
X_bg   = torch.from_numpy(X_train[bg_idx]).to(device)

# Tính SHAP cho lớp F (class 3) trên 30 beat
print(f"Computing SHAP for class F (class 3) on {len(f_shared_idx)} beats...")
print(f"  Background: {N_BG} samples")
X_f_t = torch.from_numpy(X_f).to(device)
explainer = shap.GradientExplainer(wrapper, X_bg)
sv_raw    = explainer.shap_values(X_f_t)   # (30, 187, 1, 5) hoặc list

# Normalize output
sv_arr = np.array(sv_raw)
print(f"  sv_arr shape: {sv_arr.shape}")
if sv_arr.ndim == 4:
    # shape: (30, 187, 1, 5)
    sv_f = sv_arr[:, :, :, F_CLASS]   # (30, 187, 1)
elif sv_arr.ndim == 5:
    # shape: (5, 30, 187, 1)
    sv_f = sv_arr[F_CLASS]             # (30, 187, 1)
else:
    print(f"  UNEXPECTED shape: {sv_arr.shape}")
    sys.exit(1)

sv_f_abs = np.abs(sv_f).squeeze(-1)   # (30, 187)
peak_ts   = sv_f_abs.argmax(axis=1)   # (30,) — peak timestep mỗi beat

# Kiểm tra bounds
for i, t in enumerate(peak_ts):
    assert 0 <= t < 187, f"[FAIL] beat {i}: peak_t={t} out of bounds"

print(f"\nPer-beat peak timesteps (30 beats F shared):")
print(f"  {peak_ts.tolist()}")

# Phân vùng
onset_beats   = [i for i, t in enumerate(peak_ts) if ONSET_ZONE[0]  <= t < ONSET_ZONE[1]]
qrs_beats     = [i for i, t in enumerate(peak_ts) if QRS_ZONE[0]    <= t < QRS_ZONE[1]]
t_end_beats   = [i for i, t in enumerate(peak_ts) if T_END_ZONE[0]  <= t < T_END_ZONE[1]]
other_beats   = [i for i, t in enumerate(peak_ts)
                 if not (ONSET_ZONE[0] <= t < ONSET_ZONE[1])
                 and not (QRS_ZONE[0]  <= t < QRS_ZONE[1])
                 and not (T_END_ZONE[0] <= t < T_END_ZONE[1])]

n = len(peak_ts)
print(f"\nPhân vùng peak (T=187, R-peak≈t=90):")
print(f"  Cụm onset sớm t∈[0,40)   : {len(onset_beats):2d}/{n} = {len(onset_beats)/n:.0%}  "
      f"beats={[int(peak_ts[i]) for i in onset_beats]}")
print(f"  Cụm QRS/R-peak t∈[80,105): {len(qrs_beats):2d}/{n} = {len(qrs_beats)/n:.0%}  "
      f"beats={[int(peak_ts[i]) for i in qrs_beats]}")
print(f"  Cụm T-end rìa t∈[145,187): {len(t_end_beats):2d}/{n} = {len(t_end_beats)/n:.0%}  "
      f"beats={[int(peak_ts[i]) for i in t_end_beats]}")
print(f"  Vùng khác                : {len(other_beats):2d}/{n} = {len(other_beats)/n:.0%}  "
      f"beats={[int(peak_ts[i]) for i in other_beats]}")

print("\n── Phán quyết về cụm rìa ──")
if len(onset_beats) >= 3:
    print(f"  ✓ Cụm onset sớm (t≈21–24): PHỔ BIẾN ({len(onset_beats)}/{n} beat) — "
          f"finding thật, không phải artefact.")
else:
    print(f"  ✗ Cụm onset sớm (t≈21–24): YẾU ({len(onset_beats)}/{n} beat) — "
          f"khả năng cao là artefact 1–2 beat.")

if len(t_end_beats) >= 3:
    print(f"  ✓ Cụm T-end rìa (t≈158–161): PHỔ BIẾN ({len(t_end_beats)}/{n} beat) — "
          f"finding thật, không phải artefact.")
else:
    print(f"  ✗ Cụm T-end rìa (t≈158–161): YẾU ({len(t_end_beats)}/{n} beat) — "
          f"khả năng cao là artefact 1–2 beat.")

print("\n[diag_auc_shap_f.py] Done.")
