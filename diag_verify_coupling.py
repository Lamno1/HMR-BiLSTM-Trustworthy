"""
diag_verify_coupling.py
========================
Kiểm tra rẻ để khóa finding "R-peak beat trước":

Với mỗi beat trong 30 F shared:
  1. Phát hiện đỉnh tín hiệu thực trong vùng onset [0, 45] bằng argmax biên độ
  2. So sánh với peak SHAP|F| mà diag_auc_shap_f.py đã tính
  3. Báo: khớp (|shap_peak - signal_peak| <= 3) hay không

Nếu đa số beat khớp → finding "model attend vào R-peak beat trước" được xác nhận
bằng đối chiếu trực tiếp tín hiệu, không chỉ suy luận từ arithmetic.

Chạy: venv\\Scripts\\python.exe diag_verify_coupling.py
"""

import sys
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import shap

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, INTER_TEST, INTER_TRAIN

SHAP_RESULTS = "outputs/v1.0_20260611_160039/explainability/results.json"
F_CLASS      = 3
ONSET_ZONE   = (0, 45)        # vùng đầu cửa sổ tìm R-peak beat trước
MATCH_TOL    = 3               # |shap_peak - signal_peak| <= 3 là khớp
N_BG         = 50
SEED         = 42

# ── Load shared F indices ─────────────────────────────────────────────────────
print(f"Loading shared_idx_c from {SHAP_RESULTS}...")
with open(SHAP_RESULTS, "r", encoding="utf-8") as f:
    shap_res = json.load(f)
f_shared_idx = shap_res["metrics"]["shared_idx_c"]["F"]   # 30 indices
print(f"  Loaded {len(f_shared_idx)} F beats: {f_shared_idx[:5]}...")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading test data...")
test   = np.load(INTER_TEST)
X_test = test["X"].astype(np.float32)

print("Loading train data (for SHAP background)...")
train   = np.load(INTER_TRAIN)
X_train = train["X"].astype(np.float32)

X_f = X_test[f_shared_idx]   # (30, 187, 1)

# ── Tính SHAP cho class F trên 30 beat ───────────────────────────────────────
device = torch.device("cpu")
model, _ = load_hmr_bilstm(RLSTM_CKPT, device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)
wrapper = ModelWrapper(model).eval()

rng_bg = np.random.default_rng(SEED)
bg_idx = rng_bg.choice(len(X_train), N_BG, replace=False)
X_bg   = torch.from_numpy(X_train[bg_idx]).to(device)

print(f"Computing SHAP for class F on {len(f_shared_idx)} beats (bg={N_BG})...")
X_f_t    = torch.from_numpy(X_f).to(device)
explainer = shap.GradientExplainer(wrapper, X_bg)
sv_raw    = explainer.shap_values(X_f_t)
sv_arr    = np.array(sv_raw)
print(f"  sv_arr shape: {sv_arr.shape}")

if sv_arr.ndim == 4:      # (30, 187, 1, 5)
    sv_f = sv_arr[:, :, :, F_CLASS]
elif sv_arr.ndim == 5:    # (5, 30, 187, 1)
    sv_f = sv_arr[F_CLASS]
else:
    raise ValueError(f"Unexpected sv_arr shape: {sv_arr.shape}")

sv_f_abs  = np.abs(sv_f).squeeze(-1)    # (30, 187)
shap_peak = sv_f_abs.argmax(axis=1)     # (30,) — peak SHAP timestep mỗi beat

# ── Phát hiện R-peak beat trước bằng argmax tín hiệu thực trong [0,45] ───────
# Tín hiệu: X_f shape (30, 187, 1) → squeeze thành (30, 187)
signal_2d   = X_f.squeeze(-1)   # (30, 187)
onset_slice = signal_2d[:, ONSET_ZONE[0]:ONSET_ZONE[1]]   # (30, 45)
signal_peak = onset_slice.argmax(axis=1) + ONSET_ZONE[0]  # (30,) — tuyệt đối trong [0,187)

# ── Đối chiếu ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(" ĐỒNG THUẬN SHAP-peak vs Signal-peak (beat F, vùng onset [0,45])")
print("=" * 68)
print(f"{'Beat':>5} | {'SHAP_peak':>10} | {'Sig_peak':>9} | {'|diff|':>7} | Match?")
print("-" * 50)

n_match   = 0
n_onset_shap = 0
match_records = []
for i in range(len(f_shared_idx)):
    sp = int(shap_peak[i])
    sgp = int(signal_peak[i])
    in_onset = ONSET_ZONE[0] <= sp < ONSET_ZONE[1]
    diff     = abs(sp - sgp)
    matched  = in_onset and diff <= MATCH_TOL
    if in_onset:
        n_onset_shap += 1
    if matched:
        n_match += 1
    match_records.append({
        "beat_idx": f_shared_idx[i], "shap_peak": sp, "sig_peak": sgp,
        "in_onset": in_onset, "diff": diff, "match": matched
    })
    flag = "✓" if matched else ("(onset,no match)" if in_onset else "—QRS/T—")
    print(f"{i+1:>5} | {sp:>10} | {sgp:>9} | {diff:>7} | {flag}")

print("-" * 50)
n = len(f_shared_idx)
print(f"\nSHAP peak nằm trong onset zone [0,45): {n_onset_shap}/{n}")
print(f"Trong số đó, khớp signal peak (|diff|≤{MATCH_TOL}): {n_match}/{n_onset_shap}")
print()

if n_match >= n_onset_shap * 0.75 and n_match >= 8:
    print("✓ VERDICT: SHAP peak KHỚP với R-peak beat trước trong tín hiệu thực.")
    print("  Finding 'model attend vào R-peak beat trước (coupling interval)' được")
    print("  xác nhận bằng đối chiếu trực tiếp, không chỉ suy luận arithmetic.")
elif n_match >= 3:
    print("△ VERDICT: Một phần khớp. Cần thêm phân tích — tín hiệu onset có thể")
    print("  không phải lúc nào cũng có đỉnh rõ trong [0,45].")
else:
    print("✗ VERDICT: Không khớp tốt. Có thể SHAP peak không track R-peak trước.")
    print("  Cần xem lại diễn giải 'R-peak beat trước'.")

print("\n── Amplitude thực tại peak SHAP vs. baseline (để xác nhận là đỉnh nhọn) ──")
for rec in match_records:
    i   = f_shared_idx.index(rec["beat_idx"])
    sp  = rec["shap_peak"]
    if ONSET_ZONE[0] <= sp < ONSET_ZONE[1]:
        amp = float(signal_2d[i, sp])
        baseline = float(signal_2d[i, 55:75].mean())   # vùng PR sau onset, thường phẳng
        ratio = amp / (abs(baseline) + 1e-6)
        print(f"  beat {f_shared_idx[i]} t={sp}: amp={amp:.3f}, baseline≈{baseline:.3f}, ratio={ratio:.1f}x")

print("\n[diag_verify_coupling.py] Done.")
