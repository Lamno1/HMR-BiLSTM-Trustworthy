"""
diag_verify_coupling_v2.py
===========================
Hai kiểm bắt buộc trước khi viết bất kỳ câu coupling nào vào paper.

KIỂM 1 — Đối chứng N/V onset% (coupling có phân biệt được không?)
  Với MỖI class cn ∈ {N,S,V,F}:
    - Lấy danh sách idx_shared từ shared_idx_c[cn]
    - Tăng cỡ mẫu lên N_KIEM1 (mặc định 80) bằng cách lấy thêm các beat phân loại đúng khác của lớp đó
    - Tính SHAP của ĐÚNG head cls_id = CLS_MAP[cn]
    - Đếm % beat có peak |SHAP| trong onset [0,45]
  In n thô (k/N_KIEM1) + Wilson 95% CI cho mỗi tỉ lệ.

KIỂM 2 — Ratio |SHAP|/amplitude: beat-trước vs beat-hiện-tại (F và S)
  Chỉ tính trên beat có đỉnh thật ở cả hai vị trí (để công bằng):
    amp_before > 3 × baseline AND amp_cur > 3 × baseline
    (baseline = median signal tại t∈[55,80])
  So sánh ratio theo cặp trên từng beat: ratio_before vs ratio_cur.
  Báo median (không mean) + in phân phối đầy đủ từng beat.

Chạy:
    venv\\Scripts\\python.exe diag_verify_coupling_v2.py
"""

import sys
import io
import json
import math
import numpy as np
import torch
from pathlib import Path
import shap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, INTER_TEST, INTER_TRAIN

SHAP_RESULTS     = "outputs/v1.0_20260611_160039/explainability/results.json"
ONSET_ZONE       = (0, 45)      # vùng chứa R-peak beat trước
QRS_WIN          = (85, 97)     # cửa sổ nhỏ quanh R-peak hiện tại
BASELINE_ZONE    = (55, 80)     # PR segment: thường phẳng, dùng làm baseline
AMP_PEAK_THRESH  = 3.0          # amp > AMP_PEAK_THRESH × baseline để coi là đỉnh thật
N_BG             = 50
SEED             = 42
CLASSES          = ["N", "S", "V", "F"]
CLS_MAP          = {"N": 0, "S": 1, "V": 2, "F": 3}

# Cấu hình cỡ mẫu cho riêng KIỂM 1 để CI hẹp lại
N_KIEM1          = 80  # n=80 giúp khoảng Wilson CI hẹp lại rõ rệt so với n=30


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI cho tỉ lệ k/n. Trả (lo, hi) dạng phần trăm."""
    if n == 0:
        return (0.0, 0.0)
    p  = k / n
    d  = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    c  = 1 + z**2 / n
    lo = (p + z**2 / (2 * n) - d) / c
    hi = (p + z**2 / (2 * n) + d) / c
    return (max(0.0, lo), min(1.0, hi))


# ── Load shared indices ───────────────────────────────────────────────────────
print(f"Loading shared_idx_c from {SHAP_RESULTS}...")
with open(SHAP_RESULTS, "r", encoding="utf-8") as f:
    shap_res = json.load(f)
shared_idx_c = shap_res["metrics"]["shared_idx_c"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
test    = np.load(INTER_TEST)
X_test  = test["X"].astype(np.float32)
y_test  = test["y"].astype(np.int64)
train   = np.load(INTER_TRAIN)
X_train = train["X"].astype(np.float32)

# ── Model + SHAP explainer setup ──────────────────────────────────────────────
device = torch.device("cpu")
model, _ = load_hmr_bilstm(RLSTM_CKPT, device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)

wrapper = ModelWrapper(model).eval()

# Get all predictions
print("Getting predictions...")
all_preds = []
X_t_pred = torch.from_numpy(X_test)
with torch.no_grad():
    for i in range(0, len(X_t_pred), 256):
        b = X_t_pred[i:i+256].to(device)
        all_preds.append(wrapper(b).argmax(dim=-1).cpu().numpy())
preds_all = np.concatenate(all_preds)
print(f"  Accuracy: {(preds_all == y_test).mean():.4f}")

print("  Xác nhận beat + head mỗi lớp ban đầu (shared):")
for cn in CLASSES:
    print(f"    [{cn}] shared_idx_c['{cn}'] = {len(shared_idx_c[cn])} beats, "
          f"SHAP head = cls_id {CLS_MAP[cn]}  ← {cn} beats, {cn} head")

rng_bg = np.random.default_rng(SEED)
bg_idx = rng_bg.choice(len(X_train), N_BG, replace=False)
X_bg   = torch.from_numpy(X_train[bg_idx]).to(device)


def compute_shap_abs(cn, idx_list):
    """
    Tính |SHAP| trên danh sách indices idx_list của lớp cn, với ĐÚNG head cls_id.
    Trả sv_abs: (n_beats, 187).
    """
    cls_id    = CLS_MAP[cn]
    X_cls     = X_test[idx_list]
    X_t       = torch.from_numpy(X_cls).to(device)
    explainer = shap.GradientExplainer(wrapper, X_bg)
    sv_raw    = explainer.shap_values(X_t)
    sv_arr    = np.array(sv_raw)
    if sv_arr.ndim == 4:           # (n, 187, 1, 5)
        sv = sv_arr[:, :, :, cls_id]
    elif sv_arr.ndim == 5:         # (5, n, 187, 1)
        sv = sv_arr[cls_id]
    else:
        raise ValueError(f"Unexpected sv_arr shape: {sv_arr.shape}")
    return np.abs(sv).squeeze(-1)  # (n, 187)


# ════════════════════════════════════════════════════════════════════════════
#  KIỂM 1: Onset% per class — đối chứng 4 lớp
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f" KIỂM 1: Onset-zone [0,45] % per class — đối chứng N/V/S/F (n={N_KIEM1})")
print("=" * 72)

k1_results = {}   # cn -> {"k": int, "n": int, "pct": float, "ci": (lo, hi), "peak_ts": list}

for cn in CLASSES:
    idx_shared = shared_idx_c[cn]
    cls_id = CLS_MAP[cn]
    
    # Lấy tất cả các beat phân loại đúng của lớp này
    correct_all = np.where((y_test == cls_id) & (preds_all == cls_id))[0].tolist()
    
    # Tạo list indices cho KIỂM 1:
    # 1. Bắt đầu bằng 30 shared beats để đồng bộ
    idx_kiem1 = list(idx_shared)
    
    # 2. Bổ sung thêm các beat đúng khác cho đủ N_KIEM1
    remaining = [i for i in correct_all if i not in idx_kiem1]
    n_needed = N_KIEM1 - len(idx_kiem1)
    if n_needed > 0 and len(remaining) > 0:
        rng_k1 = np.random.default_rng(SEED)
        extra = rng_k1.choice(remaining, min(n_needed, len(remaining)), replace=False).tolist()
        idx_kiem1.extend(extra)
        
    n        = len(idx_kiem1)
    print(f"\n[{cn}] Computing SHAP — beats: {n} beats (prefix has {len(idx_shared)} shared beats), "
          f"head: cls_id={cls_id}")
    sv_abs   = compute_shap_abs(cn, idx_kiem1)           # (n, 187)
    peak_ts  = sv_abs.argmax(axis=1).tolist() # (n,)
    k_onset  = sum(ONSET_ZONE[0] <= t < ONSET_ZONE[1] for t in peak_ts)
    k_qrs    = sum(QRS_WIN[0]    <= t < QRS_WIN[1]    for t in peak_ts)
    k_tend   = sum(145           <= t < 187            for t in peak_ts)
    pct      = k_onset / n
    lo, hi   = wilson_ci(k_onset, n)
    k1_results[cn] = {"k": k_onset, "n": n, "pct": pct, "ci": (lo, hi), "peak_ts": peak_ts}
    print(f"  Peak timesteps: {peak_ts}")
    print(f"  Onset [0,45) :  {k_onset:2d}/{n}  = {pct:.1%}  "
          f"[Wilson 95% CI: {lo:.1%} – {hi:.1%}]")
    print(f"  QRS  [85,97) :  {k_qrs:2d}/{n}  = {k_qrs/n:.1%}")
    print(f"  T-end[145,187): {k_tend:2d}/{n}  = {k_tend/n:.1%}")

# Bảng tóm tắt thô
print("\n" + "-" * 72)
print(" BẢNG KIỂM 1 — SỐ THÔ (đọc trước verdict):")
print(f"  {'Lớp':>5} | {'onset k/n':>10} | {'onset%':>7} | {'Wilson 95% CI':>20}")
print("  " + "-" * 55)
for cn in CLASSES:
    r   = k1_results[cn]
    bar = "█" * int(r["pct"] * 20)
    print(f"  {cn:>5} | {r['k']:>2}/{r['n']:<2}       | "
          f"  {r['pct']:.1%}   | [{r['ci'][0]:.1%} – {r['ci'][1]:.1%}]  {bar}")

# Delta F–N, F–V
f_pct  = k1_results["F"]["pct"]; f_k = k1_results["F"]["k"]; f_n = k1_results["F"]["n"]
n_pct  = k1_results["N"]["pct"]; n_k = k1_results["N"]["k"]; n_n = k1_results["N"]["n"]
v_pct  = k1_results["V"]["pct"]; v_k = k1_results["V"]["k"]; v_n = k1_results["V"]["n"]
s_pct  = k1_results["S"]["pct"]; s_k = k1_results["S"]["k"]; s_n = k1_results["S"]["n"]
delta_fn = f_pct - n_pct
delta_fv = f_pct - v_pct

print(f"\n  Δ(F−N) = {f_k}/{f_n} − {n_k}/{n_n} = {delta_fn:+.1%}")
print(f"  Δ(F−V) = {f_k}/{f_n} − {v_k}/{v_n} = {delta_fv:+.1%}")

print("\n── GỢI Ý VERDICT KIỂM 1 (đọc số thô trước, đây chỉ là gợi ý) ──")
if delta_fn >= 0.25 and delta_fv >= 0.25:
    k1_verdict = "COUPLING CÓ THỂ PHÂN BIỆT"
    print(f"  → Δ(F−N)={delta_fn:+.1%} và Δ(F−V)={delta_fv:+.1%} ≥ 25%.")
    print("    GỢI Ý: onset% của F cao hơn rõ so với N/V.")
    print("    Câu 'consistent with coupling interval' có thể dùng, kèm caveat.")
elif delta_fn >= 0.10:
    k1_verdict = "COUPLING BIÊN YẾU"
    print(f"  → Δ(F−N)={delta_fn:+.1%} (10–25%). GỢI Ý: biên nhỏ, n={N_KIEM1} dễ nhiễu.")
    print("    Không nói 'coupling'. Nếu viết, dùng 'marginal tendency'.")
else:
    k1_verdict = "COUPLING KHÔNG PHÂN BIỆT"
    print(f"  → Δ(F−N)={delta_fn:+.1%} < 10%. GỢI Ý: N/V cũng onset% cao tương đương F.")
    print("    Không viết coupling. Viết limitation amplitude-dominated SHAP.")

print("  *** Bạn đọc số thô ở bảng trên rồi tự quyết — ngưỡng này tùy ý. ***")


# ════════════════════════════════════════════════════════════════════════════
#  KIỂM 2: |SHAP|/amplitude ratio — R-peak-trước vs R-peak-hiện-tại
#  Chỉ cho F và S, trên 30 shared beats gốc, và lọc đỉnh thật ở cả hai phía
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" KIỂM 2: |SHAP|/amplitude — trước vs hiện tại (F và S)")
print(f"  Filter: chỉ tính beat có amp_before và amp_cur > {AMP_PEAK_THRESH}× baseline [{BASELINE_ZONE}]")
print("  Metric: MEDIAN ratio (không mean) — so sánh theo cặp trên từng beat")
print("=" * 72)

def run_ratio_test(cn):
    idx_list  = shared_idx_c[cn]
    sv_abs    = compute_shap_abs(cn, idx_list)
    signal_2d = X_test[idx_list].squeeze(-1)     # (n, 187)
    peak_ts   = sv_abs.argmax(axis=1).tolist()

    records   = []
    skipped   = []

    for i, (beat_global, t_peak) in enumerate(zip(idx_list, peak_ts)):
        sig_row  = signal_2d[i]     # (187,)
        shap_row = sv_abs[i]        # (187,)

        # Baseline: median signal trong vùng PR [55,80]
        baseline_median = float(np.median(np.abs(sig_row[BASELINE_ZONE[0]:BASELINE_ZONE[1]])))
        baseline_median = max(baseline_median, 1e-4)   # tránh chia 0

        # R-peak-trước: argmax biên độ trong onset [0,45]
        onset_sig = sig_row[ONSET_ZONE[0]:ONSET_ZONE[1]]
        t_before  = int(np.argmax(onset_sig)) + ONSET_ZONE[0]
        amp_before = float(sig_row[t_before])
        amp_before_abs = abs(amp_before)

        # R-peak-hiện-tại: argmax trong QRS_WIN [85,97]
        qrs_sig  = sig_row[QRS_WIN[0]:QRS_WIN[1]]
        t_cur    = int(np.argmax(qrs_sig)) + QRS_WIN[0]
        amp_cur  = float(sig_row[t_cur])
        amp_cur_abs = abs(amp_cur)

        # Lọc đỉnh thật cho cả 2 phía để đảm bảo công bằng
        if amp_before_abs < AMP_PEAK_THRESH * baseline_median:
            skipped.append({
                "beat": beat_global, "t_peak": t_peak,
                "t_before": t_before, "amp_before": amp_before,
                "t_cur": t_cur, "amp_cur": amp_cur,
                "baseline_median": baseline_median,
                "reason": f"amp_before {amp_before:.3f} < {AMP_PEAK_THRESH}× baseline {baseline_median:.3f}"
            })
            continue

        if amp_cur_abs < AMP_PEAK_THRESH * baseline_median:
            skipped.append({
                "beat": beat_global, "t_peak": t_peak,
                "t_before": t_before, "amp_before": amp_before,
                "t_cur": t_cur, "amp_cur": amp_cur,
                "baseline_median": baseline_median,
                "reason": f"amp_cur {amp_cur:.3f} < {AMP_PEAK_THRESH}× baseline {baseline_median:.3f}"
            })
            continue

        shap_before = float(shap_row[t_before])
        shap_cur    = float(shap_row[t_cur])

        ratio_before  = shap_before  / amp_before_abs
        ratio_current = shap_cur     / max(amp_cur_abs, 1e-6)

        records.append({
            "beat": beat_global, "t_peak": t_peak,
            "t_before": t_before, "amp_before": amp_before,
            "shap_before": shap_before, "ratio_before": ratio_before,
            "t_cur": t_cur, "amp_cur": amp_cur,
            "shap_cur": shap_cur, "ratio_cur": ratio_current,
            "baseline_median": baseline_median,
            "before_exceeds_cur": ratio_before > ratio_current,
        })

    return records, skipped

for cn in ["F", "S"]:
    print(f"\n── Class {cn} ──")
    print(f"  Computing SHAP for ratio test (head={CLS_MAP[cn]}, beats=shared_idx_c['{cn}'])...")
    records, skipped = run_ratio_test(cn)

    print(f"\n  Skipped (không thỏa mãn bộ lọc đỉnh ở cả 2 phía): {len(skipped)}/{len(shared_idx_c[cn])} beat")
    for s in skipped:
        print(f"    beat={s['beat']} t_peak={s['t_peak']} amp_before={s['amp_before']:.3f} amp_cur={s['amp_cur']:.3f} "
              f"[{s['reason']}]")

    n_rec = len(records)
    print(f"\n  Hợp lệ cho KIỂM 2 (thông qua bộ lọc đỉnh): {n_rec} beat")
    if n_rec == 0:
        print("  Không đủ beat → không tính ratio.")
        continue

    # Print bảng đầy đủ
    print(f"\n  {'Beat':>7} | {'t_bef':>6} | {'amp_bef':>8} | {'shap_bef':>10} | "
          f"{'r_before':>9} | {'r_cur':>8} | {'bef>cur?':>9}")
    print("  " + "-" * 72)
    for r in records:
        flag = "✓ YES" if r["before_exceeds_cur"] else "  NO"
        print(f"  {r['beat']:>7} | {r['t_before']:>6} | {r['amp_before']:>8.3f} | "
              f"{r['shap_before']:>10.5f} | {r['ratio_before']:>9.5f} | "
              f"{r['ratio_cur']:>8.5f} | {flag:>9}")

    # Phân phối ratio_before và ratio_cur
    rb_arr = np.array([r["ratio_before"]  for r in records])
    rc_arr = np.array([r["ratio_cur"]     for r in records])
    n_exc  = sum(r["before_exceeds_cur"] for r in records)

    print(f"\n  PHÂN PHỐI ratio_before (n={n_rec}):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct:2d}: {np.percentile(rb_arr, pct):.5f}")

    print(f"\n  PHÂN PHỐI ratio_cur (n={n_rec}):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct:2d}: {np.percentile(rc_arr, pct):.5f}")

    med_rb = float(np.median(rb_arr))
    med_rc = float(np.median(rc_arr))
    print(f"\n  MEDIAN ratio_before  = {med_rb:.5f}")
    print(f"  MEDIAN ratio_cur     = {med_rc:.5f}")
    print(f"  Ratio median_before / median_cur = {med_rb / max(med_rc, 1e-9):.3f}x")
    print(f"  beat có ratio_before > ratio_cur: {n_exc}/{n_rec}")

    print(f"\n  ── GỢI Ý VERDICT KIỂM 2 [{cn}] (đọc phân phối trước) ──")
    if n_exc >= n_rec * 0.6 and med_rb > med_rc * 1.2:
        print(f"  → median_before {med_rb:.4f} > {med_rc:.4f} = median_cur ({med_rb/med_rc:.2f}x)")
        print("    GỢI Ý: Model gán trọng số vượt biên độ cho R-peak-trước.")
        print("    Có thể ủng hộ coupling — nhưng đọc phân phối để xem outlier.")
    elif med_rb > med_rc:
        print(f"  → Marginal: median_before ({med_rb:.4f}) > median_cur ({med_rc:.4f}) nhưng biên nhỏ.")
        print("    Không đủ để xác nhận coupling.")
    else:
        print(f"  → median_before ({med_rb:.4f}) ≤ median_cur ({med_rc:.4f}).")
        print("    GỢI Ý: SHAP chỉ track biên độ. Coupling không vững.")
    print("  *** Đọc phân phối đầy đủ và n_exc/n_rec trước khi kết luận. ***")


# ════════════════════════════════════════════════════════════════════════════
#  TÓM TẮT CÁC SỐ THÔ ĐỂ GỬI ĐI ĐỌC
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" TÓM TẮT SỐ THÔ (gửi toàn bộ phần này để đọc)")
print("=" * 72)
print(f"\nKIỂM 1 — onset% với Wilson 95% CI (n={N_KIEM1}):")
for cn in CLASSES:
    r = k1_results[cn]
    print(f"  {cn}: {r['k']}/{r['n']} = {r['pct']:.1%}  [95% CI: {r['ci'][0]:.1%}–{r['ci'][1]:.1%}]")
print(f"\n  Δ(F−N) = {delta_fn:+.1%}   Δ(F−V) = {delta_fv:+.1%}")
print(f"\nKIỂM 1 GỢI Ý: {k1_verdict}")
print("\nKIỂM 2 — xem bảng phân phối ở trên cho F và S.")
print("\n[diag_verify_coupling_v2.py] Done.")
