"""
T3 — Integrated Gradients for Trustworthy ECG Classification (captum).

Calculates Integrated Gradients and compares with SHAP on shared beats.
Now updated with:
1. Multi-beat onset% + Wilson CI (n=80) for IG to compare directly with SHAP.
2. Self-healing SHAP directory lookup to prevent run ID mismatch.
"""

import json
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
from report_results import load_hmr_bilstm

CLASS_NAMES   = {0: "N (Normal)", 1: "S (APC)", 2: "V (PVC)", 3: "F (Fusion)"}
PLOT_CLASSES  = {0: "ig_normal.png", 1: "ig_apc.png", 2: "ig_pvc.png", 3: "ig_fusion.png"}
CLS_MAP       = {"N": 0, "S": 1, "V": 2, "F": 3}
CLASSES       = ["N", "S", "V", "F"]

ONSET_ZONE    = (0, 45)
QRS_WIN       = (85, 97)
N_KIEM1       = 80
SEED          = 42

# Approximate ECG region boundaries for MIT-BIH 187-sample beat
ECG_REGIONS = {
    "P-wave":  (10,  40),
    "QRS":     (60,  100),
    "T-wave":  (110, 155),
}


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


def plot_ig(signal, attribution, cls_name, save_path, title):
    """Overlay integrated gradients on ECG signal with anatomical region annotations."""
    T   = len(signal)
    t   = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(t, signal, color='#1565C0', linewidth=1.5, label='ECG signal', zorder=3)

    # Region annotations
    colors_region = {'P-wave': '#FFF9C4', 'QRS': '#FFCCBC', 'T-wave': '#C8E6C9'}
    for region, (s, e) in ECG_REGIONS.items():
        ax1.axvspan(s, e, alpha=0.35, color=colors_region[region], label=region, zorder=1)

    # IG overlay as coloured fill
    pos = np.where(attribution >= 0, attribution, 0)
    neg = np.where(attribution < 0, np.abs(attribution), 0)
    ax1.fill_between(t, signal - pos * 0.5, signal + pos * 0.5,
                     alpha=0.5, color='#D32F2F', label='Positive IG', zorder=2)
    ax1.fill_between(t, signal - neg * 0.5, signal + neg * 0.5,
                     alpha=0.5, color='#388E3C', label='Negative IG', zorder=2)

    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.grid(alpha=0.25)
    ax1.set_xlim([0, T - 1])

    # Bottom panel: attribution bar chart
    ax2 = axes[1]
    ax2.bar(t, attribution, color=np.where(attribution >= 0, '#D32F2F', '#388E3C'),
            width=0.9, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("Attribution", fontsize=10)
    ax2.set_xlim([0, T - 1])
    ax2.grid(alpha=0.2, linestyle='--', axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {save_path}")


def jaccard_with_tolerance(set_a, set_b, tolerance=2, T=187):
    """Symmetric Jaccard index with a temporal tolerance window.

    Uses greedy bijective matching so each element of set_b is consumed
    by at most one element of set_a (matched to the nearest within tolerance).
    This yields a true intersection cardinality and avoids double-counting
    when one element in B is within tolerance of multiple elements in A.

    Example: A={90,91,92}, B={90}, tol=2
      matched=1, union=3+1-1=3, Jaccard=1/3  (was wrongly 1.0 before)
    """
    set_a_sorted = sorted(set_a)
    set_b_remaining = sorted(set_b)
    matched = 0
    for a in set_a_sorted:
        candidates = [b for b in set_b_remaining if abs(a - b) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda b: abs(a - b))
            matched += 1
            set_b_remaining.remove(best)
    union = len(set_a) + len(set_b) - matched
    return matched / max(1, union)


def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)

    # ── Self-healing SHAP directory lookup ──
    shap_results_path = paths["out_explain"] / "results.json"
    output_dir = paths["out_explain"]

    if not shap_results_path.exists():
        print(f"  Warning: results.json not found at {shap_results_path}")
        print("  Searching for the latest SHAP results directory in outputs/...")
        candidate_paths = sorted(
            Path("outputs").glob("*/explainability/results.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidate_paths:
            shap_results_path = candidate_paths[0]
            output_dir = shap_results_path.parent
            print(f"  Found latest SHAP results at: {shap_results_path}")
            print(f"  Updating IG output directory to: {output_dir}")
        else:
            print("  No SHAP results found anywhere in outputs/.")

    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Output Directory: {output_dir}")

    # Load model
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # Load test data & get predictions
    print("Loading test data...")
    print(f"  Using: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    all_preds = []
    X_t = torch.from_numpy(X_test)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256].to(device)
            all_preds.append(model(b).argmax(dim=-1).cpu().numpy())
    preds_all = np.concatenate(all_preds)

    # ── Load shared_idx_c from SHAP results.json ──
    shared_idx_c: dict = {}     # cls_name (str) -> list[int]
    shap_shared_top10: dict = {}
    if shap_results_path.exists():
        with open(shap_results_path, "r", encoding="utf-8") as f:
            shap_results = json.load(f)
        shared_idx_c   = shap_results.get("metrics", {}).get("shared_idx_c",    {})
        shap_shared_top10 = shap_results.get("metrics", {}).get("shap_shared_top10", {})
        if shared_idx_c:
            print(f"  Loaded shared_idx_c from SHAP results: "
                  f"{ {cn: len(v) for cn, v in shared_idx_c.items()} }")
        else:
            print("  Warning: shared_idx_c not found in SHAP results — falling back to empty.")
    else:
        print("  Warning: SHAP results.json not found — Jaccard analysis will be skipped.")

    # — Integrated Gradients —
    ig = IntegratedGradients(model)
    n_steps  = 200
    ig_stats = {}
    k1_results = {}
    consistency = {}

    for cls, fname in PLOT_CLASSES.items():
        cls_label = CLASS_NAMES[cls]
        cn = cls_label.split()[0]   # e.g. "N", "S", "V", "F"
        print(f"\nComputing IG for class {cls_label}...")

        # — Sample selection for 80 beats (similar to diag_verify_coupling_v2.py) —
        idx_shared = shared_idx_c.get(cn, [])
        correct_all = np.where((y_test == cls) & (preds_all == cls))[0].tolist()

        # Build N_KIEM1 beats index list
        idx_kiem1 = list(idx_shared)
        remaining = [i for i in correct_all if i not in idx_kiem1]
        n_needed = N_KIEM1 - len(idx_kiem1)
        if n_needed > 0 and len(remaining) > 0:
            rng_k1 = np.random.default_rng(SEED)
            extra = rng_k1.choice(remaining, min(n_needed, len(remaining)), replace=False).tolist()
            idx_kiem1.extend(extra)

        n_beats = len(idx_kiem1)
        if n_beats == 0:
            print(f"  Warning: no correctly predicted beats found for {cls_label}, skipping.")
            continue

        print(f"  Running IG on {n_beats} beats...")
        all_attrs = []
        deltas    = []
        for idx in idx_kiem1:
            x        = torch.from_numpy(X_test[idx:idx+1]).to(device)  # (1, T, 1)
            baseline = torch.zeros_like(x)
            attrs_i, delta_i = ig.attribute(
                x, baseline, target=cls,
                n_steps=n_steps, return_convergence_delta=True
            )
            attr_np = attrs_i.squeeze().cpu().detach().numpy()
            if attr_np.ndim > 1:
                attr_np = attr_np.squeeze(-1)
            all_attrs.append(attr_np)
            deltas.append(float(delta_i.item()))

        # KIỂM 1: Onset-zone [0, 45) percentage and Wilson CI
        peak_ts  = [int(np.abs(attr).argmax()) for attr in all_attrs]
        k_onset  = sum(ONSET_ZONE[0] <= t < ONSET_ZONE[1] for t in peak_ts)
        k_qrs    = sum(QRS_WIN[0]    <= t < QRS_WIN[1]    for t in peak_ts)
        k_tend   = sum(145           <= t < 187            for t in peak_ts)
        pct      = k_onset / n_beats
        lo, hi   = wilson_ci(k_onset, n_beats)

        k1_results[cn] = {
            "k": k_onset,
            "n": n_beats,
            "pct": pct,
            "ci": (lo, hi),
            "peak_ts": peak_ts
        }

        print(f"  Onset [0,45) :  {k_onset:2d}/{n_beats}  = {pct:.1%}  [Wilson 95% CI: {lo:.1%} – {hi:.1%}]")
        print(f"  QRS  [85,97) :  {k_qrs:2d}/{n_beats}  = {k_qrs/n_beats:.1%}")
        print(f"  T-end[145,187): {k_tend:2d}/{n_beats}  = {k_tend/n_beats:.1%}")

        # Top-10 over the shared beats (first len(idx_shared) elements of all_attrs)
        n_shared = len(idx_shared)
        if n_shared > 0:
            shared_attrs = all_attrs[:n_shared]
            attribution_mean_shared = np.mean(np.abs(shared_attrs), axis=0)
            top10_shared = np.argsort(attribution_mean_shared)[::-1][:10].tolist()
            for idx in top10_shared:
                assert 0 <= idx < 187, f"[INDEX CHECK FAIL] {cn} IG top10 index {idx} out of bounds (T=187)"
        else:
            top10_shared = []

        # Jaccard consistency calculation
        shap_top10_cn = shap_shared_top10.get(cn, [])
        jaccard_val = -1.0
        if n_shared > 0 and shap_top10_cn:
            jaccard_val = jaccard_with_tolerance(set(shap_top10_cn), set(top10_shared), tolerance=2)
            consistency[cn] = {
                "jaccard_shared_basis": float(jaccard_val),
                "shap_top10": shap_top10_cn,
                "ig_top10": top10_shared,
                "n_beats_shared": n_shared
            }
            print(f"  Jaccard(SHAP_shared, IG_shared) [{cn}]: {jaccard_val:.4f} (n_beats={n_shared})")

        # — Plot with first representative beat —
        repr_idx   = idx_kiem1[0]
        signal     = X_test[repr_idx].squeeze()
        pred_lbl   = CLASS_NAMES.get(int(preds_all[repr_idx]), str(preds_all[repr_idx]))
        true_lbl   = CLASS_NAMES[cls]
        mean_delta = float(np.mean(deltas))
        title = (f"Integrated Gradients — {cls_label}  (mean |IG| over {n_beats} beats)\n"
                 f"Representative beat — True: {true_lbl}  |  Predicted: {pred_lbl}  |  "
                 f"Mean conv. delta: {mean_delta:.4f}")
        
        # Plot utilizing the mean absolute attribution of the KIỂM 1 beats
        mean_abs_attribution_plot = np.mean(np.abs(all_attrs), axis=0)
        plot_ig(signal, mean_abs_attribution_plot, cls_label, output_dir / fname, title)

        ig_stats[cn] = {
            "top10_timesteps_ig_shared": top10_shared,
            "n_beats_kiem1": n_beats,
            "k_onset": k_onset,
            "onset_pct": pct,
            "wilson_ci": (lo, hi),
            "mean_convergence_delta": mean_delta,
            "max_mean_abs_attr": float(np.mean(np.abs(all_attrs), axis=0).max()),
        }

    # — Save IG-specific JSON —
    ig_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "explainability_ig",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "ig_stats": ig_stats,
            "k1_results": k1_results,
            "consistency": consistency
        }
    }
    ig_json_path = output_dir / "ig_results.json"
    with open(ig_json_path, "w", encoding="utf-8") as f:
        json.dump(ig_json, f, indent=2)
    print(f"\n  [OK] ig_results.json")

    if consistency:
        consistency_path = output_dir / "shap_ig_consistency.json"
        with open(consistency_path, "w", encoding="utf-8") as f:
            json.dump({
                "jaccard_shared_basis": consistency,
                "methodology": (
                    "Both SHAP and IG top-10 computed by averaging |attribution| "
                    "over the SAME up-to-30 correctly-classified beats per class. "
                    "jaccard_with_tolerance(tolerance=2) applied."
                )
            }, f, indent=2)
        print(f"  [OK] shap_ig_consistency.json (shared-basis Jaccard)")

    # ════════════════════════════════════════════════════════════════════════════
    #  TÓM TẮT SỐ THÔ CHO IG
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print(" TÓM TẮT SỐ THÔ CHO INTEGRATED GRADIENTS (gửi toàn bộ phần này để đọc)")
    print("=" * 72)
    print(f"\nKIỂM 1 (IG) — onset% với Wilson 95% CI (n={N_KIEM1}):")
    for cn in CLASSES:
        r = k1_results.get(cn, {"k": 0, "n": 0, "pct": 0.0, "ci": (0.0, 0.0)})
        print(f"  {cn}: {r['k']}/{r['n']} = {r['pct']:.1%}  [95% CI: {r['ci'][0]:.1%}–{r['ci'][1]:.1%}]")
        
    f_pct = k1_results.get("F", {}).get("pct", 0.0)
    n_pct = k1_results.get("N", {}).get("pct", 0.0)
    v_pct = k1_results.get("V", {}).get("pct", 0.0)
    delta_fn = f_pct - n_pct
    delta_fv = f_pct - v_pct
    print(f"\n  Δ(F−N) = {delta_fn:+.1%}   Δ(F−V) = {delta_fv:+.1%}")
    
    if consistency:
        print("\nCONSISTENCY JACCARD (SHAP vs IG) TRÊN 30 SHARED BEATS:")
        for cn, c_data in consistency.items():
            print(f"  {cn}: Jaccard = {c_data['jaccard_shared_basis']:.4f} (SHAP={c_data['shap_top10'][:5]}..., IG={c_data['ig_top10'][:5]}...)")

    print("\n[integrated_gradients.py] Done.")


if __name__ == "__main__":
    main()
