"""
T2 — SHAP GradientExplainer for Trustworthy ECG Classification.

Uses GradientExplainer (vs DeepExplainer) because HMR-BiLSTM contains
LayerNorm layers that DeepExplainer cannot handle correctly.
GradientExplainer works with any differentiable PyTorch function.

Classes: N (0), S/APC (1), V/PVC (2), F/Fusion (3)
Sample strategy (per class):
  - 10 correctly predicted samples
  - 5 misclassified samples  -> "Why does the model fail?"

Outputs:
  outputs/<run_id>/explainability/
    shap_summary_plot.png
    shap_class_N.png  / S / V / F
    shap_misclassified_N.png  / S / V / F
    shap_importance_ranking.csv
    results.json
"""

import json
import csv
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TRAIN, INTER_TEST
from report_results import load_hmr_bilstm


CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F"}
SHAP_CLASSES = [0, 1, 2, 3]


# ── Wrapper: logits only ─────────────────────────────────────────────────────
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# ── Sample selection ─────────────────────────────────────────────────────────
def select_samples(X, labels, preds, cls, n_correct=10, n_mis=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    tp  = np.where((labels == cls) & (preds == cls))[0]
    fn  = np.where((labels == cls) & (preds != cls))[0]
    nc  = min(n_correct, len(tp))
    nm  = min(n_mis,     len(fn))
    ci  = rng.choice(tp, nc, replace=False) if nc > 0 else np.array([], dtype=int)
    mi  = rng.choice(fn, nm, replace=False) if nm > 0 else np.array([], dtype=int)
    return ci, mi


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_shap_timeseries(shap_vals, X_samples, true_labels, pred_labels,
                         cls_name, title, save_path, max_plots=5):
    """Overlay SHAP attributions (for target class output) on ECG signal."""
    from matplotlib.patches import Patch
    n = min(len(X_samples), max_plots)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i in range(n):
        ax  = axes[i, 0]
        sig = X_samples[i].squeeze(-1)   # (T,1) → (T,): explicit channel axis
        sv  = shap_vals[i].squeeze(-1)   # (T,1) → (T,): explicit, safe for any batch size
        t   = np.arange(len(sig))
        
        # Plot ECG signal
        ax.plot(t, sig, color='#1565C0', linewidth=1.5, label='ECG', zorder=3)
        
        # Calculate limits and scale SHAP values
        ymin, ymax = float(sig.min() - 0.1), float(sig.max() + 0.1)
        ax.set_ylim(ymin, ymax)
        
        max_sv = np.max(np.abs(sv)) if np.max(np.abs(sv)) > 1e-9 else 1.0
        norm_sv = sv / max_sv
        
        # Draw opacity-scaled axvspan segments
        for j in range(len(t) - 1):
            val = float((norm_sv[j] + norm_sv[j+1]) / 2.0)
            alpha_val = float(min(0.4, abs(val) * 0.4))
            color = '#D32F2F' if val > 0 else '#388E3C'
            ax.axvspan(t[j], t[j+1], ymin=0.0, ymax=1.0, alpha=alpha_val, color=color, linewidth=0)
            
        tl = CLASS_NAMES.get(true_labels[i], str(true_labels[i]))
        pl = CLASS_NAMES.get(pred_labels[i],  str(pred_labels[i]))
        ax.set_title(f"Sample {i+1}  True: {tl}  Pred: {pl}", fontsize=9)
        ax.set_ylabel("Amp", fontsize=8)
        ax.grid(alpha=0.15)
        
        # Show legend on every subplot for clarity, or just the first one
        legend_elements = [
            plt.Line2D([0], [0], color='#1565C0', lw=1.5, label='ECG'),
            Patch(facecolor='#D32F2F', alpha=0.3, label='+SHAP (Supports Class)'),
            Patch(facecolor='#388E3C', alpha=0.3, label='-SHAP (Opposes Class)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)
        
    axes[-1, 0].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {save_path.name}")


def plot_shap_summary(mean_importance, T, out_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(T), mean_importance, color='#7B1FA2', alpha=0.8, width=0.9)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Mean |SHAP|", fontsize=12)
    ax.set_title("SHAP Feature Importance — All Classes (Mean |SHAP| per Timestep)",
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_plot.png", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] shap_summary_plot.png")


# Helper to normalize SHAP output shape
def normalize_shap_output(sv_raw, n_classes):
    """Normalize SHAP output to a list of length n_classes, each of shape (n_samples, T, 1).

    GradientExplainer returns ndarray of shape (n_samples, T, channel, n_classes)
    where classes are always on the LAST axis.
    Older SHAP versions return a list of (n_samples, T, channel) — one per class.
    """
    if isinstance(sv_raw, list):
        # Already a list of per-class arrays
        return sv_raw
    if isinstance(sv_raw, np.ndarray):
        ndim = sv_raw.ndim
        if ndim == 4:
            # Shape: (n_samples, T, channel, n_classes) → class axis is LAST
            # Do NOT use shape[0] here even if shape[0]==n_classes (that would slice samples)
            assert sv_raw.shape[-1] == n_classes, (
                f"Expected last dim = n_classes={n_classes}, got shape {sv_raw.shape}"
            )
            return [sv_raw[..., c] for c in range(n_classes)]
        elif ndim == 3 and sv_raw.shape[0] == n_classes:
            # Legacy format: (n_classes, n_samples, T) — class axis is FIRST
            return [sv_raw[c] for c in range(n_classes)]
        elif sv_raw.shape[-1] == n_classes:
            # Fallback: generic last-axis rule
            return [sv_raw[..., c] for c in range(n_classes)]
    return sv_raw


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_explain"].mkdir(parents=True, exist_ok=True)

    exp_cfg      = cfg["explainability"]
    n_background = exp_cfg.get("shap_background_samples", 200)
    n_correct    = exp_cfg.get("shap_correct_per_class",   10)
    n_mis        = exp_cfg.get("shap_misclassified_per_class", 5)

    # ── Axis sanity-check: verify normalize_shap_output gives correct class axis ──
    # Use a synthetic 4D array where class c has a known spike at a UNIQUE, NON-ZERO position.
    # BUG GUARD: spike at c*10 means class-0 spike is at t=0. argmax([0,0,...,0]) also returns 0
    # on wrong-axis arrays (all-zero), so class-0 would tình cờ pass even with wrong axis.
    # Fix: use (c+1)*10 — all spikes land at t≥10, argmax-on-zeros returns 0 ≠ any spike → fail.
    print("[AXIS CHECK] Verifying normalize_shap_output with synthetic 4D array...")
    n_cls_check = 5
    sv_synth = np.zeros((3, 187, 1, n_cls_check), dtype=np.float32)  # (samples, T, ch, classes)
    for c in range(n_cls_check):
        sv_synth[:, (c + 1) * 10, 0, c] = 1.0  # spike at t=(c+1)*10, non-zero, unique per class
    sv_norm = normalize_shap_output(sv_synth, n_cls_check)
    assert len(sv_norm) == n_cls_check, f"[AXIS CHECK FAIL] Expected list len={n_cls_check}, got {len(sv_norm)}"
    for c in range(n_cls_check):
        arr = sv_norm[c]              # should be (3, 187, 1)
        assert arr.shape == (3, 187, 1), f"[AXIS CHECK FAIL] Class {c} shape {arr.shape} != (3,187,1)"
        peak_t = int(np.abs(arr).squeeze(-1).mean(axis=0).argmax())
        expected_t = (c + 1) * 10
        assert 0 <= peak_t < 187, f"[AXIS CHECK FAIL] Class {c}: peak_t={peak_t} out of bounds (T=187)"
        assert peak_t == expected_t, (
            f"[AXIS CHECK FAIL] Class {c}: expected peak at t={expected_t}, got t={peak_t}. "
            f"WRONG AXIS — stop and debug normalize_shap_output."
        )
    print("[AXIS CHECK] PASSED — class axis confirmed correct for 4D ndarray (t=0 blind spot excluded).")
    shap_classes = exp_cfg.get("shap_classes", [0, 1, 2, 3])


    # CPU budget: cap background and summary to keep runtime < 10 min
    n_background = min(n_background, 100)   # GradientExplainer is heavier per-sample
    n_summary    = 200                       # samples for global importance
    # Number of background resamplings to average SHAP values.
    # GradientExplainer uses sampled baselines → attribution varies run-to-run.
    # Averaging over multiple resampled backgrounds stabilises Jaccard(SHAP, IG).
    n_shap_runs  = exp_cfg.get("shap_averaging_runs", 3)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")   # GradientExplainer requires grad on inputs → CPU safer
    print(f"Device: {device} | Run ID: {run_id}")

    # Load model
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    wrapper   = ModelWrapper(model).to(device)
    wrapper.eval()

    # Load test data (using centralized INTER_TEST path)
    print("Loading test data...")
    print(f"  Using: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    T_len  = X_test.shape[1]

    # Load training data for background (using centralized INTER_TRAIN path)
    print("Loading training data for background...")
    print(f"  Using: {INTER_TRAIN}")
    train = np.load(INTER_TRAIN)
    X_train = train["X"].astype(np.float32)

    # Get all predictions
    print("Getting predictions...")
    all_preds = []
    X_t = torch.from_numpy(X_test)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256].to(device)
            all_preds.append(wrapper(b).argmax(dim=-1).cpu().numpy())
    preds_all = np.concatenate(all_preds)
    print(f"  Accuracy: {(preds_all == y_test).mean():.4f}")

    # ── Clinical smoke test: verify SHAP attribution is clinically plausible ──
    # Synthetic axis-check confirms AXIS. This confirms MEANING using real V beats.
    # Beat format: 187 samples, R-peak at t≈90 (90 samples before, 97 after).
    # QRS complex: t≈80-100 | T wave: t≈100-145 → clinical "hot zone": t in [60, 150]
    # NOTE: t=[60,150] is a heuristic, not ground truth — R-peak centering can vary.
    # Running on N_SMOKE beats and reporting the FRACTION in zone is more robust than 1 beat.
    V_CLASS = 2
    N_SMOKE = 10           # number of V beats to sample
    QRS_T_ZONE = (60, 150) # inclusive, conservative window for QRS+T
    v_correct_idx = np.where((y_test == V_CLASS) & (preds_all == V_CLASS))[0]
    if len(v_correct_idx) >= 2:
        n_smoke = min(N_SMOKE, len(v_correct_idx))
        smoke_idx = v_correct_idx[np.linspace(0, len(v_correct_idx) - 1, n_smoke, dtype=int)]
        X_smoke_np = X_test[smoke_idx]                                   # (n_smoke, 187, 1)
        X_smoke_t  = torch.from_numpy(X_smoke_np).to(device)
        bg_smoke   = torch.from_numpy(
            X_train[np.random.default_rng(42).choice(len(X_train), 20, replace=False)]
        ).to(device)
        print(f"[CLINICAL SMOKE TEST] Running SHAP on {n_smoke} real V beats (class 2)...")
        explainer_smoke = shap.GradientExplainer(wrapper, bg_smoke)
        sv_smoke_raw    = explainer_smoke.shap_values(X_smoke_t)
        sv_smoke        = normalize_shap_output(sv_smoke_raw, 5)     # list of 5 arrays
        sv_v = np.abs(sv_smoke[V_CLASS]).squeeze(-1)                 # (n_smoke, 187)
        peak_ts = sv_v.argmax(axis=1)                                # (n_smoke,) — per-beat peak
        for t in peak_ts:
            assert 0 <= t < 187, f"[CLINICAL SMOKE TEST FAIL] peak_t={t} is out of bounds (signal length T=187)"
        in_zone = [(QRS_T_ZONE[0] <= int(t) <= QRS_T_ZONE[1]) for t in peak_ts]
        frac = sum(in_zone) / n_smoke
        marker = "✓" if frac >= 0.6 else "⚠ WARNING"
        print(f"[CLINICAL SMOKE TEST] V-beat peaks in QRS/T zone [{QRS_T_ZONE[0]},{QRS_T_ZONE[1]}]: "
              f"{sum(in_zone)}/{n_smoke} = {frac:.0%}  {marker}")
        print(f"  Per-beat peak timesteps: {[int(t) for t in peak_ts]}")
        if frac < 0.6:
            print(f"  Less than 60% of V beats peak inside QRS/T zone. Attribution may be diffuse")
            print(f"  or R-peak centering deviates from expectation. Check shap_class_V.png.")
            print(f"  This is a WARNING (not fatal) — a scientific observation, not a bug.")
    else:
        print("[CLINICAL SMOKE TEST] Fewer than 2 correctly classified V beats — skipping smoke test.")



    # ── Shared per-class samples for reproducible Jaccard(SHAP, IG) ──
    # Both SHAP and IG must use the SAME up-to-30 correctly-classified beats per class.
    # Using the same samples removes the "population SHAP vs single-beat IG" base mismatch.
    SHARED_N = 30
    shared_idx_c: dict = {}   # cls (int) → sorted list of test indices
    for cls in shap_classes:
        correct = np.where((y_test == cls) & (preds_all == cls))[0]
        n_sel = min(SHARED_N, len(correct))
        if n_sel > 0:
            shared_idx_c[cls] = sorted(rng.choice(correct, n_sel, replace=False).tolist())
        else:
            shared_idx_c[cls] = []
    all_shared_flat = sorted(set(i for idxs in shared_idx_c.values() for i in idxs))
    print(f"  Shared per-class samples for Jaccard: { {CLASS_NAMES[c]: len(v) for c, v in shared_idx_c.items()} }")

    # ── Select samples for detailed per-class visualization ──
    print("Selecting samples for detailed per-class analysis...")
    sample_indices = {}
    all_selected_idx = []
    for cls in shap_classes:
        ci, mi = select_samples(X_test, y_test, preds_all, cls,
                                n_correct=n_correct, n_mis=n_mis, rng=rng)
        sample_indices[cls] = {"correct": ci, "misclassified": mi}
        all_selected_idx.extend(ci)
        all_selected_idx.extend(mi)

    unique_selected_idx = np.array(sorted(list(set(all_selected_idx))))
    n_details = len(unique_selected_idx)
    print(f"  Selected {n_details} unique samples for detailed visualization.")

    # Compute SHAP on X_sum, X_details, and X_shared in one pass
    X_sum     = torch.from_numpy(X_test[rng.choice(len(X_test), n_summary, replace=False)]).to(device)
    X_details = torch.from_numpy(X_test[unique_selected_idx]).to(device)
    X_shared  = torch.from_numpy(X_test[all_shared_flat]).to(device)
    X_all_explain = torch.cat([X_sum, X_details, X_shared], dim=0)
    n_shared  = len(X_shared)
    n_all = len(X_all_explain)

    # GradientExplainer: average over multiple background resamplings.
    # Each resampling uses a deterministic seed offset to ensure reproducibility.
    # This stabilises the top-K timestep set used in Jaccard(SHAP, IG).
    print(f"Computing SHAP on {n_all} samples averaged over {n_shap_runs} background resamplings...")
    
    n_classes = 5  # model has 5 classes
    shap_runs = []  # list of shap_vals per run, each a list/ndarray of length n_classes
    for run_i in range(n_shap_runs):
        run_seed = seed + run_i * 1000
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        bg_idx = np.random.choice(len(X_train), n_background, replace=False)
        X_bg   = torch.from_numpy(X_train[bg_idx]).to(device)
        
        explainer = shap.GradientExplainer(wrapper, X_bg)
        sv_raw = explainer.shap_values(X_all_explain)
        sv = normalize_shap_output(sv_raw, n_classes)
        shap_runs.append(sv)
        print(f"  Run {run_i+1}/{n_shap_runs} done (seed={run_seed}, bg_idx[0]={bg_idx[0]})")

    # Average across runs: shap_vals_all[c] shape (n_all, T, 1)
    shap_vals_all = [
        np.mean([shap_runs[r][c] for r in range(n_shap_runs)], axis=0)
        for c in range(n_classes)
    ]
    print(f"  SHAP averaging complete. Stability note: averaged over {n_shap_runs} runs.")

    # Split back to sum, details, and shared
    shap_vals_summary = [sv[:n_summary]                          for sv in shap_vals_all]
    shap_vals_details = [sv[n_summary:n_summary + n_details]     for sv in shap_vals_all]
    shap_vals_shared  = [sv[n_summary + n_details:]              for sv in shap_vals_all]

    # Global importance: mean |SHAP| over classes 0-3 and samples
    mean_imp = np.stack(
        [np.abs(shap_vals_summary[c]).squeeze(-1).mean(axis=0) for c in shap_classes], axis=0
    ).mean(axis=0)  # (T,)

    plot_shap_summary(mean_imp, T_len, paths["out_explain"])

    # Top-K CSV
    top_k   = 20
    top_idx = np.argsort(mean_imp)[::-1][:top_k]
    for idx in top_idx:
        assert 0 <= idx < 187, f"[INDEX CHECK FAIL] Global rank index {idx} out of bounds (T=187)"
    with open(paths["out_explain"] / "shap_importance_ranking.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "timestep", "mean_abs_shap"])
        for rank, idx in enumerate(top_idx, 1):
            w.writerow([rank, int(idx), f"{mean_imp[idx]:.6f}"])
    print(f"  [OK] shap_importance_ranking.csv")

    # Per-class plots
    top_per_class = {}
    for cls in shap_classes:
        cn = CLASS_NAMES[cls]
        print(f"Per-class SHAP: {cn}...")
        ci = sample_indices[cls]["correct"]
        mi = sample_indices[cls]["misclassified"]

        if len(ci) > 0:
            # Map test indices to positions in unique_selected_idx
            pos_ci = [np.where(unique_selected_idx == idx)[0][0] for idx in ci]
            svc = shap_vals_details[cls][pos_ci]   # (n, T, 1)
            plot_shap_timeseries(
                svc, X_test[ci], y_test[ci], preds_all[ci], cn,
                f"SHAP — Correct Predictions  Class {cn}",
                paths["out_explain"] / f"shap_class_{cn}.png",
                max_plots=min(5, n_correct)
            )
        else:
            print(f"  No correct samples for {cn}")

        if len(mi) > 0:
            # Map test indices to positions in unique_selected_idx
            pos_mi = [np.where(unique_selected_idx == idx)[0][0] for idx in mi]
            svm = shap_vals_details[cls][pos_mi]   # (n, T, 1)
            plot_shap_timeseries(
                svm, X_test[mi], y_test[mi], preds_all[mi], cn,
                f"SHAP — Misclassified  Class {cn}  (Why does the model fail?)",
                paths["out_explain"] / f"shap_misclassified_{cn}.png",
                max_plots=min(5, n_mis)
            )
        else:
            print(f"  No misclassified samples for {cn}")

        cls_imp = np.abs(shap_vals_summary[cls]).squeeze(-1).mean(axis=0)
        top10    = np.argsort(cls_imp)[::-1][:10].tolist()
        for idx in top10:
            assert 0 <= idx < 187, f"[INDEX CHECK FAIL] {cn} top10 index {idx} out of bounds (T=187)"
        top_per_class[cn] = {
            "top10_timesteps": top10,
            "mean_importance": float(cls_imp.mean()),
            "correct_found":   int(len(ci)),
            "misclassified_found": int(len(mi))
        }

    # ── Per-class SHAP top-10 on SHARED samples (for valid Jaccard with IG) ──
    # Averaged over the same up-to-30 correctly-classified beats per class.
    shap_shared_top10: dict = {}
    for cls in shap_classes:
        cn = CLASS_NAMES[cls]
        idx_c = shared_idx_c.get(cls, [])
        if len(idx_c) == 0:
            shap_shared_top10[cn] = []
            continue
        # Positions of idx_c within all_shared_flat
        pos_c = [all_shared_flat.index(i) for i in idx_c]
        sv_c  = shap_vals_shared[cls][pos_c]          # (n, T, 1)
        cls_imp_shared = np.abs(sv_c).squeeze(-1).mean(axis=0)  # (T,)
        shap_shared_top10[cn] = np.argsort(cls_imp_shared)[::-1][:10].tolist()
        for idx in shap_shared_top10[cn]:
            assert 0 <= idx < 187, f"[INDEX CHECK FAIL] {cn} shared top10 index {idx} out of bounds (T=187)"
        print(f"  [Shared SHAP] {cn}: top-3 timesteps = {shap_shared_top10[cn][:3]}")

    # results.json
    results_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "explainability",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "shap_summary": "shap_summary_plot.png",
            "top_global_timesteps": top_idx[:3].tolist(),
            "per_class": top_per_class,
            # Shared-basis top-10: averaged over same idx_c used by IG → valid Jaccard
            "shap_shared_top10": shap_shared_top10,
            "shared_idx_c": {CLASS_NAMES[c]: idxs for c, idxs in shared_idx_c.items()},
        }
    }
    with open(paths["out_explain"] / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"  [OK] results.json (includes shap_shared_top10 for Jaccard)")

    print("\nSHAP analysis completed.")


if __name__ == "__main__":
    main()
