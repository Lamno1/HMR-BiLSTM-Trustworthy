"""
T3 — Integrated Gradients for Trustworthy ECG Classification (captum).

Annotates QRS, P-wave, T-wave regions on ECG signal.
Uses zero-signal as baseline.

Outputs:
  outputs/<run_id>/explainability/
    ig_normal.png
    ig_pvc.png      (class V)
    ig_apc.png      (class S)
    ig_fusion.png   (class F)
    ig_results.json (merged into explainability/results.json by T8)
"""

import json
import yaml
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

# Approximate ECG region boundaries for MIT-BIH 187-sample beat
# These are rough heuristics; the plot just annotates regions visually
ECG_REGIONS = {
    "P-wave":  (10,  40),
    "QRS":     (60,  100),
    "T-wave":  (110, 155),
}


def select_one_sample_per_class(X, y, preds, cls, rng):
    """Pick one correctly classified sample for the class, or any true class sample."""
    correct = np.where((y == cls) & (preds == cls))[0]
    if len(correct) > 0:
        idx = rng.choice(correct)
    else:
        fallback = np.where(y == cls)[0]
        if len(fallback) == 0:
            return None
        idx = rng.choice(fallback)
    return int(idx)


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
    """Symmetric Jaccard index with a temporal tolerance window."""
    matched_a = set(a for a in set_a if any(abs(a - b) <= tolerance for b in set_b))
    matched_b = set(b for b in set_b if any(abs(b - a) <= tolerance for a in set_a))
    intersection = (len(matched_a) + len(matched_b)) / 2.0
    union = len(set_a) + len(set_b) - intersection
    return intersection / max(1.0, union)


def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_explain"].mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")

    # Load model
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # Load test data & get predictions (using centralized INTER_TEST path)
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

    # ── Load shared_idx_c from SHAP results.json (same beats SHAP used for shap_shared_top10) ──
    shap_results_path = paths["out_explain"] / "results.json"
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
            print("  Warning: shared_idx_c not found in SHAP results — "
                  "will fall back to single-sample IG.")

    # — Integrated Gradients —
    ig = IntegratedGradients(model)
    n_steps  = 50
    ig_stats = {}
    SHARED_N = 30   # kept in sync with shap_analysis.py constant

    for cls, fname in PLOT_CLASSES.items():
        cls_label = CLASS_NAMES[cls]
        cn = cls_label.split()[0]   # e.g. "N", "S", "V", "F"
        print(f"Computing IG for class {cls_label}...")

        # — Determine sample set: shared if available, else single-sample fallback —
        use_shared = cn in shared_idx_c and len(shared_idx_c[cn]) > 0
        if use_shared:
            idx_list = shared_idx_c[cn]   # list[int], already correctly-classified beats
        else:
            idx_single = select_one_sample_per_class(X_test, y_test, preds_all, cls, rng)
            if idx_single is None:
                print(f"  Warning: no sample found for {cls_label}, skipping.")
                continue
            idx_list = [idx_single]

        # — Average |IG attribution| over all shared beats (same basis as SHAP) —
        all_attrs = []
        deltas    = []
        for idx in idx_list:
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

        attribution_mean = np.mean(np.abs(all_attrs), axis=0)  # (T,) mean |IG|
        mean_delta       = float(np.mean(deltas))
        top10 = np.argsort(attribution_mean)[::-1][:10].tolist()
        print(f"  n_beats={len(idx_list)}  mean|IG| max={attribution_mean.max():.4f}  "
              f"mean_conv_delta={mean_delta:.4f}  top3={top10[:3]}")

        # — Plot with first representative beat —
        repr_idx   = idx_list[0]
        signal     = X_test[repr_idx].squeeze()
        pred_lbl   = CLASS_NAMES.get(int(preds_all[repr_idx]), str(preds_all[repr_idx]))
        true_lbl   = CLASS_NAMES[cls]
        n_beats    = len(idx_list)
        title = (f"Integrated Gradients — {cls_label}  (mean |IG| over {n_beats} beats)\n"
                 f"Representative beat — True: {true_lbl}  |  Predicted: {pred_lbl}  |  "
                 f"Mean conv. delta: {mean_delta:.4f}")
        plot_ig(signal, attribution_mean, cls_label,
                paths["out_explain"] / fname, title)

        ig_stats[cn] = {
            "top10_timesteps_ig": top10,
            "n_beats_averaged":   n_beats,
            "mean_convergence_delta": mean_delta,
            "max_mean_abs_attr":  float(attribution_mean.max()),
            "used_shared_basis":  use_shared,
        }

    # — Save IG-specific JSON —
    ig_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "explainability_ig",
        "timestamp": datetime.now().isoformat(),
        "metrics": ig_stats
    }
    ig_json_path = paths["out_explain"] / "ig_results.json"
    with open(ig_json_path, "w", encoding="utf-8") as f:
        json.dump(ig_json, f, indent=2)
    print(f"  [OK] ig_results.json")

    # — Jaccard(SHAP, IG) — both now on the same shared beats —
    # shap_shared_top10 comes from shap_analysis.py averaged over shared_idx_c.
    # ig_stats[cn]["top10_timesteps_ig"] is averaged over the same idx_list.
    # Comparison is now apples-to-apples.
    consistency = {}
    if shap_shared_top10:
        for cn, ig_data in ig_stats.items():
            shap_top10 = set(shap_shared_top10.get(cn, []))
            ig_top10   = set(ig_data["top10_timesteps_ig"])
            if shap_top10 and ig_top10:
                jaccard = jaccard_with_tolerance(shap_top10, ig_top10, tolerance=2)
                consistency[cn] = {
                    "jaccard_shared_basis": float(jaccard),
                    "shap_top10": list(shap_top10),
                    "ig_top10":   list(ig_top10),
                    "n_beats_shared": len(shared_idx_c.get(cn, [])),
                    "note": "Both averaged over same correctly-classified beats per class.",
                }
                print(f"  Jaccard(SHAP_shared, IG_shared) [{cn}]: {jaccard:.4f}  "
                      f"(n_beats={len(shared_idx_c.get(cn, []))})")
        if consistency:
            consistency_path = paths["out_explain"] / "shap_ig_consistency.json"
            with open(consistency_path, "w", encoding="utf-8") as f:
                json.dump({"jaccard_shared_basis": consistency,
                           "methodology": (
                               "Both SHAP and IG top-10 computed by averaging |attribution| "
                               "over the SAME up-to-30 correctly-classified beats per class. "
                               "jaccard_with_tolerance(tolerance=2) applied."
                           )}, f, indent=2)
            print(f"  [OK] shap_ig_consistency.json (shared-basis Jaccard)")
    else:
        print("  Note: SHAP shared_top10 not available — run shap_analysis.py first "
              "to enable Jaccard comparison.")

    print("\nIntegrated Gradients analysis completed.")


if __name__ == "__main__":
    main()
