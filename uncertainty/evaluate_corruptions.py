import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
from report_results import load_hmr_bilstm
from uncertainty.mc_dropout import (
    ood_gaussian_noise, ood_baseline_wander, ood_random_crop_pad, ood_signal_shift
)

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

@torch.no_grad()
def evaluate_model(model, X, y_true, device, batch_size=128):
    model.eval()
    all_preds = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i+batch_size]).to(device)
        logits = model(batch)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(all_preds)
    
    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, preds, average=None, zero_division=0)
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))}
    }

def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_uncert"].mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Force CPU: ensemble training occupies GPU, inference-only corruption sweep runs fine on CPU
    device = torch.device("cpu")
    print(f"Device: {device} (CPU forced — GPU reserved for ensemble training) | Run ID: {run_id}")

    # Load model
    print("Loading HMR-BiLSTM model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # Load test data
    print(f"Loading inter-patient test data: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    # ── Define sweeps ──
    gaussian_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    wander_levels   = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    crop_levels     = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    shift_levels    = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

    results = {
        "gaussian": [],
        "wander": [],
        "crop": [],
        "shift": []
    }

    # Evaluate Gaussian Noise
    print("\nSweeping Gaussian Noise std...")
    for std in gaussian_levels:
        X_corr = ood_gaussian_noise(X_test, std, rng=rng)
        metrics = evaluate_model(model, X_corr, y_test, device)
        results["gaussian"].append({"intensity": std, **metrics})
        print(f"  sigma={std:.2f} -> Macro F1={metrics['f1_macro']:.4f}")

    # Evaluate Baseline Wander
    print("\nSweeping Baseline Wander amp...")
    for amp in wander_levels:
        X_corr = ood_baseline_wander(X_test, freq=0.3, amp=amp, rng=rng)
        metrics = evaluate_model(model, X_corr, y_test, device)
        results["wander"].append({"intensity": amp, **metrics})
        print(f"  amp={amp:.2f} -> Macro F1={metrics['f1_macro']:.4f}")

    # Evaluate Random Crop/Pad
    print("\nSweeping Random Crop fraction...")
    for crop in crop_levels:
        X_corr = ood_random_crop_pad(X_test, crop_frac=crop, rng=rng)
        metrics = evaluate_model(model, X_corr, y_test, device)
        results["crop"].append({"intensity": crop, **metrics})
        print(f"  crop={crop:.2f} -> Macro F1={metrics['f1_macro']:.4f}")

    # Evaluate Signal Shift
    print("\nSweeping Temporal Shift fraction...")
    for shift in shift_levels:
        X_corr = ood_signal_shift(X_test, shift_frac=shift, rng=rng)
        metrics = evaluate_model(model, X_corr, y_test, device)
        results["shift"].append({"intensity": shift, **metrics})
        print(f"  shift={shift:.2f} -> Macro F1={metrics['f1_macro']:.4f}")

    # ── Save results JSON ──
    out_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id":             run_id,
        "checkpoint_hash":    get_checkpoint_hash(RLSTM_CKPT),
        "module":             "corruption_robustness_sweep",
        "timestamp":          datetime.now().isoformat(),
        "sweeps": {
            "gaussian": {
                "intensity_name": "Standard Deviation (sigma)",
                "data": results["gaussian"]
            },
            "baseline_wander": {
                "intensity_name": "Amplitude (freq=0.3 Hz)",
                "data": results["wander"]
            },
            "random_crop": {
                "intensity_name": "Crop Fraction",
                "data": results["crop"]
            },
            "temporal_shift": {
                "intensity_name": "Shift Fraction",
                "data": results["shift"]
            }
        }
    }
    
    out_path = paths["out_uncert"] / "corruption_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[OK] Saved results JSON to {out_path}")

    # ── Plotting degradation curves ──
    print("Generating degradation plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#121212")

    plot_configs = [
        ("gaussian", "Gaussian Noise std (sigma)", axs[0, 0], "#4CAF50"),
        ("wander", "Baseline Wander amp (freq=0.3 Hz)", axs[0, 1], "#2196F3"),
        ("crop", "Random Crop Fraction", axs[1, 0], "#FF9800"),
        ("shift", "Temporal Shift Fraction", axs[1, 1], "#9C27B0")
    ]

    for key, xlabel, ax, color in plot_configs:
        ax.set_facecolor("#1e1e1e")
        intensities = [x["intensity"] for x in results[key]]
        macro_f1s = [x["f1_macro"] for x in results[key]]
        accuracies = [x["accuracy"] for x in results[key]]
        
        ax.plot(intensities, macro_f1s, "o-", linewidth=2.5, color=color, label="Macro F1-Score")
        ax.plot(intensities, accuracies, "s--", linewidth=1.5, color="#888888", alpha=0.7, label="Accuracy")
        
        ax.set_xlabel(xlabel, color="#e0e0e0", fontsize=11)
        ax.set_ylabel("Metric Score", color="#e0e0e0", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        ax.grid(True, linestyle="--", alpha=0.2, color="#555555")
        ax.set_ylim([0.0, 1.02])
        ax.legend(loc="lower left", facecolor="#2e2e2e", edgecolor="#444444", labelcolor="#e0e0e0")
        ax.spines["bottom"].set_color("#555555")
        ax.spines["top"].set_color("#555555")
        ax.spines["left"].set_color("#555555")
        ax.spines["right"].set_color("#555555")

    plt.suptitle("HMR-BiLSTM: Sensitivity to Signal Degradation (Corruptions)", 
                 color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_path = paths["out_uncert"] / "corruption_degradation.png"
    plt.savefig(fig_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Saved degradation plot to {fig_path}")

if __name__ == "__main__":
    main()
