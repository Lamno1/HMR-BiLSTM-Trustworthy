import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader, TensorDataset

# Import model loaders from existing scripts
from report_results import load_hmr_bilstm
from evaluate_fgsm import load_baseline_model, build_test_loader
from calibration.temperature_scaling import TemperatureScaling, build_val_loader

def calculate_brier_score(probs, labels, num_classes=5):
    """
    Calculate the Brier Score for multi-class classification.
    probs: numpy array of shape (N, C)
    labels: numpy array of shape (N,)
    """
    # One-hot encode labels
    y_true = np.eye(num_classes)[labels]
    # Brier score is the mean squared difference between predicted probabilities and actual outcomes
    brier_score = np.mean(np.sum((probs - y_true)**2, axis=1))
    return brier_score

def calculate_ece(probs, labels, num_bins=15):
    """
    Calculate Expected Calibration Error (ECE) for multi-class classification.
    Uses the predicted class's probability and whether it matches the true label.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if a prediction falls into the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        # For the first bin, also include exactly 0
        if bin_lower == 0.0:
            in_bin = in_bin | (confidences == 0.0)
            
        prop_in_bin = in_bin.mean()
        count_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append({
                'bin_center': (bin_lower + bin_upper) / 2,
                'conf': avg_confidence_in_bin,
                'acc': accuracy_in_bin,
                'count': count_in_bin,
                'prop': prop_in_bin
            })
    
    return ece, bin_stats

def get_model_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels

def get_model_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)

def plot_reliability_diagram(models_stats, output_path):
    """
    Plot a reliability diagram with:
      - Top panel : confidence vs accuracy (reliability curve)
      - Bottom panel: bin sample counts histogram (raw counts, not fractions)
    """
    num_models = len(models_stats)
    # Wider figure when 3 models so bars don't overlap
    fig_w = 9 if num_models >= 3 else 8
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [3, 1.4]},
        figsize=(fig_w, 11),
        sharex=True
    )
    fig.subplots_adjust(hspace=0.08)

    # Perfect calibration reference line
    ax1.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", linewidth=1.8, zorder=0)

    colors  = {'LSTM': '#1f77b4', 'BiLSTM': '#ff7f0e', 'HMR-BiLSTM': '#2ca02c'}
    markers = {'LSTM': 'o',       'BiLSTM': 's',        'HMR-BiLSTM': '^'}

    # ── histogram layout ────────────────────────────────────────────────────
    # Spread bars across the full bin width (0.1) divided evenly by # models
    BIN_W   = 0.10          # each confidence bin spans 0.10
    bar_w   = BIN_W / (num_models + 1)   # leave a small gap between groups
    offsets_base = np.linspace(
        -(bar_w * (num_models - 1) / 2),
         (bar_w * (num_models - 1) / 2),
        num_models
    )

    for idx, (model_name, stats) in enumerate(models_stats.items()):
        bin_stats = stats['bin_stats']
        if not bin_stats:
            continue

        confs       = [b['conf']       for b in bin_stats]
        accs        = [b['acc']        for b in bin_stats]
        bin_centers = [b['bin_center'] for b in bin_stats]
        counts      = [b['count']      for b in bin_stats]   # ← raw counts

        c = colors.get(model_name,  '#333333')
        m = markers.get(model_name, 'x')

        # ── Top panel: reliability curve ────────────────────────────────────
        ax1.plot(
            confs, accs,
            marker=m, color=c,
            label=f"{model_name}  (ECE={stats['ece']:.4f}, Brier={stats['brier']:.4f})",
            linewidth=2, markersize=8, zorder=2
        )

        # ── Bottom panel: count histogram ───────────────────────────────────
        bar_positions = np.array(bin_centers) + offsets_base[idx]
        bars = ax2.bar(
            bar_positions, counts,
            width=bar_w * 0.9,   # 0.9 so bars don't touch
            color=c, alpha=0.75,
            label=model_name,
            zorder=2
        )

        # Use max(counts)*0.02 to place text nicely
        # get_ylim() được gọi trước khi canvas finalize → trả về stale default (0,1)
        # → offset = 0.01 unit thay vì ~50+ sample units → label bị chồng lên bar.
        _offset = max(counts) * 0.02 if max(counts) > 0 else 1
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + _offset,
                    str(int(cnt)),
                    ha='center', va='bottom',
                    fontsize=6.5, color=c, fontweight='bold'
                )

    # ── Top panel formatting ─────────────────────────────────────────────────
    ax1.set_ylabel("Fraction of Positives (Accuracy)", fontsize=13)
    ax1.set_title("Reliability Diagram", fontsize=16, fontweight='bold', pad=10)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper left", fontsize=10.5, framealpha=0.9)
    ax1.grid(alpha=0.35, linestyle="--")

    # ── Bottom panel formatting ───────────────────────────────────────────────
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=13)
    ax2.set_ylabel("Sample Count\n(per bin)", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_xticks(np.arange(0, 1.1, 0.1))
    ax2.grid(alpha=0.35, linestyle="--", axis='y', zorder=0)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.set_title("Bin Frequency (# Samples)", fontsize=11, pad=4)

    # Re-draw to get correct ylim for annotation offset, then tighten
    fig.canvas.draw()
    # Re-annotate with correct ylim after draw
    ax2.autoscale_view()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved reliability diagram -> {output_path}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_path = Path("configs/experiment_config.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    num_bins  = cfg.get("calibration", {}).get("num_bins", 15)
    temp_lr   = cfg.get("calibration", {}).get("temp_lr", 0.01)
    temp_iter = cfg.get("calibration", {}).get("temp_max_iter", 50)

    # Load test data
    test_loader = build_test_loader(batch_size=128)

    models_info = {
        "LSTM": ("results/checkpoints/best_lstm.pt", "baseline"),
        "BiLSTM": ("results/checkpoints/best_bilstm.pt", "baseline"),
        "HMR-BiLSTM": ("results/checkpoints/best_rlstm.pt", "rlstm"),
    }

    results = {}

    for model_name, (ckpt_path, m_type) in models_info.items():
        if not Path(ckpt_path).exists():
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        if m_type == "rlstm":
            model, _ = load_hmr_bilstm(ckpt_path, device)
        else:
            model, _ = load_baseline_model(ckpt_path, device)

        probs, labels = get_model_predictions(model, test_loader, device)

        brier                    = calculate_brier_score(probs, labels)
        ece_uncal, bin_stats     = calculate_ece(probs, labels, num_bins=num_bins)
        ece_cal, bin_stats_cal   = None, None
        optimal_temp             = None

        if m_type == "rlstm":
            val_loader   = build_val_loader(batch_size=128)
            ts_model     = TemperatureScaling().to(device)
            optimal_temp = ts_model.fit(model, val_loader, device,
                                        lr=temp_lr, max_iter=temp_iter)
            logits, _ = get_model_logits(model, test_loader, device)
            logits     = logits.to(device)
            with torch.no_grad():
                cal_probs = torch.softmax(ts_model(logits), dim=1).cpu().numpy()
            ece_cal, bin_stats_cal = calculate_ece(cal_probs, labels, num_bins=num_bins)

        print(f"{model_name} Results:")
        print(f"  ECE (uncalibrated): {ece_uncal:.4f}")
        if ece_cal is not None:
            print(f"  ECE (calibrated,  T={optimal_temp:.4f}): {ece_cal:.4f}")
        print(f"  Brier Score: {brier:.4f}")

        results[model_name] = {
            'ece':            ece_uncal,
            'ece_calibrated': ece_cal,
            'temperature':    optimal_temp,
            'brier':          brier,
            'bin_stats':      bin_stats,
        }

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = Path("results/tables/calibration_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by ECE ascending so best model is first
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ece'])
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'Model',
            'ECE_uncalibrated', 'ECE_calibrated (HMR-BiLSTM only)',
            'Brier Score', 'ECE Interpretation', 'Brier Interpretation'
        ])
        for rank, (model_name, stats) in enumerate(sorted_results, start=1):
            ece_interp   = 'Excellent (<0.02)' if stats['ece'] < 0.02 else (
                           'Good (0.02-0.05)'  if stats['ece'] < 0.05 else (
                           'Fair (0.05-0.10)'  if stats['ece'] < 0.10 else 'Poor (>=0.10)'))
            brier_interp = 'Excellent (<0.05)' if stats['brier'] < 0.05 else (
                           'Good (0.05-0.10)'  if stats['brier'] < 0.10 else (
                           'Fair (0.10-0.20)'  if stats['brier'] < 0.20 else 'Poor (>=0.20)'))
            ece_cal_str = f"{stats['ece_calibrated']:.4f}" if stats.get('ece_calibrated') is not None else "N/A"
            writer.writerow([
                rank, model_name,
                f"{stats['ece']:.4f}", ece_cal_str,
                f"{stats['brier']:.4f}",
                ece_interp, brier_interp
            ])
    print(f"\n[OK] Saved calibration metrics -> {csv_path}")

    # -- Interpretation Helper ------------------------------------------------
    print("\n" + "=" * 55)
    print(" CALIBRATION INTERPRETATION HELPER ")
    print("=" * 55)
    print(" Metric         | Direction | Ideal")
    print(" ---------------+-----------+----------------------")
    print(" ECE            | lower     | 0.00  (perfect align)")
    print(" Brier Score    | lower     | 0.00  (perfect probs)")
    print("=" * 55)
    print("\n Calibration quality thresholds:")
    print("   ECE  : <0.02 Excellent | 0.02-0.05 Good | 0.05-0.10 Fair | >=0.10 Poor")
    print("   Brier: <0.05 Excellent | 0.05-0.10 Good | 0.10-0.20 Fair | >=0.20 Poor")
    print("=" * 55)
    print("\n Ranked results (by ECE):")
    print(f"   {'Rank':<5} {'Model':<14} {'ECE':>8} {'Brier':>10}")
    print("   " + "-" * 42)
    for rank, (model_name, stats) in enumerate(sorted_results, start=1):
        medal = ['[1]', '[2]', '[3]'][rank - 1] if rank <= 3 else '   '
        print(f"   {medal} {rank:<3} {model_name:<14} {stats['ece']:>8.4f} {stats['brier']:>10.4f}")
    print("=" * 55)
    
    fig_path = output_dir / "reliability_diagram.png"
    plot_reliability_diagram(results, fig_path)

if __name__ == "__main__":
    main()
