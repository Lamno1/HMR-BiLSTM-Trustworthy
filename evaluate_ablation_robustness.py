"""
evaluate_ablation_robustness.py
-------------------------------
Evaluate the adversarial robustness (FGSM and PGD) of the ablation variants.

Features:
  - Incremental save after EACH variant (safe against crashes)
  - Resume support: skips variants that already have full PGD data
  - cudnn disabled for RNN compatibility during PGD backward pass

Usage:
    python evaluate_ablation_robustness.py
    python evaluate_ablation_robustness.py --skip-pgd    # FGSM only
    python evaluate_ablation_robustness.py --force       # re-run all variants
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hmr_bilstm_ablation import RLSTMClassifier, RLSTMLoss
from evaluate_fgsm import evaluate_fgsm, build_test_loader
from evaluate_pgd import evaluate_pgd_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ablation_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    variant_flags = ckpt["variant_flags"]

    model = RLSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
        use_rmc=variant_flags.get("use_rmc", True),
        use_hybrid=variant_flags.get("use_hybrid", True),
        use_cnn=variant_flags.get("use_cnn", True),
        use_attention=variant_flags.get("use_attention", True),
        use_interaction=variant_flags.get("use_interaction", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cfg, variant_flags


def load_existing_results(csv_path):
    """Load existing CSV results into a dict keyed by variant."""
    if not csv_path.exists():
        return {}
    existing = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["variant"]] = row
    return existing


def has_full_pgd_data(row):
    """Return True if this row already contains valid PGD numbers."""
    pgd_cols = ["pgd_f1_002", "pgd_asr_002", "pgd_f1_005"]
    return all(row.get(c, "-") not in ("", "-", "N/A") for c in pgd_cols)


def has_full_fgsm_data(row):
    """Return True if this row already has valid FGSM numbers."""
    fgsm_cols = ["fgsm_f1_002", "fgsm_asr_002", "fgsm_f1_005", "clean_f1"]
    return all(row.get(c, "-") not in ("", "-", "N/A") for c in fgsm_cols)


def save_results(results_list, csv_path):
    """Write the full results list to CSV (atomic overwrite via temp file)."""
    if not results_list:
        return
    header = [
        "variant", "label", "clean_f1",
        "fgsm_f1_002", "fgsm_asr_002", "fgsm_f1_005",
        "pgd_f1_002",  "pgd_asr_002",  "pgd_f1_005",
    ]
    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_list)
    # Atomic rename
    if csv_path.exists():
        csv_path.unlink()
    tmp_path.rename(csv_path)
    print(f"  [SAVED] {csv_path}  ({len(results_list)} rows)")


def print_summary(results):
    print("\n" + "=" * 110)
    print(f"{'Variant':<28} {'Clean':<8} | {'FGSM 0.02':<10} {'ASR':<9} {'FGSM 0.05':<10} | {'PGD 0.02':<10} {'ASR':<9} {'PGD 0.05':<10}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['label']:<28} {r['clean_f1']:<8} | "
            f"{r.get('fgsm_f1_002','-'):<10} {r.get('fgsm_asr_002','-'):<9} {r.get('fgsm_f1_005','-'):<10} | "
            f"{r.get('pgd_f1_002','-'):<10} {r.get('pgd_asr_002','-'):<9} {r.get('pgd_f1_005','-'):<10}"
        )
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Evaluate ablation robustness (FGSM + PGD)")
    parser.add_argument("--epsilons",   nargs="*", type=float, default=[0.0, 0.02, 0.05])
    parser.add_argument("--batch-size", type=int,  default=128)
    parser.add_argument("--ckpt-dir",   default="results/ablation/checkpoints")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--pgd-steps",  type=int,  default=20,    help="PGD iterations")
    parser.add_argument("--pgd-alpha",  type=float, default=0.005, help="PGD step size")
    parser.add_argument("--skip-pgd",   action="store_true",       help="Run FGSM only (fast)")
    parser.add_argument("--force",      action="store_true",       help="Re-evaluate all variants even if already saved")
    parser.add_argument("--cpu",        action="store_true",       help="Force CPU (use when cuDNN not installed)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN globally for RNN backward compatibility on GPU")

    if args.skip_pgd:
        print("Mode   : FGSM-only (--skip-pgd flag set)")
    else:
        print(f"PGD    : steps={args.pgd_steps}  alpha={args.pgd_alpha}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_robustness.csv"

    # --- Load previously saved results ---
    existing = load_existing_results(csv_path)
    print(f"Existing results: {len(existing)} variants in {csv_path}")

    # --- Discover checkpoints ---
    ckpt_dir = Path(args.ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("best_rlstm_*.pt"))
    if not checkpoints:
        print(f"[ERROR] No ablation checkpoints found in {ckpt_dir}")
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoint(s): {[p.name for p in checkpoints]}")

    # --- Build test loader & criterion once ---
    test_loader = build_test_loader(batch_size=args.batch_size)

    cw_path = Path("data/processed/class_weights.npy")
    class_weights = torch.from_numpy(np.load(cw_path)).float().to(device) if cw_path.exists() else None
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=class_weights,
        use_focal=True,
        focal_gamma=1.5,
    )

    # Start from existing results (variant -> row dict)
    results_map = dict(existing)

    # --- Evaluate each checkpoint ---
    for ckpt_path in checkpoints:
        variant = ckpt_path.stem.replace("best_rlstm_", "")

        # Resume: skip if already have full data
        if not args.force and variant in results_map:
            row = results_map[variant]
            if args.skip_pgd or has_full_pgd_data(row):
                print(f"\n[SKIP] {variant}  (already complete)")
                continue

        print(f"\n{'='*60}")
        print(f"[Evaluating Variant: {variant}]  ({ckpt_path.name})")
        print(f"{'='*60}")

        # Check if we can reuse existing FGSM data
        existing_row = results_map.get(variant, {})
        fgsm_already_done = not args.force and has_full_fgsm_data(existing_row)

        try:
            model, cfg, flags = load_ablation_model(ckpt_path, device)

            # --- FGSM (skip if already computed) ---
            if fgsm_already_done:
                print(f"  FGSM  [SKIP - already saved]")
                row = dict(existing_row)  # keep existing FGSM values
            else:
                fgsm_res = []
                for eps in args.epsilons:
                    print(f"  FGSM  epsilon={eps:.3f}")
                    r = evaluate_fgsm(model, test_loader, device, criterion, eps)
                    fgsm_res.append(r)

                clean_f1 = fgsm_res[0]["macro_f1"]
                r02_fgsm = min(fgsm_res, key=lambda r: abs(r["epsilon"] - 0.02))
                r05_fgsm = min(fgsm_res, key=lambda r: abs(r["epsilon"] - 0.05))

                row = {
                    "variant":      variant,
                    "label":        flags.get("label", variant),
                    "clean_f1":     f"{clean_f1:.4f}",
                    "fgsm_f1_002":  f"{r02_fgsm['macro_f1']:.4f}",
                    "fgsm_asr_002": f"{r02_fgsm['attack_success_rate']:.4f}",
                    "fgsm_f1_005":  f"{r05_fgsm['macro_f1']:.4f}",
                    "pgd_f1_002":   "-",
                    "pgd_asr_002":  "-",
                    "pgd_f1_005":   "-",
                }

            # --- PGD ---
            if not args.skip_pgd:
                print(f"  PGD   steps={args.pgd_steps}  alpha={args.pgd_alpha}  epsilons={args.epsilons}")
                pgd_res = evaluate_pgd_grid(
                    model, test_loader, device, criterion,
                    args.epsilons, alpha=args.pgd_alpha, steps=args.pgd_steps
                )
                r02_pgd = min(pgd_res, key=lambda r: abs(r["epsilon"] - 0.02))
                r05_pgd = min(pgd_res, key=lambda r: abs(r["epsilon"] - 0.05))
                row["pgd_f1_002"]  = f"{r02_pgd['macro_f1']:.4f}"
                row["pgd_asr_002"] = f"{r02_pgd['attack_success_rate']:.4f}"
                row["pgd_f1_005"]  = f"{r05_pgd['macro_f1']:.4f}"

            results_map[variant] = row
            print(f"  [OK] {variant}  clean={row['clean_f1']}  fgsm_f1@0.02={row['fgsm_f1_002']}  pgd_f1@0.02={row['pgd_f1_002']}")

        except Exception as e:
            import traceback
            print(f"  [ERR] {variant}: {e}")
            traceback.print_exc()
            # Keep existing data if any

        finally:
            # -------------------------------------------------------
            # INCREMENTAL SAVE — runs after every variant (success OR fail)
            # -------------------------------------------------------
            current_rows = list(results_map.values())
            try:
                save_results(current_rows, csv_path)
            except Exception as save_err:
                print(f"  [WARN] Could not save to {csv_path}: {save_err}")
                # Fallback to backup path
                backup = output_dir / "ablation_robustness_backup.csv"
                try:
                    save_results(current_rows, backup)
                    print(f"  [BACKUP] Saved to {backup}")
                except Exception as be:
                    print(f"  [CRITICAL] Backup also failed: {be}")

    # --- Final print ---
    final_rows = list(results_map.values())
    if final_rows:
        print_summary(final_rows)
        print(f"\n[DONE] Results saved to {csv_path}")
    else:
        print("[DONE] No results to save.")


if __name__ == "__main__":
    main()
