"""
evaluate_ablation_robustness.py
-------------------------------
Evaluate the adversarial robustness (FGSM and PGD) of the ablation variants.

Usage:
    python evaluate_ablation_robustness.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from hmr_bilstm_ablation import RLSTMClassifier, RLSTMLoss
from evaluate_fgsm import evaluate_fgsm, build_test_loader
from evaluate_pgd import evaluate_pgd_grid

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

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="Evaluate ablation robustness")
    parser.add_argument("--epsilons", nargs="*", type=float, default=[0.0, 0.02, 0.05])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ckpt-dir", default="results/ablation/checkpoints")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We evaluate all variants in the checkpoints directory
    ablation_checkpoints = {}
    
    # Do not load existing results if we want a fresh run
    # csv_path = output_dir / "ablation_robustness.csv"
    existing_results = []
    # if csv_path.exists():
    #     with open(csv_path, "r", encoding="utf-8") as f:
    #         reader = csv.DictReader(f)
    #         existing_results = list(reader)
    
    results = existing_results.copy()
    print(f"Device : {device}")
    
    test_loader = build_test_loader(batch_size=args.batch_size)
    
    cw_path = Path("data/processed/class_weights.npy")
    if cw_path.exists():
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
    else:
        class_weights = None
        
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=class_weights,
        use_focal=True,
        focal_gamma=1.5,
    )
    
    ckpt_dir = Path(args.ckpt_dir)
    checkpoints = list(ckpt_dir.glob("best_rlstm_*.pt"))
    
    if not checkpoints:
        print(f"No ablation checkpoints found in {ckpt_dir}")
        return
        
    results = []
    
    for ckpt_path in checkpoints:
        variant = ckpt_path.stem.replace("best_rlstm_", "")
        print(f"\n[Evaluating Variant: {variant}]")
        
        try:
            model, cfg, flags = load_ablation_model(ckpt_path, device)
            
            # evaluate FGSM
            fgsm_res = []
            for eps in args.epsilons:
                print(f"  FGSM epsilon={eps}")
                r = evaluate_fgsm(model, test_loader, device, criterion, eps)
                fgsm_res.append(r)
                
            # evaluate PGD (skipped for ablation to save time)
            pgd_res = evaluate_pgd_grid(model, test_loader, device, criterion, args.epsilons, alpha=0.005, steps=20)
            
            clean_f1 = fgsm_res[0]["macro_f1"]
            
            row = {
                "variant": variant,
                "label": flags.get("label", variant),
                "clean_f1": f"{clean_f1:.4f}",
            }
            
            # FGSM
            r02_fgsm = min(fgsm_res, key=lambda r: abs(r["epsilon"] - 0.02))
            row["fgsm_f1_002"] = f"{r02_fgsm['macro_f1']:.4f}"
            row["fgsm_asr_002"] = f"{r02_fgsm['attack_success_rate']:.4f}"
            
            r05_fgsm = min(fgsm_res, key=lambda r: abs(r["epsilon"] - 0.05))
            row["fgsm_f1_005"] = f"{r05_fgsm['macro_f1']:.4f}"
            
            # PGD (skipped)
            r02_pgd = min(pgd_res, key=lambda r: abs(r["epsilon"] - 0.02))
            row["pgd_f1_002"] = f"{r02_pgd['macro_f1']:.4f}"   
            row["pgd_asr_002"] = f"{r02_pgd['attack_success_rate']:.4f}"
            
            r05_pgd = min(pgd_res, key=lambda r: abs(r["epsilon"] - 0.05))
            row["pgd_f1_005"] = f"{r05_pgd['macro_f1']:.4f}"
            
            results.append(row)
            
        except Exception as e:
            print(f"  [ERR] {variant}: {e}")
            
    csv_path = output_dir / "ablation_robustness.csv"
    if results:
        header = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OK] Saved ablation robustness results to {csv_path}")

        print("\n" + "=" * 100)
        print(f"{'Variant':<28} {'Clean':<8} | {'FGSM 0.02':<10} {'ASR':<9} {'FGSM 0.05':<10} | {'PGD 0.02':<10} {'ASR':<9} {'PGD 0.05':<10}")
        print("-" * 100)
        for r in results:
            print(f"{r['label']:<28} {r['clean_f1']:<8} | {r['fgsm_f1_002']:<10} {r['fgsm_asr_002']:<9} {r['fgsm_f1_005']:<10} | {r['pgd_f1_002']:<10} {r['pgd_asr_002']:<9} {r['pgd_f1_005']:<10}")
        print("=" * 100)

if __name__ == "__main__":
    main()
