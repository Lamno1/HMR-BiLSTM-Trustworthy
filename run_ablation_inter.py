"""
run_ablation_inter.py
======================
Chạy ablation study cho HMR-BiLSTM trên MIT-BIH ECG (Inter-Patient split).
Các variant được hỗ trợ:
    full   - HMR-BiLSTM đầy đủ (có thể load trực tiếp từ results/checkpoints/inter_best_rlstm.pt)
    no_rmc - bỏ RMC path, c_t = c_lstm (BiLSTM baseline với CNN + Attention)
    no_adv - adversarial_training=False (huấn luyện trên dữ liệu sạch inter-patient)

Cách chạy:
    python run_ablation_inter.py
"""

import os
import sys
import json
import time
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
import argparse

from hmr_bilstm_ablation import RLSTMClassifier, RLSTMLoss

# CONFIG matches train_inter_patient.py
BASE_CONFIG = {
    "data_dir":                "data/processed/splits",
    "seed":                    42,
    "batch_size":              128,
    "hidden_size":             96,
    "dropout":                 0.25,
    "learning_rate":           1e-3,
    "min_lr":                  1e-5,
    "weight_decay":            1e-4,
    "lambda_smooth":           0.003,
    "epochs":                  45,
    "early_stopping_patience": 8,
    "grad_clip":               1.0,
    "num_classes":             5,
    "input_size":              1,
    "cnn_out_channels":        64,
    "num_layers":              2,
    "use_focal_loss":          True,
    "focal_gamma":             1.5,
    "adversarial_training":    True,
    "adv_epsilon":             0.02,
    "adv_ratio":               0.3,
    "use_class_weights":       True,
}

VARIANTS = {
    "full": {
        "use_rmc": True, "use_cnn": True, "use_attention": True,
        "label": "HMR-BiLSTM (full)",
    },
    "no_rmc": {
        "use_rmc": False, "use_cnn": True, "use_attention": True,
        "label": "No-RMC (c_t = c_lstm)",
    },
    "no_adv": {
        "use_rmc": True, "use_cnn": True, "use_attention": True,
        "label": "No-Adv-Training",
        "override_config": {"adversarial_training": False},
    },
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    progress = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def load_data(data_dir, batch_size):
    def make_loader(split, shuffle=False):
        d = np.load(f"{data_dir}/inter_{split}.npz")
        ds = TensorDataset(
            torch.from_numpy(d["X"]).float(),
            torch.from_numpy(d["y"]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        make_loader("train", shuffle=True),
        make_loader("val"),
        make_loader("test"),
    )

@torch.no_grad()
def evaluate(model, loader, device, num_classes=5):
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_logits.append(logits.cpu())
        all_y.append(y)

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()

    metrics = {
        "accuracy":        float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_weighted":     float(f1_score(y_true, preds, labels=[0, 1, 2, 3], average="weighted", zero_division=0)),
    }
    
    try:
        # AUC: restrict to AAMI 4-class (labels 0-3) and only classes present in y_true.
        valid_idx = np.isin(y_true, [0, 1, 2, 3])
        y_true_4c = y_true[valid_idx]
        y_prob_4c = probs[valid_idx]

        present_labels = sorted(set(y_true_4c.tolist()) & {0, 1, 2, 3})
        if len(present_labels) >= 2:
            if y_prob_4c.shape[1] < 4:
                y_prob_4c_padded = np.zeros((y_prob_4c.shape[0], 4))
                y_prob_4c_padded[:, :y_prob_4c.shape[1]] = y_prob_4c
                y_prob_4c = y_prob_4c_padded
            else:
                y_prob_4c = y_prob_4c[:, :4]
                
            # Re-normalize so rows sum to 1 for OvR
            row_sums = y_prob_4c.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-9, row_sums)
            y_prob_4c = y_prob_4c / row_sums
            
            metrics["auc_ovr"] = float(roc_auc_score(
                y_true_4c, y_prob_4c,
                multi_class="ovr",
                average="macro",
                labels=present_labels,
            ))
        else:
            metrics["auc_ovr"] = 0.0
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}")
        metrics["auc_ovr"] = 0.0

    # Per-class F1 & Recall for N, S, V, F
    f1_per_class = f1_score(y_true, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    rec_per_class = recall_score(y_true, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    for i, cls in enumerate(["N", "S", "V", "F"]):
        metrics[f"f1_{cls}"] = float(f1_per_class[i])
        metrics[f"rec_{cls}"] = float(rec_per_class[i])

    metrics["_preds"]  = preds
    metrics["_y_true"] = y_true
    return metrics

def fgsm_attack_train(model, x, y, epsilon, criterion):
    """Generate FGSM adversarial examples during training."""
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_adv)
        loss, _ = criterion(logits, y, r_fwd=None, r_bwd=None)
        model.zero_grad()
        loss.backward()
    return (x + epsilon * x_adv.grad.sign()).detach()

def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss, n = 0.0, 0

    adv_training = cfg["adversarial_training"]
    adv_epsilon  = cfg["adv_epsilon"]
    adv_ratio    = cfg["adv_ratio"]

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if adv_training and adv_epsilon > 0:
            split  = int(len(X) * (1 - adv_ratio))
            X_adv  = fgsm_attack_train(
                model, X[split:], y[split:], adv_epsilon, criterion
            )
            X = torch.cat([X[:split], X_adv], dim=0)
            y = torch.cat([y[:split], y[split:]], dim=0)

        optimizer.zero_grad()

        logits, internals = model(X, return_internals=True)
        loss, _ = criterion(
            logits, y,
            r_fwd=internals["r_fwd"],
            r_bwd=internals["r_bwd"],
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n          += X.size(0)

    return total_loss / max(1, n)

def train_variant(variant_name, variant_flags, cfg, device,
                  train_loader, val_loader, test_loader,
                  class_weights, out_dir):
    label = variant_flags["label"]
    cfg = {**cfg, **variant_flags.get("override_config", {})}

    print(f"\n{'='*60}")
    print(f"  Variant (Inter-Patient): {label}")
    print(f"  use_rmc={variant_flags['use_rmc']}  "
          f"use_cnn={variant_flags['use_cnn']}  "
          f"use_attention={variant_flags['use_attention']}")
    if "override_config" in variant_flags:
        print(f"  override_config={variant_flags['override_config']}")
    print(f"{'='*60}")

    set_seed(cfg["seed"])

    model = RLSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
        use_rmc=variant_flags["use_rmc"],
        use_cnn=variant_flags["use_cnn"],
        use_attention=variant_flags["use_attention"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    ckpt_path = out_dir / f"best_rlstm_{variant_name}.pt"
    
    # Check if checkpoint already exists and no force-retrain
    force_retrain = "--force-retrain" in sys.argv
    if ckpt_path.exists() and not force_retrain:
        print(f"  Found pre-trained ablation checkpoint: {ckpt_path}. Loading for evaluation...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        best_epoch = ckpt.get("epoch", 0)
        best_f1 = ckpt.get("val_f1_macro", 0.0)
    else:
        criterion = RLSTMLoss(
            lambda_smooth=cfg["lambda_smooth"],
            class_weights=class_weights,
            use_focal=cfg["use_focal_loss"],
            focal_gamma=cfg["focal_gamma"],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        best_f1       = 0.0
        best_epoch    = 0
        patience_cnt  = 0

        for epoch in range(1, cfg["epochs"] + 1):
            lr = cosine_lr(epoch - 1, cfg["epochs"],
                           cfg["learning_rate"], cfg["min_lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            t0         = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, cfg
            )
            val_m = evaluate(model, val_loader, device, cfg["num_classes"])
            elapsed = time.time() - t0

            marker = ""
            if val_m["f1_macro"] > best_f1:
                best_f1, best_epoch, patience_cnt = val_m["f1_macro"], epoch, 0
                torch.save({
                    "model_state":   model.state_dict(),
                    "config":        cfg,
                    "variant_flags": variant_flags,
                    "variant_name":  variant_name,
                    "epoch":         epoch,
                    "val_f1_macro":  best_f1,
                    "n_params":      n_params,
                }, ckpt_path)
                marker = " <-- best"
            else:
                patience_cnt += 1

            print(f"  ep{epoch:3d} | lr={lr:.5f} | loss={train_loss:.4f} | "
                  f"val_f1={val_m['f1_macro']:.4f} "
                  f"acc={val_m['accuracy']:.4f} | {elapsed:.0f}s{marker}")

            if patience_cnt >= cfg["early_stopping_patience"]:
                print(f"\n  [Early stop] ep={epoch} best={best_epoch} F1={best_f1:.4f}")
                break

    # Evaluation on test set
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt, strict=True)
    test_m = evaluate(model, test_loader, device, cfg["num_classes"])

    report = classification_report(
        test_m["_y_true"], test_m["_preds"],
        labels=[0, 1, 2, 3],
        target_names=["N", "S", "V", "F"],
        zero_division=0, digits=4,
    )

    print(f"\n  Test F1 macro (4-class): {test_m['f1_macro']:.4f}")
    print(f"  Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"  Test AUC:      {test_m['auc_ovr']:.4f}")
    print(f"\n  Per-class:\n{report}")

    clean_test = {k: v for k, v in test_m.items() if not k.startswith("_")}

    return {
        "variant":      variant_name,
        "label":        label,
        "n_params":     n_params,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_f1,
        "test_metrics": clean_test,
        "report":       report,
    }

def generate_table(results, out_dir):
    rows = []
    for r in results:
        tm = r["test_metrics"]
        rows.append({
            "Variant":  r["label"],
            "Params":   f"{r['n_params']:,}",
            "Acc":      f"{tm['accuracy']:.4f}",
            "F1-macro": f"{tm['f1_macro']:.4f}",
            "F1-N":     f"{tm.get('f1_N', 0.0):.4f}",
            "F1-S":     f"{tm.get('f1_S', 0.0):.4f}",
            "F1-V":     f"{tm.get('f1_V', 0.0):.4f}",
            "F1-F":     f"{tm.get('f1_F', 0.0):.4f}",
            "Rec-N":    f"{tm.get('rec_N', 0.0):.4f}",
            "Rec-S":    f"{tm.get('rec_S', 0.0):.4f}",
            "Rec-V":    f"{tm.get('rec_V', 0.0):.4f}",
            "Rec-F":    f"{tm.get('rec_F', 0.0):.4f}",
            "AUC":      f"{tm['auc_ovr']:.4f}",
            "Best Ep":  str(r["best_epoch"]),
        })

    cols = ["Variant", "Params", "Acc", "F1-macro", "F1-N", "F1-S", "F1-V", "F1-F", "Rec-N", "Rec-S", "Rec-V", "Rec-F", "AUC"]

    # CSV
    csv_path = out_dir / "ablation_table_inter.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(row[c] for c in cols) + "\n")
    print(f"\n[OK] ablation_table_inter.csv -> {csv_path}")

    # Console print
    print("\n" + "=" * 110)
    print("  ABLATION RESULTS - MIT-BIH ECG (INTER-PATIENT 4-CLASS)")
    print("=" * 110)
    header = f"{'Variant':<25} {'F1-mac':>8} {'F1-N':>7} {'F1-S':>7} {'F1-V':>7} {'F1-F':>7} | {'Rec-S':>7} {'Rec-F':>7} {'AUC':>8}"
    print(header)
    print("-" * 110)
    for row in rows:
        print(
            f"{row['Variant']:<25} {row['F1-macro']:>8} "
            f"{row['F1-N']:>7} {row['F1-S']:>7} {row['F1-V']:>7} {row['F1-F']:>7} | "
            f"{row['Rec-S']:>7} {row['Rec-F']:>7} {row['AUC']:>8}"
        )
    print("=" * 110)

def main():
    parser = argparse.ArgumentParser(description="HMR-BiLSTM Ablation Study on Inter-Patient Split")
    parser.add_argument(
        "--variants", nargs="*",
        choices=["full", "no_rmc", "no_adv"],
        default=["full", "no_rmc", "no_adv"],
        help="Variants to train/evaluate. Default: full, no_rmc, no_adv.",
    )
    parser.add_argument(
        "--epochs", type=int, default=45,
        help="Max training epochs (overrides config)."
    )
    parser.add_argument(
        "--patience", type=int, default=8,
        help="Early stopping patience (overrides config)."
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Force retrain variants even if checkpoint exists."
    )
    args = parser.parse_args()

    cfg = {**BASE_CONFIG}
    cfg["epochs"] = args.epochs
    cfg["early_stopping_patience"] = args.patience

    out_dir = Path("results/ablation/inter")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Copy HMR-BiLSTM (full) checkpoint if it exists in the main checkpoints directory
    main_full_ckpt = Path("results/checkpoints/inter_best_rlstm.pt")
    dest_full_ckpt = ckpt_dir / "best_rlstm_full.pt"
    if main_full_ckpt.exists() and not dest_full_ckpt.exists():
        print(f"  Copying main full model checkpoint to {dest_full_ckpt} to save training time...")
        shutil.copy(main_full_ckpt, dest_full_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Variants to run: {args.variants}")

    print("\n[Loading data]")
    train_loader, val_loader, test_loader = load_data(cfg["data_dir"], cfg["batch_size"])

    # Class weights
    class_weights = None
    if cfg["use_class_weights"]:
        cw_path = Path("data/processed/class_weights.npy")
        if cw_path.exists():
            class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
            print(f"  Class weights: {class_weights.cpu().numpy()}")

    json_path = out_dir / "ablation_results_inter.json"
    all_results = []
    
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"[OK] Loaded {len(all_results)} existing results from {json_path}")
        except Exception as e:
            print(f"[WARNING] Could not load existing results: {e}")

    # Index existing results by variant name
    existing_dict = {r["variant"]: r for r in all_results}

    updated_results = []
    for variant_name in args.variants:
        if variant_name not in VARIANTS:
            continue

        # If it already exists in the JSON, and no force-retrain flag is set, keep it
        force_retrain = "--force-retrain" in sys.argv
        if variant_name in existing_dict and not force_retrain:
            print(f"[KEEP] Using existing result for variant: {variant_name}")
            updated_results.append(existing_dict[variant_name])
            continue

        variant_flags = VARIANTS[variant_name]
        result = train_variant(
            variant_name=variant_name,
            variant_flags=variant_flags,
            cfg=cfg,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            out_dir=ckpt_dir,
        )
        updated_results.append(result)

        # Save immediately to prevent loss in case of crash
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(updated_results, f, indent=2)
        print(f"[OK] Saved intermediate ablation results -> {json_path}")

    if updated_results:
        generate_table(updated_results, out_dir)

if __name__ == "__main__":
    main()
