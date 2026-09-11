"""
Shared trainer for the modern-architecture benchmark.

Protocol is intentionally identical to ../train.py (focal loss, class weights,
cosine LR, FGSM adversarial training, grad clip, early stopping on val F1-macro)
so that swapping --model is the *only* variable between runs. Reads
../data/processed/*.npz read-only; writes only inside this folder.

Usage:
    python train_modern.py --model bilstm --seed 42 --out-dir results/bilstm_s42
    python train_modern.py --model cnn_mamba --seed 42 --quick   # 3-epoch smoke run
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from pathlib import Path

from models import build_model

MODEL_KWARGS = {
    "bilstm":           dict(hidden_size=96, cnn_out_channels=64, num_layers=2),
    "cnn_transformer":  dict(d_model=128, nhead=4, num_layers=4, dim_feedforward=256, cnn_out_channels=64),
    "cnn_mamba":        dict(d_model=128, d_state=16, num_layers=4),
    "cnn_xlstm":        dict(hidden_size=96, cnn_out_channels=64, num_layers=2),
    "pure_transformer": dict(d_model=128, nhead=4, num_layers=6, dim_feedforward=256, patch_size=11),
}


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    progress = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        ce_raw = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_raw)
        focal = ((1.0 - pt) ** self.gamma) * ce_raw
        if self.alpha is not None:
            focal = focal * self.alpha[targets]
        return focal.mean() if self.reduction == "mean" else focal.sum()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        all_logits.append(model(X.to(device)).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()

    metrics = {
        "accuracy":        accuracy_score(y_true, preds),
        "precision_macro": precision_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "f1_weighted":     f1_score(y_true, preds, labels=[0, 1, 2, 3], average="weighted", zero_division=0),
    }
    for i, cls in enumerate(["N", "S", "V", "F", "Q"]):
        metrics[f"f1_{cls}"] = f1_score(y_true, preds, labels=[i], average="macro", zero_division=0)
    try:
        metrics["auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_ovr"] = 0.0
    return metrics, y_true, preds


def fgsm_attack_train(model, x, y, epsilon, criterion):
    x_adv = x.clone().detach().requires_grad_(True)
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d,))]
    for bn in bn_layers:
        bn.eval()
    with torch.enable_grad():
        loss = criterion(model(x_adv), y)
        loss.backward()
    for bn in bn_layers:
        bn.train()
    perturbation = epsilon * x_adv.grad.sign()
    return (x + perturbation).detach()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip,
                     adv_training, adv_epsilon, adv_ratio):
    model.train()
    total_loss, n_samples, n_nan = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if adv_training and adv_epsilon > 0:
            split = int(len(X) * (1 - adv_ratio))
            X_clean, y_clean = X[:split], y[:split]
            X_adv_src, y_adv = X[split:], y[split:]
            if len(X_adv_src) > 0:
                X_adv = fgsm_attack_train(model, X_adv_src, y_adv, adv_epsilon, criterion)
                X = torch.cat([X_clean, X_adv], dim=0)
                y = torch.cat([y_clean, y_adv], dim=0)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        if torch.isnan(loss) or torch.isinf(loss):
            n_nan += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    if n_nan:
        print(f"  [WARNING] {n_nan} batch(es) skipped due to NaN/Inf loss.")
    return total_loss / max(1, n_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_KWARGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--adv-epsilon", type=float, default=0.02)
    p.add_argument("--adv-ratio", type=float, default=0.3)
    p.add_argument("--no-adv-training", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--data-dir", default="../data/processed/splits")
    p.add_argument("--split-prefix", default="inter_",
                   help="Filename prefix: inter_ (patient-disjoint, correct protocol) or intra_ (leaky, do not use for reported results)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--quick", action="store_true", help="3-epoch smoke run")
    args = p.parse_args()

    if args.quick:
        args.epochs, args.patience = 3, 3
    out_dir = Path(args.out_dir or f"results/{args.model}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model} | Seed: {args.seed} | Device: {device}")

    train_ds = ECGDataset(f"{args.data_dir}/{args.split_prefix}train.npz")
    val_ds = ECGDataset(f"{args.data_dir}/{args.split_prefix}val.npz")
    test_ds = ECGDataset(f"{args.data_dir}/{args.split_prefix}test.npz")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Computed from train split itself (matches ../train_inter_patient.py's formula)
    # rather than the root data/processed/class_weights.npy, which was fit on the
    # leaky non-inter-patient split and would not match this split's class counts.
    class_weights = None
    if not args.no_class_weights:
        counts = np.bincount(train_ds.y.numpy(), minlength=5).astype(np.float64)
        counts = np.where(counts == 0, 1e-9, counts)
        cw_arr = counts.sum() / (5.0 * counts)
        cw_arr = np.clip(cw_arr, 0.5, 50.0).astype(np.float32)
        class_weights = torch.from_numpy(cw_arr).float().to(device)
        print(f"  Class weights ({args.split_prefix}train): {cw_arr}")

    model = build_model(
        args.model, num_classes=5, dropout=args.dropout, **MODEL_KWARGS[args.model]
    ).to(device)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,}")

    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    adv_training = not args.no_adv_training
    best_f1, best_epoch, patience_counter = float("-inf"), 0, 0
    history = []
    ckpt_path = out_dir / "best.pt"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(epoch - 1, max(1, args.epochs - 1), args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip,
            adv_training, args.adv_epsilon, args.adv_ratio,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                         **{f"val_{k}": v for k, v in val_metrics.items()}, "time_sec": elapsed})

        marker = ""
        if val_metrics["f1_macro"] > best_f1:
            best_f1, best_epoch, patience_counter = val_metrics["f1_macro"], epoch, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val_f1_macro": best_f1}, ckpt_path)
            marker = " <-- best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d} | lr={lr:.5f} | loss={train_loss:.4f} | "
              f"val: acc={val_metrics['accuracy']:.4f} f1_mac={val_metrics['f1_macro']:.4f} "
              f"auc={val_metrics['auc_ovr']:.4f} | {elapsed:.0f}s{marker}")

        if patience_counter >= args.patience:
            print(f"[Early stop] epoch {epoch} (best={best_epoch}, F1={best_f1:.4f})")
            break

    total_time = time.time() - t_start

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    test_metrics, y_true, preds = evaluate(model, test_loader, device)
    report = classification_report(
        y_true, preds, target_names=["N", "S", "V", "F", "Q"], zero_division=0, digits=4,
    )
    print("\n[Test metrics]")
    for k, v in test_metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\n[Per-class report]\n" + report)

    with open(out_dir / "log.json", "w") as f:
        json.dump({
            "args": vars(args), "n_params": n_params, "total_time_sec": total_time,
            "history": history, "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
            "test_metrics": test_metrics, "classification_report": report,
        }, f, indent=2)
    print(f"\n[OK] Log -> {out_dir / 'log.json'}")


if __name__ == "__main__":
    main()
