"""
Tach bach 2 co che dang xung dot nhau trong CAMPLoss, phat hien sau lan chay
that dau tien cua camp_net.py (F1-F sap ve 0.0023):

  - Confusion-Margin: ep logit(F) > logit(N), logit(V) -- muon F TACH XA
    N va V trong khong gian quyet dinh.
  - Mixture-Decomposition: ep embedding cua F giai thich duoc bang to hop
    loi cua N va V -- muon F NAM GAN/GIUA N va V trong khong gian embedding.

2 luc nay keo NGUOC CHIEU nhau. Prototype-Repulsion cung thuoc phe "tach xa"
(giong Confusion-Margin ve huong), nen TAT repulsion o ca 3 bien the duoi day
de co lap dung 1 cau hoi: Confusion-Margin va Mixture-Decomposition, cai nao
that su co ich, cai nao dang pha nhau.

3 bien the, cung kien truc CAMPClassifier, cung recipe (epochs=45,
patience=15, LR scheduler, FGSM adv training) nhu lan chay CAMP-Net dau tien
(results/logs/camp_net_results.json, F1_macro=0.4952, F1-F=0.0023):

  A_cm_only    : chi Confusion-Margin  (lambda_mix=0, lambda_rep=0)
  B_mixture_only: chi Mixture-Decomposition (lambda_cm=0, lambda_rep=0)
  C_neither    : tat ca =0 -- baseline "sach", chi con kien truc
                 (prototype classifier + mixture head ton tai nhung khong
                 co loss nao ep dung) + task loss (Focal) -- de biet
                 CAMPClassifier's backbone/prototype-gate tu no cho ra sao
                 khi khong bi ep boi loss phu tro nao ca.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from camp_net import (
    CAMPClassifier, CAMPLoss, fgsm_adversarial_batch, predict_in_batches,
    load_split, class_weights_from, CLASS_NAMES,
    EPOCHS, PATIENCE, BATCH_SIZE, SEED, ADV_EPSILON, ADV_RATIO,
)

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

VARIANTS = {
    "A_cm_only":      dict(lambda_cm=0.3, lambda_mix=0.0, lambda_rep=0.0),
    "B_mixture_only": dict(lambda_cm=0.0, lambda_mix=0.1, lambda_rep=0.0),
    "C_neither":      dict(lambda_cm=0.0, lambda_mix=0.0, lambda_rep=0.0),
}

COMBINED_ALL_THREE = {"f1_macro": 0.4952, "f1_per_class": {"N": 0.9208, "S": 0.2402, "V": 0.8175, "F": 0.0023}}
OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}


def train_one_variant(name, lambda_kwargs, X_train, y_train, X_val, y_val, X_test, y_test, device):
    print(f"\n{'='*70}\n[{name}]  {lambda_kwargs}\n{'='*70}")
    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = CAMPClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
        cnn_out_channels=64, num_layers=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = CAMPLoss(class_weights=cw, focal_gamma=1.5, **lambda_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5,
    )

    Xtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xva_t, yva_t = torch.from_numpy(X_val), torch.from_numpy(y_val)
    Xte_t, yte_t = torch.from_numpy(X_test), torch.from_numpy(y_test)

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        criterion.set_epoch(epoch)
        perm = torch.randperm(n_train)

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)

            optimizer.zero_grad()
            logits, internals = model(xb, labels=yb, update_prototypes=True, return_internals=True)
            loss, comps = criterion(logits, yb, internals)

            if ADV_RATIO > 0:
                n_adv = max(int(len(xb) * ADV_RATIO), 1)
                adv_idx = torch.randperm(len(xb), device=device)[:n_adv]
                x_adv = fgsm_adversarial_batch(model, criterion, xb[adv_idx], yb[adv_idx], ADV_EPSILON)
                logits_adv, internals_adv = model(
                    x_adv, labels=yb[adv_idx], update_prototypes=False, return_internals=True,
                )
                loss_adv, _ = criterion(logits_adv, yb[adv_idx], internals_adv)
                loss = loss + loss_adv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, yb, logits, internals, loss

        val_preds = predict_in_batches(model, Xva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f} | lr={lr_now:.2e}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [early stop] het patience={PATIENCE} o epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    test_preds = predict_in_batches(model, Xte_t, device)

    y_true = yte_t.numpy()
    acc = accuracy_score(y_true, test_preds)
    f1_macro = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    print(f"\n  [{name}] best val F1_macro={best_f1:.4f}  train_time={train_time:.1f}s")
    print(f"  [{name}] TEST accuracy={acc:.4f}  F1_macro={f1_macro:.4f}")
    print(f"  [{name}] TEST per-class F1: " +
          "  ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.4f}" for i in range(4)))
    print(classification_report(y_true, test_preds, labels=[0, 1, 2, 3],
                                 target_names=["N", "S", "V", "F"], zero_division=0, digits=4))

    return {
        "n_params": n_params, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    results = {}
    for name, kwargs in VARIANTS.items():
        results[name] = train_one_variant(name, kwargs, X_train, y_train, X_val, y_val,
                                           X_test, y_test, device)

    print("\n" + "=" * 100)
    print(" SUMMARY: CAMP-Net component ablation ")
    print("=" * 100)
    print(f"{'Variant':<24} {'F1_macro':>10} {'F1-N':>8} {'F1-S':>8} {'F1-V':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<24} {OFFICIAL['f1_macro']:>10.4f} "
          f"{OFFICIAL['f1_per_class']['N']:>8.4f} {OFFICIAL['f1_per_class']['S']:>8.4f} "
          f"{OFFICIAL['f1_per_class']['V']:>8.4f} {OFFICIAL['f1_per_class']['F']:>8.4f}")
    print(f"{'CAMP-Net (ca 3 co che)':<24} {COMBINED_ALL_THREE['f1_macro']:>10.4f} "
          f"{COMBINED_ALL_THREE['f1_per_class']['N']:>8.4f} {COMBINED_ALL_THREE['f1_per_class']['S']:>8.4f} "
          f"{COMBINED_ALL_THREE['f1_per_class']['V']:>8.4f} {COMBINED_ALL_THREE['f1_per_class']['F']:>8.4f}")
    for name, r in results.items():
        fc = r["f1_per_class"]
        print(f"{name:<24} {r['f1_macro']:>10.4f} {fc['N']:>8.4f} {fc['S']:>8.4f} {fc['V']:>8.4f} {fc['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/camp_net_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": OFFICIAL, "camp_net_all_three": COMBINED_ALL_THREE,
                    "variants": results}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
