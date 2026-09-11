"""
Rerun công bằng cho 2 biến thể DSTA-Lite tốt nhất (C, D) từ
`ablation_dsta_variants.py`, sau khi audit phát hiện recipe cũ (epochs=30,
patience=6, không LR schedule, không adversarial training) KHÔNG khớp với
recipe thật của checkpoint ResMemECG chính thức
(results/checkpoints/inter_best_rlstm.pt): epochs=45, patience=15,
LR->min_lr=1e-5, adversarial_training=True (FGSM epsilon=0.02, ratio=0.3).

Kết quả ablation cũ (chưa công bằng, để tham chiếu):
  A: F1_macro=0.4693  F1-F=0.0000
  B: F1_macro=0.5333  F1-F=0.1573
  C: F1_macro=0.6357  F1-F=0.4322   <- tốt nhất, đã vượt official (0.6279)
  D: F1_macro=0.5440  F1-F=0.2687   <- nhì

Theo kế hoạch đã chốt: chỉ rerun C và D với recipe công bằng, xem có thật sự
vượt official hay chênh lệch cũ một phần do recipe không công bằng.
Nếu sau rerun công bằng mà CẢ 2 vẫn quá thấp (model không học được gì đáng kể
ở lớp hiếm) -> bỏ hẳn nhánh DSTA-Lite, chuyển sang camp_net.py.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hmr_dsta_lite import DSTALiteClassifier
from hmr_bilstm import RLSTMLoss

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# --- Recipe khớp ĐÚNG checkpoint chính thức (results/checkpoints/inter_best_rlstm.pt) ---
EPOCHS = 45
PATIENCE = 15
BATCH_SIZE = 128
SEED = 42
ADV_EPSILON = 0.02
ADV_RATIO = 0.3
EVAL_BATCH_SIZE = 256

DATA_DIR = Path("data/processed/splits")

VARIANTS = {
    "C_perdim_5edge_softmax": dict(blend_mode="per_dim", extra_edge=True, gamma_mode="softmax"),
    # D bị loại theo quyết định của user (2026-08-21): chỉ theo đuổi C, D không
    # đủ hứa hẹn để đáng thêm ~8-13h GPU-time cho việc rerun công bằng.
}

OLD_UNFAIR_RESULTS = {
    "A_scalar_4edge_softmax": {"f1_macro": 0.4693, "f1_F": 0.0000},
    "B_perdim_4edge_softmax": {"f1_macro": 0.5333, "f1_F": 0.1573},
    "C_perdim_5edge_softmax": {"f1_macro": 0.6357, "f1_F": 0.4322},
    "D_perdim_5edge_sigmoid": {"f1_macro": 0.5440, "f1_F": 0.2687},
}


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


@torch.no_grad()
def predict_in_batches(model, X_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        preds.append(model(xb).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def fgsm_adversarial_batch(model, criterion, x, y, epsilon):
    """FGSM 1 bước, khớp adv_epsilon=0.02 của checkpoint chính thức."""
    x_req = x.clone().detach().requires_grad_(True)
    logits, internals = model(x_req, return_internals=True)
    loss, _ = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
    grad = torch.autograd.grad(loss, x_req)[0]
    x_adv = (x_req + epsilon * grad.sign()).detach()
    return x_adv


def train_and_eval(name, variant_kwargs, X_train, y_train, X_val, y_val, X_test, y_test, device):
    print(f"\n{'='*70}\n[{name}] (FAIR RECIPE)  kwargs={variant_kwargs}\n{'='*70}")
    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = DSTALiteClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES,
        cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2, rank=4,
        **variant_kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
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
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)

            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])

            if ADV_RATIO > 0:
                n_adv = max(int(len(xb) * ADV_RATIO), 1)
                adv_idx = torch.randperm(len(xb), device=device)[:n_adv]
                x_adv = fgsm_adversarial_batch(model, criterion, xb[adv_idx], yb[adv_idx], ADV_EPSILON)
                logits_adv, internals_adv = model(x_adv, return_internals=True)
                loss_adv, _ = criterion(logits_adv, yb[adv_idx], internals_adv["r_fwd"], internals_adv["r_bwd"])
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
        results[name] = train_and_eval(name, kwargs, X_train, y_train, X_val, y_val,
                                        X_test, y_test, device)

    official = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}

    print("\n" + "=" * 100)
    print(" SUMMARY: DSTA-Lite (C, D) recipe cong bang vs cu vs official ")
    print("=" * 100)
    print(f"{'Variant':<28} {'F1_macro cu':>12} {'F1_macro moi':>13} {'F1-F cu':>9} {'F1-F moi':>10}")
    print(f"{'ResMemECG (official)':<28} {'-':>12} {official['f1_macro']:>13.4f} "
          f"{'-':>9} {official['f1_per_class']['F']:>10.4f}")
    for name, r in results.items():
        old = OLD_UNFAIR_RESULTS[name]
        print(f"{name:<28} {old['f1_macro']:>12.4f} {r['f1_macro']:>13.4f} "
              f"{old['f1_F']:>9.4f} {r['f1_per_class']['F']:>10.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_ablation_fair_rerun_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": official, "old_unfair_results": OLD_UNFAIR_RESULTS,
                    "fair_rerun_variants": results}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
