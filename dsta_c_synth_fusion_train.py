"""
Train DSTA-C + tong hop F gia lap (tron N that + V that) + dau alpha co giam
sat. Xem dsta_c_synth_fusion.py de biet kien truc/ly do.

Da kiem chung truoc bang CPU: AUC(F gia lap vs F that ve QRS-width) = 0.527
(~0.5 = khong phan biet duoc) -- du lieu tong hop hop ly ve mat thong ke.

Dung recipe nhanh cua C (epochs=30, patience=6, khong adv training) --
single-variable sach so voi C khong tang cuong (F1_macro=0.6357,
F1-S=0.2547, F1-F=0.4322). Val/test CHI dung du lieu that, khong co mau
gia lap -- danh gia dung hieu nang thuc te.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from dsta_c_synth_fusion import (
    DSTALiteClassifierSynthFusion, SynthFusionLoss, synthesize_fusion_batch,
    class_weights_from, NUM_CLASSES, CLASS_NAMES, F_IDX,
)

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = 42
EVAL_BATCH_SIZE = 256
N_SYNTH = 1000  # giam tu 5000 (v1) -- giu du lieu that ap dao (~72% F van la that)
ALPHA_LO, ALPHA_HI = 0.3, 0.7

DATA_DIR = Path("data/processed/splits")

C_NO_AUG = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}
OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


@torch.no_grad()
def predict_in_batches(model, X_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        preds.append(model(xb).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train that={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    rng = np.random.default_rng(SEED)
    X_synth, y_synth, alpha_synth = synthesize_fusion_batch(
        X_train, y_train, n_synth=N_SYNTH, rng=rng, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
    )
    print(f"Da tong hop {N_SYNTH} mau F gia lap (alpha~U({ALPHA_LO},{ALPHA_HI}))")

    X_aug = np.concatenate([X_train, X_synth], axis=0)
    y_aug = np.concatenate([y_train, y_synth], axis=0)
    alpha_aug = np.concatenate([np.zeros(len(X_train), dtype=np.float32), alpha_synth], axis=0)
    is_synth_aug = np.concatenate(
        [np.zeros(len(X_train), dtype=bool), np.ones(N_SYNTH, dtype=bool)], axis=0,
    )
    print(f"Train sau tang cuong={X_aug.shape}  "
          f"F count: that={int((y_train==F_IDX).sum())} -> sau tang cuong={int((y_aug==F_IDX).sum())}")

    cw = class_weights_from(y_aug).to(device)
    print(f"Class weights (sau tang cuong): {cw.cpu().numpy()}")

    torch.manual_seed(SEED)
    model = DSTALiteClassifierSynthFusion(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES,
        cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2, rank=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (C goc: 603,118)")

    criterion = SynthFusionLoss(lambda_smooth=0.003, class_weights=cw, focal_gamma=1.5, lambda_alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(X_aug)
    ytr_t = torch.from_numpy(y_aug)
    alphatr_t = torch.from_numpy(alpha_aug)
    is_synth_t = torch.from_numpy(is_synth_aug)
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
            alphab = alphatr_t[idx].to(device)
            is_synth_b = is_synth_t[idx].to(device)

            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals, alphab, is_synth_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, yb, alphab, is_synth_b, logits, internals, loss

        val_preds = predict_in_batches(model, Xva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

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

    print(f"\n  best val F1_macro={best_f1:.4f}  train_time={train_time:.1f}s")
    print(f"  TEST accuracy={acc:.4f}  F1_macro={f1_macro:.4f}")
    print(f"  TEST per-class F1: " + "  ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.4f}" for i in range(4)))
    print(classification_report(y_true, test_preds, labels=[0, 1, 2, 3],
                                 target_names=["N", "S", "V", "F"], zero_division=0, digits=4))

    result = {
        "n_params": n_params, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1),
    }

    print("\n" + "=" * 100)
    print(" SUMMARY: DSTA-C + Synthetic-Fusion augmentation vs cac moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<28} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<28} {OFFICIAL['f1_macro']:>10.4f} "
          f"{OFFICIAL['f1_per_class']['S']:>8.4f} {OFFICIAL['f1_per_class']['F']:>8.4f}")
    print(f"{'C (khong tang cuong)':<28} {C_NO_AUG['f1_macro']:>10.4f} "
          f"{C_NO_AUG['f1_per_class']['S']:>8.4f} {C_NO_AUG['f1_per_class']['F']:>8.4f}")
    print(f"{'C + Synthetic-Fusion':<28} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_c_synth_fusion_v2_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": OFFICIAL, "C_no_aug": C_NO_AUG, "C_synth_fusion": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
