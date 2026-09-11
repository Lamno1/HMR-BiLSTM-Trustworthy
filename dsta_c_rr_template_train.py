"""
DSTA bien the C + 4 dac trung phu tro noi shallow truoc classifier:
  3 dac trung RR-interval cu (pre_rr_ratio, post_rr_ratio, local_rr_ratio)
  + 1 dac trung MOI: tuong quan hinh dang voi "template N" (thay cho
    QRS-width proxy cu) -- do DUNG dinh nghia lam sang cua nhip F (Fusion):
    F la nhip HOA TRON hinh dang giua N va V, nen do do giong hinh dang voi
    nhip N chuan la dac trung truc tiep, khong phai doan mo nhu QRS-width.

  Da kiem chung bang thong ke that (Mann-Whitney AUC, template fit CHI tren
  train, danh gia tren val/test that -- khong ro ri):
    test set (n_F=388): V-vs-F AUC=0.8834 (QRS-width cu chi 0.6580)
                         S-vs-F AUC=0.9005
                         N-vs-F AUC=0.7146 (QRS-width cu 0.6848)
  Day la dac trung manh nhat tung tim duoc cho dung cap lop gay kho (F vs
  V/S) -- ky vong khong chi tang F1-F ma con giam bat on dinh theo seed cua
  thi nghiem C+RR+QRS truoc do (0.6900 o seed 42 vs 0.6055 o seed 123).

Template: trung binh cac nhip N trong TRAIN, chuan hoa L2. He so tuong quan
= cosine similarity giua tung nhip (da chuan hoa L2) va template nay.
Chuan hoa (z-score) gia tri tuong quan cung fit tren train, ap lai cho
val/test giong het pattern qrs_mean/qrs_std cu.

Tai su dung nguyen DSTALiteClassifierRRShallow (rr_dim=4) -- khong doi
kien truc, chi doi cach tinh dac trung phu tro thu 4.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from dsta_c_rr_shallow import DSTALiteClassifierRRShallow, NUM_CLASSES, CLASS_NAMES, S_IDX
from hmr_bilstm import RLSTMLoss

_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--out-suffix", type=str, default="")
_args = _parser.parse_args()

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = _args.seed
EVAL_BATCH_SIZE = 256
N_IDX = 0

DATA_DIR = Path("data/processed/splits_rr")

OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}
C_NO_RR = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}
C_RR3 = {"f1_macro": 0.6061, "f1_per_class": {"N": 0.9682, "S": 0.4849, "V": 0.8516, "F": 0.1198}}
C_RR3_QRS1_SEED42 = {"f1_macro": 0.6900}
C_RR3_QRS1_SEED123 = {"f1_macro": 0.6055309782764909, "f1_per_class": {"N": 0.9564, "S": 0.4995, "V": 0.8527, "F": 0.1136}}


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def compute_template(X_train, y_train, n_idx=N_IDX):
    """Trung binh cac nhip N trong TRAIN, chuan hoa L2. CHI dung train --
    tranh ro ri sang val/test."""
    x = X_train.squeeze(-1) if X_train.ndim == 3 else X_train
    template = x[y_train == n_idx].mean(axis=0)
    return (template / (np.linalg.norm(template) + 1e-8)).astype(np.float32)


def template_corr_batch(X, template_norm):
    """Cosine similarity giua tung nhip (da chuan hoa L2) va template N."""
    x = X.squeeze(-1) if X.ndim == 3 else X
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return (xn @ template_norm).astype(np.float32)


def build_rr_template4(X, rr, template_norm, corr_mean, corr_std, fit_stats=False):
    corr = template_corr_batch(X, template_norm)
    if fit_stats:
        corr_mean = corr.mean()
        corr_std = corr.std() + 1e-8
    corr_norm = (corr - corr_mean) / corr_std
    rr4 = np.concatenate([rr, corr_norm[:, None].astype(np.float32)], axis=-1)
    return rr4, corr_mean, corr_std


@torch.no_grad()
def predict_in_batches(model, X_t, RR_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        rrb = RR_t[i:i + EVAL_BATCH_SIZE].to(device)
        preds.append(model(xb, rr=rrb).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  SEED={SEED}")

    X_train, RR_train, y_train = load_split("train")
    X_val, RR_val, y_val = load_split("val")
    X_test, RR_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    print("Tinh dac trung template-correlation (thay QRS-width)...")
    template_norm = compute_template(X_train, y_train)
    RR4_train, corr_mean, corr_std = build_rr_template4(X_train, RR_train, template_norm, None, None, fit_stats=True)
    RR4_val, _, _ = build_rr_template4(X_val, RR_val, template_norm, corr_mean, corr_std)
    RR4_test, _, _ = build_rr_template4(X_test, RR_test, template_norm, corr_mean, corr_std)
    print(f"  template-corr train stats: mean={corr_mean:.4f} std={corr_std:.4f}")
    print(f"  RR4 shape: {RR4_train.shape} (3 RR-ratio + 1 template-corr)")

    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = DSTALiteClassifierRRShallow(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES,
        cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2, rank=4,
        rr_dim=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (C+RR3: 607,374)")

    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, RRtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(RR4_train), torch.from_numpy(y_train)
    Xva_t, RRva_t, yva_t = torch.from_numpy(X_val), torch.from_numpy(RR4_val), torch.from_numpy(y_val)
    Xte_t, RRte_t, yte_t = torch.from_numpy(X_test), torch.from_numpy(RR4_test), torch.from_numpy(y_test)

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            rrb = RRtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            logits, internals = model(xb, rr=rrb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, rrb, yb, logits, internals, loss

        val_preds = predict_in_batches(model, Xva_t, RRva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        val_f1_s = f1_score(yva_t.numpy(), val_preds, labels=[S_IDX], average=None, zero_division=0)[0]
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f} | val F1-S = {val_f1_s:.4f}")

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
    test_preds = predict_in_batches(model, Xte_t, RRte_t, device)

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
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1), "seed": SEED,
    }

    print("\n" + "=" * 100)
    print(" SUMMARY: DSTA-C + RR(3) + Template-corr(1) vs cac moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<32} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<32} {OFFICIAL['f1_macro']:>10.4f} "
          f"{OFFICIAL['f1_per_class']['S']:>8.4f} {OFFICIAL['f1_per_class']['F']:>8.4f}")
    print(f"{'C (khong RR)':<32} {C_NO_RR['f1_macro']:>10.4f} "
          f"{C_NO_RR['f1_per_class']['S']:>8.4f} {C_NO_RR['f1_per_class']['F']:>8.4f}")
    print(f"{'C + RR(3) + QRS(1) seed42':<32} {C_RR3_QRS1_SEED42['f1_macro']:>10.4f}")
    print(f"{'C + RR(3) + QRS(1) seed123':<32} {C_RR3_QRS1_SEED123['f1_macro']:>10.4f} "
          f"{C_RR3_QRS1_SEED123['f1_per_class']['S']:>8.4f} {C_RR3_QRS1_SEED123['f1_per_class']['F']:>8.4f}")
    print(f"{'C + RR(3) + Template(1) seed=' + str(SEED):<32} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    suffix = f"_{_args.out_suffix}" if _args.out_suffix else ""
    out_path = Path(f"results/logs/dsta_c_rr_template_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": OFFICIAL, "C_no_rr": C_NO_RR, "C_rr3": C_RR3,
                    "C_rr3_qrs1_seed42": C_RR3_QRS1_SEED42, "C_rr3_qrs1_seed123": C_RR3_QRS1_SEED123,
                    "C_rr3_template1": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
