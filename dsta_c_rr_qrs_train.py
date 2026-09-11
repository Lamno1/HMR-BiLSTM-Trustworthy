"""
DSTA bien the C + 4 dac trung phu tro noi shallow truoc classifier:
  3 dac trung RR-interval cu (pre_rr_ratio, post_rr_ratio, local_rr_ratio --
    do THOI GIAN, da xac nhan giup F1-S nhung khong tach duoc S khoi V vi
    ca 2 deu la nhip den som)
  + 1 dac trung MOI: do rong QRS-proxy (do HINH DANG, khong phai thoi gian)
    -- da kiem chung bang thong ke that (AUC(V,S)=0.80 o thresh=0.6-0.8,
    xem phan phan tich truoc khi code file nay) -- ky vong tach duoc dung
    cap S/V ma RR khong tach duoc.

Tai su dung nguyen DSTALiteClassifierRRShallow (rr_dim=4 thay vi 3) --
khong doi kien truc, chi doi so chieu input phu tro.
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
QRS_THRESH_FRAC = 0.6  # dinh AUC(V,S) o vung 0.6-0.8, xem phan phan tich CPU truoc do

DATA_DIR = Path("data/processed/splits_rr")

C_NO_RR = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}
C_RR3 = {"f1_macro": 0.6061, "f1_per_class": {"N": 0.9682, "S": 0.4849, "V": 0.8516, "F": 0.1198}}
OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def qrs_width_batch(X, thresh_frac=QRS_THRESH_FRAC):
    """X: (N, 187, 1) hoac (N, 187). Tra ve (N,) so diem lien tuc vuot nguong
    quanh dinh R -- proxy do rong QRS. Vector hoa qua numpy, khong loop
    Python tung sample de nhanh voi ~40k+ mau."""
    x = X.squeeze(-1) if X.ndim == 3 else X
    n, T = x.shape
    peak_idx = np.argmax(np.abs(x), axis=1)
    peak_amp = np.abs(x[np.arange(n), peak_idx])
    thresh = (thresh_frac * peak_amp)[:, None]
    above = np.abs(x) > thresh  # (N, T)

    widths = np.zeros(n, dtype=np.float32)
    for i in range(n):  # T=187 nho, vong lap ngoai van du nhanh (~vai giay cho 40k mau)
        row = above[i]
        p = peak_idx[i]
        left = p
        while left > 0 and row[left - 1]:
            left -= 1
        right = p
        while right < T - 1 and row[right + 1]:
            right += 1
        widths[i] = right - left + 1
    return widths


def build_rr4(X, rr, qrs_mean, qrs_std, fit_stats=False):
    qrs = qrs_width_batch(X)
    qrs = np.clip(qrs, 1, 60)  # cat outlier (vd nhieu/artifact) truoc khi chuan hoa
    if fit_stats:
        qrs_mean = qrs.mean()
        qrs_std = qrs.std() + 1e-8
    qrs_norm = (qrs - qrs_mean) / qrs_std
    rr4 = np.concatenate([rr, qrs_norm[:, None].astype(np.float32)], axis=-1)
    return rr4, qrs_mean, qrs_std


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

    print("Tinh dac trung do rong QRS-proxy...")
    RR4_train, qrs_mean, qrs_std = build_rr4(X_train, RR_train, None, None, fit_stats=True)
    RR4_val, _, _ = build_rr4(X_val, RR_val, qrs_mean, qrs_std)
    RR4_test, _, _ = build_rr4(X_test, RR_test, qrs_mean, qrs_std)
    print(f"  qrs_width train stats: mean={qrs_mean:.2f} std={qrs_std:.2f}")
    print(f"  RR4 shape: {RR4_train.shape} (3 RR-ratio + 1 QRS-width)")

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
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1),
    }

    print("\n" + "=" * 100)
    print(" SUMMARY: DSTA-C + RR(3) + QRS-width(1) vs cac moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<28} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<28} {OFFICIAL['f1_macro']:>10.4f} "
          f"{OFFICIAL['f1_per_class']['S']:>8.4f} {OFFICIAL['f1_per_class']['F']:>8.4f}")
    print(f"{'C (khong RR)':<28} {C_NO_RR['f1_macro']:>10.4f} "
          f"{C_NO_RR['f1_per_class']['S']:>8.4f} {C_NO_RR['f1_per_class']['F']:>8.4f}")
    print(f"{'C + RR(3 -- chi RR-ratio)':<28} {C_RR3['f1_macro']:>10.4f} "
          f"{C_RR3['f1_per_class']['S']:>8.4f} {C_RR3['f1_per_class']['F']:>8.4f}")
    print(f"{'C + RR(3) + QRS-width(1)':<28} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    suffix = f"_{_args.out_suffix}" if _args.out_suffix else ""
    out_path = Path(f"results/logs/dsta_c_rr_qrs_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": OFFICIAL, "C_no_rr": C_NO_RR, "C_rr3": C_RR3,
                    "C_rr3_qrs1": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
