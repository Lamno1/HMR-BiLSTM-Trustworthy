"""
Huan luyen DSTADualBranch: 2 nhanh CNN doc lap (MLII rieng, kenh phu rieng),
hop nhat o tang dac trung truoc 1 cong DSTA duy nhat + balance regularizer.
Xem dsta_c_dualbranch.py de biet ly do thiet ke.

Danh gia BANG CA ma tran nham lan day du (khong chi F1 tung lop) -- de do
truc tiep ty le sai sot TRON LOP giua tung cap, dung theo yeu cau: "phan
lop hoan hao va ty le sai sot tron lop thap xuong muc toi thieu".

Du lieu: data/processed/splits_rr_2ch/ (da co san tu buoc truoc).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

from dsta_c_dualbranch import DSTADualBranch, DualBranchLoss, class_weights_from, NUM_CLASSES, CLASS_NAMES, N_IDX
from dsta_c_rr_template_2ch_train import compute_template_2ch, build_rr_template_2ch

_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--out-suffix", type=str, default="")
_parser.add_argument("--lambda-balance", type=float, default=0.01)
_args = _parser.parse_args()

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = _args.seed
EVAL_BATCH_SIZE = 256

DATA_DIR = Path("data/processed/splits_rr_2ch")

OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}
C_NO_RR = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}
C_RR_TEMPLATE_1CH_AVG3SEED = {"f1_macro": 0.5808, "f1_per_class": {"N": 0.9431, "S": 0.5315, "V": 0.7882, "F": 0.0606}}
C_2CH_FULL_AVG3SEED = {"f1_macro": 0.528, "f1_per_class": {"N": 0.9367, "S": 0.2129, "V": 0.8186, "F": 0.1439}}


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)


@torch.no_grad()
def predict_in_batches(model, X_t, RR_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        rrb = RR_t[i:i + EVAL_BATCH_SIZE].to(device)
        preds.append(model(xb, rr=rrb).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\n  Ma tran nham lan (hang=that, cot=du doan), % theo hang:")
    header = "        " + "".join(f"{c:>8}" for c in class_names)
    print(header)
    for i, cname in enumerate(class_names):
        row_total = cm[i].sum()
        pct_row = cm[i] / row_total * 100 if row_total > 0 else np.zeros(len(class_names))
        row_str = "".join(f"{pct_row[j]:7.2f}%" for j in range(len(class_names)))
        print(f"  {cname:<6}{row_str}   (n={row_total})")
    return cm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  SEED={SEED}  lambda_balance={_args.lambda_balance}")

    X_train, RR_train, y_train = load_split("train")
    X_val, RR_val, y_val = load_split("val")
    X_test, RR_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    print("Tinh dac trung template-correlation rieng cho moi kenh...")
    template_norm = compute_template_2ch(X_train, y_train)
    RR5_train, corr_mean, corr_std = build_rr_template_2ch(X_train, RR_train, template_norm, None, None, fit_stats=True)
    RR5_val, _, _ = build_rr_template_2ch(X_val, RR_val, template_norm, corr_mean, corr_std)
    RR5_test, _, _ = build_rr_template_2ch(X_test, RR_test, template_norm, corr_mean, corr_std)

    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = DSTADualBranch(
        hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES, cnn_out_channels=64,
        num_layers=2, attn_dim=24, num_heads=2, rank=4, rr_dim=5,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = DualBranchLoss(class_weights=cw, focal_gamma=1.5, lambda_smooth=0.003,
                                lambda_balance=_args.lambda_balance)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, RRtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(RR5_train), torch.from_numpy(y_train)
    Xva_t, RRva_t, yva_t = torch.from_numpy(X_val), torch.from_numpy(RR5_val), torch.from_numpy(y_val)
    Xte_t, RRte_t, yte_t = torch.from_numpy(X_test), torch.from_numpy(RR5_test), torch.from_numpy(y_test)

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
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"], model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, rrb, yb, logits, internals, loss

        val_preds = predict_in_batches(model, Xva_t, RRva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        val_f1_s = f1_score(yva_t.numpy(), val_preds, labels=[1], average=None, zero_division=0)[0]
        val_f1_f = f1_score(yva_t.numpy(), val_preds, labels=[3], average=None, zero_division=0)[0]
        print(f"  epoch {epoch:2d} | val F1_macro={val_f1:.4f} | val F1-S={val_f1_s:.4f} | val F1-F={val_f1_f:.4f}")

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

    cm = print_confusion_matrix(y_true, test_preds, ["N", "S", "V", "F", "Q"])

    result = {
        "n_params": n_params, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1), "seed": SEED,
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "=" * 100)
    print(" SUMMARY: DSTA Dual-Branch (CNN rieng MLII/kenh-phu, hop nhat dac trung) vs moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<38} {'F1_macro':>10} {'N':>8} {'S':>8} {'V':>8} {'F':>8}")
    for name, ref in [("Official", OFFICIAL), ("C khong RR", C_NO_RR),
                       ("C+RR+Template 1-kenh (TB 3 seed)", C_RR_TEMPLATE_1CH_AVG3SEED),
                       ("2-kenh day du (TB 3 seed)", C_2CH_FULL_AVG3SEED)]:
        fpc = ref["f1_per_class"]
        print(f"{name:<38} {ref['f1_macro']:>10.4f} {fpc.get('N',0):>8.4f} {fpc.get('S',0):>8.4f} "
              f"{fpc.get('V',0):>8.4f} {fpc.get('F',0):>8.4f}")
    print(f"{'Dual-Branch seed=' + str(SEED):<38} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['N']:>8.4f} {result['f1_per_class']['S']:>8.4f} "
          f"{result['f1_per_class']['V']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    suffix = f"_{_args.out_suffix}" if _args.out_suffix else ""
    out_path = Path(f"results/logs/dsta_c_dualbranch_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"official": OFFICIAL, "C_no_rr": C_NO_RR,
                    "C_rr_template_1ch_avg3seed": C_RR_TEMPLATE_1CH_AVG3SEED,
                    "C_2ch_full_avg3seed": C_2CH_FULL_AVG3SEED, "this_run": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
