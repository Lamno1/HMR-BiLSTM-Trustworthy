"""
Biến thể C (per_dim blend, 5 canh, gamma softmax) + Confusion-Margin Loss
cho cap (S, N) -- co che moi, KHONG doi kien truc DSTALiteGate (khong tang
attn_dim/rank), chi them 1 loss term.

Boi canh: 4 bien the ablation cho thay F1-S rat thap (0.12-0.25) o moi bien
the, dung ca 4 phan phat trien theo dung thu tu A<B<C (kien truc tot hon thi
S cung tot hon dan) -- nhung ngay ca C (tot nhat) van con cach official
(0.4335) rat xa. Da kiem tra: KHONG phai do thieu epoch (checkpoint chinh
thuc dung o epoch 21, tuong duong C dung o epoch 24 -- ngan sach train gan
nhu nhau). Gia thuyet hien tai: DSTALiteGate nen moi state 96 chieu xuong
attn_dim=24 roi qua rank=4 (chi 8 "khe" hang thap moi head) -- tin hieu F
(tho, manh) va tin hieu S (tinh vi, yeu) phai canh tranh chung 1 kenh hep,
tin hieu manh hon thang the.

Thay vi mo rong kenh (attn_dim/rank lon hon -- cham hon, khong chac cai
thien), them Confusion-Margin Loss cho cap (S, N) -- ep gradient huong
thang vao dung van de, khong doi kich thuoc model, khong lam train cham
hon dang ke. Dung lai recipe NHANH (30 epoch / patience 6) vi da xac nhan
ngan sach epoch khong phai nut that.
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
N_IDX, S_IDX, V_IDX, F_IDX = 0, 1, 2, 3

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = 42
EVAL_BATCH_SIZE = 256

DATA_DIR = Path("data/processed/splits")

VARIANT_KWARGS = dict(blend_mode="per_dim", extra_edge=True, gamma_mode="softmax")

# Confusion-margin: cap (lop dung, lop gay nhieu). S~N la nhieu nhat theo
# domain knowledge (S ectopic som nhung hinh thai gan giong N). Cho phep
# them cap khac neu can (vd them (S_IDX, V_IDX)).
CONFUSABLE_PAIRS = ((S_IDX, N_IDX),)
MARGIN_CM = 1.0
LAMBDA_CM = 0.3


def load_split(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def confusion_margin_loss(logits, targets, pairs, margin):
    total = logits.new_zeros(())
    n_pairs = 0
    for true_k, confuser_k in pairs:
        mask = targets == true_k
        if mask.sum() == 0:
            continue
        violation = margin - (logits[mask, true_k] - logits[mask, confuser_k])
        total = total + F.relu(violation).mean()
        n_pairs += 1
    return total / max(n_pairs, 1)


@torch.no_grad()
def predict_in_batches(model, X_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        preds.append(model(xb).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def train_and_eval(name, X_train, y_train, X_val, y_val, X_test, y_test, device):
    print(f"\n{'='*70}\n[{name}]  kwargs={VARIANT_KWARGS}  +confusion_margin={CONFUSABLE_PAIRS}\n{'='*70}")
    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = DSTALiteClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES,
        cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2, rank=4,
        **VARIANT_KWARGS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xva_t, yva_t = torch.from_numpy(X_val), torch.from_numpy(y_val)
    Xte_t, yte_t = torch.from_numpy(X_test), torch.from_numpy(y_test)

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        cm_sum, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])

            l_cm = confusion_margin_loss(logits, yb, CONFUSABLE_PAIRS, MARGIN_CM)
            loss = loss + LAMBDA_CM * l_cm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cm_sum += l_cm.item()
            n_batches += 1
            del xb, yb, logits, internals, loss

        model.eval()
        val_preds = predict_in_batches(model, Xva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        val_f1_s = f1_score(yva_t.numpy(), val_preds, labels=[S_IDX], average=None, zero_division=0)[0]
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f} | val F1-S = {val_f1_s:.4f} "
              f"| confusion_margin_loss = {cm_sum / n_batches:.4f}")

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

    result = train_and_eval("C_perdim_5edge_softmax_CM(S,N)", X_train, y_train, X_val, y_val,
                             X_test, y_test, device)

    official = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}
    c_no_cm = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}

    print("\n" + "=" * 100)
    print(" SUMMARY: C + Confusion-Margin(S,N) vs C goc vs official ")
    print("=" * 100)
    print(f"{'Model':<32} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<32} {official['f1_macro']:>10.4f} "
          f"{official['f1_per_class']['S']:>8.4f} {official['f1_per_class']['F']:>8.4f}")
    print(f"{'C (khong Confusion-Margin)':<32} {c_no_cm['f1_macro']:>10.4f} "
          f"{c_no_cm['f1_per_class']['S']:>8.4f} {c_no_cm['f1_per_class']['F']:>8.4f}")
    print(f"{'C + Confusion-Margin(S,N)':<32} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_c_confusion_margin_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": official, "C_no_confusion_margin": c_no_cm,
                    "C_with_confusion_margin": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
