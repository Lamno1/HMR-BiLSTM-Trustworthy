"""
Ablation thật trên đúng split chính thức (DS1 train/val, DS2 test) so sánh
4 biến thể DSTA-Lite -- KHÔNG giả lập, train thật trên
data/processed/splits/inter_*.npz.

Mục tiêu: tách bạch xem F1(S)/F1(F) thay đổi do PHẦN NÀO của công thức DSTA
gốc (TEN_EM_LA_GI.py) -- vì DSTA-Lite hiện tại đã âm thầm bỏ mất cơ chế
stream_attention theo từng chiều hidden khi đơn giản hoá, không chỉ đổi r_t.

Biến thể (thứ tự ưu tiên giảm dần):
  A: scalar blend, 4 cạnh, gamma softmax   (= DSTA-Lite hiện tại, control)
  B: per_dim blend, 4 cạnh, gamma softmax  (khôi phục stream_attention)
  C: per_dim blend, +extra_edge, gamma softmax  (+ khôi phục cạnh 2<->3)
  D: per_dim blend, +extra_edge, gamma sigmoid  (+ nới ràng buộc gamma)

Mỗi biến thể train từ đầu, cùng epoch budget, cùng seed, cùng loss/optimizer
-- single-variable so với nhau (chỉ khác cấu hình DSTALiteClassifier).
So sánh với ResMemECG chính thức đã train (F1-S=0.4335, F1-F=0.2411,
F1_macro=0.6279, từ results/logs/baseline_results.json).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from hmr_dsta_lite import DSTALiteClassifier
from hmr_bilstm import RLSTMLoss

NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = 42

DATA_DIR = Path("data/processed/splits")

VARIANTS = {
    "A_scalar_4edge_softmax": dict(blend_mode="scalar", extra_edge=False, gamma_mode="softmax"),
    "B_perdim_4edge_softmax": dict(blend_mode="per_dim", extra_edge=False, gamma_mode="softmax"),
    "C_perdim_5edge_softmax": dict(blend_mode="per_dim", extra_edge=True, gamma_mode="softmax"),
    "D_perdim_5edge_sigmoid": dict(blend_mode="per_dim", extra_edge=True, gamma_mode="sigmoid"),
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


EVAL_BATCH_SIZE = 256


def predict_in_batches(model, X_t, device):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), EVAL_BATCH_SIZE):
            xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
            preds.append(model(xb).argmax(-1).cpu())
            del xb
    return torch.cat(preds).numpy()


def train_and_eval(name, variant_kwargs, X_train, y_train, X_val, y_val, X_test, y_test, device):
    print(f"\n{'='*70}\n[{name}]  kwargs={variant_kwargs}\n{'='*70}")
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, yb, logits, internals, loss

        model.eval()
        val_preds = predict_in_batches(model, Xva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

        # A prior run OOM'd ~23 epochs in despite low steady-state usage --
        # CUDA allocator fragmentation across ~325 batches/epoch, not a real
        # leak. Defragment at each epoch boundary; cheap relative to an epoch.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
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
        "train_time_sec": train_time,
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

    # Official ResMemECG reference (from the just-completed real run_baselines.py run)
    official = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}

    print("\n" + "=" * 90)
    print(" SUMMARY: DSTA-Lite ablation vs official ResMemECG ")
    print("=" * 90)
    print(f"{'Variant':<28} {'Params':>10} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<28} {505038:>10,} {official['f1_macro']:>10.4f} "
          f"{official['f1_per_class']['S']:>8.4f} {official['f1_per_class']['F']:>8.4f}")
    for name, r in results.items():
        print(f"{name:<28} {r['n_params']:>10,} {r['f1_macro']:>10.4f} "
              f"{r['f1_per_class']['S']:>8.4f} {r['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": official, "variants": results}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
