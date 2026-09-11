"""
Kiem chung NHANH (subset nho, GPU, 8 epoch) co che "S-boost chi duoc cong":
so sanh Bien the A (use_sboost=False, tuong duong "C khong RR") voi Bien the
B (use_sboost=True) tren CUNG 1 tap con, CUNG seed.

Ky vong (neu dung gia thuyet): F1-S(B) > F1-S(A) RO RET, trong khi
F1-F(B) va F1-N(B) GIU NGUYEN xap xi F1-F(A)/F1-N(A) (khong bi keo xuong
nhu moi bien the noi-embedding-dung-chung da thu truoc do).
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score

from dsta_c_sboost import DSTACSBoost, class_weights_from, N_IDX, S_IDX, V_IDX, F_IDX
from hmr_bilstm import RLSTMLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH_SIZE = 64
SEED = 42
N_TRAIN_N, N_TRAIN_V = 3000, 1500

DATA_DIR = "data/processed/splits_rr"
Q_IDX = 4


def load_split_4class(name):
    d = np.load(f"{DATA_DIR}/inter_{name}.npz")
    X, rr, y = d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)
    keep = y != Q_IDX
    return X[keep], rr[keep], y[keep]


def make_subset(X, RR, y, rng):
    idx_n = np.where(y == N_IDX)[0]
    idx_s = np.where(y == S_IDX)[0]
    idx_v = np.where(y == V_IDX)[0]
    idx_f = np.where(y == F_IDX)[0]
    idx_n = rng.choice(idx_n, min(N_TRAIN_N, len(idx_n)), replace=False)
    idx_v = rng.choice(idx_v, min(N_TRAIN_V, len(idx_v)), replace=False)
    idx = np.concatenate([idx_n, idx_s, idx_v, idx_f])
    rng.shuffle(idx)
    return X[idx], RR[idx], y[idx]


def train_variant(name, use_sboost, X_train, RR_train, y_train, X_val, RR_val, y_val, cw):
    torch.manual_seed(SEED)
    model = DSTACSBoost(
        input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
        num_layers=2, attn_dim=24, num_heads=2, rank=4, rr_dim=3, use_sboost=use_sboost,
    ).to(DEVICE)
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw.to(DEVICE), use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, RRtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(RR_train), torch.from_numpy(y_train)
    n_train = len(Xtr_t)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(DEVICE)
            rrb = RRtr_t[idx].to(DEVICE)
            yb = ytr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits, internals = model(xb, rrb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        print(f"    [{name}] epoch {epoch} loss={ep_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        RRva_t = torch.from_numpy(RR_val).to(DEVICE)
        preds = model(Xva_t, RRva_t).argmax(-1).cpu().numpy()

    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    print(f"  [{name}] F1_macro(val)={f1_macro:.4f}")
    for i, cn in enumerate(["N", "S", "V", "F"]):
        print(f"  [{name}] {cn}: precision={prec[i]:.4f} recall={rec[i]:.4f} f1={f1[i]:.4f}")
    return f1_macro, dict(zip(["N", "S", "V", "F"], f1))


def main():
    rng = np.random.default_rng(SEED)
    X_train, RR_train, y_train = load_split_4class("train")
    X_val, RR_val, y_val = load_split_4class("val")

    X_train_s, RR_train_s, y_train_s = make_subset(X_train, RR_train, y_train, rng)
    print(f"Subset train: {X_train_s.shape}  nhan: " +
          "  ".join(f"{n}={int((y_train_s==i).sum())}" for i, n in enumerate(['N','S','V','F'])))
    cw = class_weights_from(y_train_s)
    print(f"Class weights: {cw.numpy()}")

    print("\n=== Bien the A: use_sboost=False (tuong duong 'C khong RR') ===")
    f1_macro_a, f1_a = train_variant("A-no_sboost", False, X_train_s, RR_train_s, y_train_s,
                                      X_val, RR_val, y_val, cw)

    print("\n=== Bien the B: use_sboost=True (co che chi cong ve S) ===")
    f1_macro_b, f1_b = train_variant("B-sboost", True, X_train_s, RR_train_s, y_train_s,
                                      X_val, RR_val, y_val, cw)

    print("\n" + "=" * 70)
    print("So sanh:")
    print(f"  A: F1_macro={f1_macro_a:.4f}  N={f1_a['N']:.4f} S={f1_a['S']:.4f} V={f1_a['V']:.4f} F={f1_a['F']:.4f}")
    print(f"  B: F1_macro={f1_macro_b:.4f}  N={f1_b['N']:.4f} S={f1_b['S']:.4f} V={f1_b['V']:.4f} F={f1_b['F']:.4f}")
    print(f"  Delta S: {f1_b['S']-f1_a['S']:+.4f}   Delta F: {f1_b['F']-f1_a['F']:+.4f}   Delta N: {f1_b['N']-f1_a['N']:+.4f}")
    ok = (f1_b['S'] > f1_a['S']) and (f1_b['F'] >= f1_a['F'] - 0.03) and (f1_b['N'] >= f1_a['N'] - 0.02)
    print(f"  => {'GIA THUYET DUNG: S tang, F/N khong bi keo xuong dang ke' if ok else 'Chua ro rang / gia thuyet khong dung nhu ky vong'}")


if __name__ == "__main__":
    main()
