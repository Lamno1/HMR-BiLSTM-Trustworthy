"""
Kiem chung NHANH (subset nho, GPU, 8 epoch): so sanh softmax dung chung
(control, tuong duong "C khong RR") voi One-vs-Rest (4 sigmoid doc lap,
khong chia se chuan hoa) -- xem OvR co thoat duoc "canh tranh gradient qua
softmax" da chung minh la nguyen nhan goc cua moi that bai truoc do hay
khong.

Cung phuong phap luan: N=3000 (subsample), V=1500 (subsample), S=836 (het),
F=384 (het), 8 epoch, cung seed.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score

from dsta_c_sboost import DSTACSBoost, N_IDX, S_IDX, V_IDX, F_IDX
from dsta_c_ovr import DSTACOvR, OvRLoss, class_pos_weights
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


def make_subset(X, y, rng):
    idx_n = np.where(y == N_IDX)[0]
    idx_s = np.where(y == S_IDX)[0]
    idx_v = np.where(y == V_IDX)[0]
    idx_f = np.where(y == F_IDX)[0]
    idx_n = rng.choice(idx_n, min(N_TRAIN_N, len(idx_n)), replace=False)
    idx_v = rng.choice(idx_v, min(N_TRAIN_V, len(idx_v)), replace=False)
    idx = np.concatenate([idx_n, idx_s, idx_v, idx_f])
    rng.shuffle(idx)
    return X[idx], y[idx]


def eval_model(model, X_val, y_val, is_ovr):
    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        logits = model(Xva_t)
        if is_ovr:
            preds = torch.sigmoid(logits).argmax(-1).cpu().numpy()
        else:
            preds = logits.argmax(-1).cpu().numpy()
    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return f1_macro, dict(zip(["N", "S", "V", "F"], f1)), dict(zip(["N", "S", "V", "F"], prec)), dict(zip(["N", "S", "V", "F"], rec))


def train_softmax_control(X_train, y_train, X_val, y_val):
    from dsta_c_sboost import class_weights_from
    torch.manual_seed(SEED)
    model = DSTACSBoost(input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
                         num_layers=2, attn_dim=24, num_heads=2, rank=4, use_sboost=False).to(DEVICE)
    cw = class_weights_from(y_train).to(DEVICE)
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    rr_dummy = np.zeros((len(X_train), 3), dtype=np.float32)  # khong dung, model nhan rr nhung use_sboost=False bo qua
    Xtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(y_train)
    rrtr_t = torch.from_numpy(rr_dummy)
    n_train = len(Xtr_t)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, rrb, yb = Xtr_t[idx].to(DEVICE), rrtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits, internals = model(xb, rrb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        print(f"    [softmax] epoch {epoch} loss={ep_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        RRva_t = torch.from_numpy(np.zeros((len(X_val), 3), dtype=np.float32)).to(DEVICE)
        preds = model(Xva_t, RRva_t).argmax(-1).cpu().numpy()
    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return f1_macro, dict(zip(["N", "S", "V", "F"], f1)), dict(zip(["N", "S", "V", "F"], prec)), dict(zip(["N", "S", "V", "F"], rec))


def train_ovr(X_train, y_train, X_val, y_val):
    torch.manual_seed(SEED)
    model = DSTACOvR(input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
                      num_layers=2, attn_dim=24, num_heads=2, rank=4).to(DEVICE)
    pw = class_pos_weights(y_train).to(DEVICE)
    print(f"    pos_weights (N,S,V,F): {pw.cpu().numpy()}")
    criterion = OvRLoss(pos_weights=pw, focal_gamma=1.5, lambda_smooth=0.003)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(y_train)
    n_train = len(Xtr_t)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        print(f"    [ovr] epoch {epoch} loss={ep_loss/n_train:.4f}")

    return eval_model(model, X_val, y_val, is_ovr=True)


def main():
    rng = np.random.default_rng(SEED)
    X_train, RR_train, y_train = load_split_4class("train")
    X_val, RR_val, y_val = load_split_4class("val")

    X_train_s, y_train_s = make_subset(X_train, y_train, rng)
    print(f"Subset train: {X_train_s.shape}  nhan: " +
          "  ".join(f"{n}={int((y_train_s==i).sum())}" for i, n in enumerate(['N','S','V','F'])))

    print("\n=== Softmax (control, tuong duong 'C khong RR') ===")
    f1_macro_sm, f1_sm, prec_sm, rec_sm = train_softmax_control(X_train_s, y_train_s, X_val, y_val)
    print(f"  [softmax] F1_macro={f1_macro_sm:.4f}")
    for cn in ["N", "S", "V", "F"]:
        print(f"  [softmax] {cn}: P={prec_sm[cn]:.4f} R={rec_sm[cn]:.4f} F1={f1_sm[cn]:.4f}")

    print("\n=== One-vs-Rest (4 sigmoid doc lap) ===")
    f1_macro_ovr, f1_ovr, prec_ovr, rec_ovr = train_ovr(X_train_s, y_train_s, X_val, y_val)
    print(f"  [ovr] F1_macro={f1_macro_ovr:.4f}")
    for cn in ["N", "S", "V", "F"]:
        print(f"  [ovr] {cn}: P={prec_ovr[cn]:.4f} R={rec_ovr[cn]:.4f} F1={f1_ovr[cn]:.4f}")

    print("\n" + "=" * 70)
    print("So sanh:")
    print(f"  Softmax: F1_macro={f1_macro_sm:.4f}  N={f1_sm['N']:.4f} S={f1_sm['S']:.4f} V={f1_sm['V']:.4f} F={f1_sm['F']:.4f}")
    print(f"  OvR    : F1_macro={f1_macro_ovr:.4f}  N={f1_ovr['N']:.4f} S={f1_ovr['S']:.4f} V={f1_ovr['V']:.4f} F={f1_ovr['F']:.4f}")
    print(f"  Delta S: {f1_ovr['S']-f1_sm['S']:+.4f}   Delta F: {f1_ovr['F']-f1_sm['F']:+.4f}   "
          f"Delta N: {f1_ovr['N']-f1_sm['N']:+.4f}   Delta V: {f1_ovr['V']-f1_sm['V']:+.4f}")


if __name__ == "__main__":
    main()
