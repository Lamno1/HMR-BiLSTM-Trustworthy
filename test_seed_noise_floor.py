"""
Do NHIEU tu nhien theo seed cho CHINH 1 cau hinh (softmax, tuong duong "C
khong RR"), KHONG doi gi khac, tren CUNG tap con nho da dung xuyen suot cac
test truoc (N=3000 subsample, V=1500 subsample, S=836 het, F=384 het, 8
epoch). Muc dich: biet do dao dong TU NHIEN giua cac seed lon co so voi
"hieu ung" da thay khi doi co che (concat/hier/effnum/detach/sboost/OvR)
hay khong -- neu nhieu tu nhien >= hieu ung da thay, thi moi ket luan truoc
do ve cac co che kia deu KHONG DANG TIN, chi la nhieu ngau nhien.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score

from dsta_c_sboost import DSTACSBoost, class_weights_from, N_IDX, S_IDX, V_IDX, F_IDX
from hmr_bilstm import RLSTMLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH_SIZE = 64
N_TRAIN_N, N_TRAIN_V = 3000, 1500
SEEDS = [42, 123, 7]

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


def train_one_seed(seed, X_train, y_train, X_val, y_val):
    torch.manual_seed(seed)
    model = DSTACSBoost(input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
                         num_layers=2, attn_dim=24, num_heads=2, rank=4, use_sboost=False).to(DEVICE)
    cw = class_weights_from(y_train).to(DEVICE)
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    rr_dummy = np.zeros((len(X_train), 3), dtype=np.float32)
    Xtr_t, ytr_t, rrtr_t = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(rr_dummy)
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
        print(f"    [seed={seed}] epoch {epoch} loss={ep_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        RRva_t = torch.from_numpy(np.zeros((len(X_val), 3), dtype=np.float32)).to(DEVICE)
        preds = model(Xva_t, RRva_t).argmax(-1).cpu().numpy()
    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return f1_macro, dict(zip(["N", "S", "V", "F"], f1))


def main():
    # QUAN TRONG: dung CUNG 1 subset (rng seed co dinh=42 cho VIEC LAY MAU
    # subset) cho moi lan chay -- chi doi seed HUAN LUYEN (init trong so,
    # thu tu batch). Neu doi ca subset lan seed thi khong tach bach duoc
    # nguon nhieu.
    rng = np.random.default_rng(42)
    X_train, RR_train, y_train = load_split_4class("train")
    X_val, RR_val, y_val = load_split_4class("val")
    X_train_s, y_train_s = make_subset(X_train, y_train, rng)
    print(f"Subset train (CO DINH cho moi seed): {X_train_s.shape}  nhan: " +
          "  ".join(f"{n}={int((y_train_s==i).sum())}" for i, n in enumerate(['N','S','V','F'])))

    results = []
    for seed in SEEDS:
        print(f"\n=== Seed huan luyen = {seed} (subset/data GIONG HET nhau) ===")
        f1_macro, f1 = train_one_seed(seed, X_train_s, y_train_s, X_val, y_val)
        print(f"  [seed={seed}] F1_macro={f1_macro:.4f}  N={f1['N']:.4f} S={f1['S']:.4f} V={f1['V']:.4f} F={f1['F']:.4f}")
        results.append((seed, f1_macro, f1))

    print("\n" + "=" * 70)
    print("TONG HOP nhieu tu nhien theo seed (CUNG 1 config, CUNG 1 subset):")
    print(f"{'seed':<8}{'F1_macro':>10}{'N':>10}{'S':>10}{'V':>10}{'F':>10}")
    for seed, f1_macro, f1 in results:
        print(f"{seed:<8}{f1_macro:>10.4f}{f1['N']:>10.4f}{f1['S']:>10.4f}{f1['V']:>10.4f}{f1['F']:>10.4f}")

    for cn in ["N", "S", "V", "F"]:
        vals = [f1[cn] for _, _, f1 in results]
        print(f"  Do dao dong tu nhien F1-{cn}: min={min(vals):.4f} max={max(vals):.4f} "
              f"bien_do={max(vals)-min(vals):.4f}")
    f1_macro_vals = [fm for _, fm, _ in results]
    print(f"  Do dao dong tu nhien F1_macro: min={min(f1_macro_vals):.4f} max={max(f1_macro_vals):.4f} "
          f"bien_do={max(f1_macro_vals)-min(f1_macro_vals):.4f}")


if __name__ == "__main__":
    main()
