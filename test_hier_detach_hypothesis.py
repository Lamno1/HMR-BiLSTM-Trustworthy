"""
Kiem chung NHANH (subset nho, vai epoch, CPU) gia thuyet: h_embed dung chung
KHONG detach cho stage2_head lam hong chat luong Tang 1 (N-vs-rest), vi
gradient tu Tang 2 (trong so nang cho F) keo meo bieu dien dung chung.

So sanh 2 bien the CUNG kien truc (chi khac detach_stage2_embed=True/False)
tren CUNG 1 tap con nho, CUNG seed, do TRUC TIEP ma tran nham lan nhi phan
N-vs-rest tren val (khong suy nguoc tu precision/recall nhu lan truoc).

Muc dich: co bang chung THAT truoc khi quyet dinh co dang chay GPU day du
voi fix nay hay khong.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score

from dsta_hier_n_first import DSTAHierNFirst, HierLoss, hier_predict, N_IDX, S_IDX, V_IDX, F_IDX, Q_IDX
from dsta_hier_n_first_train import (
    load_split_4class, class_weights_4, compute_template, build_template_feat,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH_SIZE = 64
SEED = 42

N_TRAIN_N, N_TRAIN_V = 3000, 1500  # subsample N, V (qua nhieu neu lay het); giu HET S, F (hiem)


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


def train_variant(detach_stage2, X_train, Tpl_train, RR_train, y_train,
                   X_val, Tpl_val, RR_val, y_val, cw):
    torch.manual_seed(SEED)
    model = DSTAHierNFirst(
        input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
        num_layers=2, attn_dim=24, num_heads=2, rank=4, template_dim=1, rr_dim=3,
        detach_stage2_embed=detach_stage2,
    ).to(DEVICE)
    criterion = HierLoss(class_weights_4=cw.to(DEVICE), focal_gamma=1.5, lambda_aux_stage1=0.3, lambda_smooth=0.003)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t, Tpltr_t, RRtr_t, ytr_t = (torch.from_numpy(X_train), torch.from_numpy(Tpl_train),
                                      torch.from_numpy(RR_train), torch.from_numpy(y_train))
    n_train = len(Xtr_t)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(DEVICE)
            tb = Tpltr_t[idx].to(DEVICE)
            rrb = RRtr_t[idx].to(DEVICE)
            yb = ytr_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logit_n, logits_svf, internals = model(xb, tb, rrb, return_internals=True)
            loss, _ = criterion(logit_n, logits_svf, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        print(f"    [detach={detach_stage2}] epoch {epoch} loss={ep_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        Tplva_t = torch.from_numpy(Tpl_val).to(DEVICE)
        RRva_t = torch.from_numpy(RR_val).to(DEVICE)
        logit_n, logits_svf = model(Xva_t, Tplva_t, RRva_t)
        preds = hier_predict(logit_n, logits_svf).cpu().numpy()

    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)

    # Ma tran nham lan nhi phan N-vs-rest THAT (khong suy nguoc)
    is_n_true = (y_val == N_IDX)
    is_n_pred = (preds == N_IDX)
    tp_n = int((is_n_true & is_n_pred).sum())
    fn_n = int((is_n_true & ~is_n_pred).sum())
    fp_n_leak = int((~is_n_true & is_n_pred).sum())  # true S/V/F nhung du doan la N -- LOT LUOI
    total_svf = int((~is_n_true).sum())
    total_n = int(is_n_true.sum())

    print(f"  [detach={detach_stage2}] F1_macro(val)={f1_macro:.4f}")
    print(f"  [detach={detach_stage2}] True N giu dung: {tp_n}/{total_n} ({tp_n/total_n*100:.2f}%)")
    print(f"  [detach={detach_stage2}] True S/V/F LOT LUOI thanh N: {fp_n_leak}/{total_svf} ({fp_n_leak/total_svf*100:.2f}%)")
    return f1_macro, fp_n_leak / total_svf


def main():
    rng = np.random.default_rng(SEED)
    X_train, RR_train, y_train = load_split_4class("train")
    X_val, RR_val, y_val = load_split_4class("val")

    X_train_s, RR_train_s, y_train_s = make_subset(X_train, RR_train, y_train, rng)
    print(f"Subset train: {X_train_s.shape}  nhan: " +
          "  ".join(f"{n}={int((y_train_s==i).sum())}" for i, n in enumerate(['N','S','V','F'])))

    template_norm = compute_template(X_train_s, y_train_s)
    Tpl_train_s, cm, cs = build_template_feat(X_train_s, template_norm, None, None, fit_stats=True)
    Tpl_val, _, _ = build_template_feat(X_val, template_norm, cm, cs)

    cw = class_weights_4(y_train_s)
    print(f"Class weights (subset): {cw.numpy()}")

    print("\n=== Bien the A: detach_stage2_embed=False (kien truc HIEN TAI) ===")
    f1_a, leak_a = train_variant(False, X_train_s, Tpl_train_s, RR_train_s, y_train_s,
                                  X_val, Tpl_val, RR_val, y_val, cw)

    print("\n=== Bien the B: detach_stage2_embed=True (FIX de xuat) ===")
    f1_b, leak_b = train_variant(True, X_train_s, Tpl_train_s, RR_train_s, y_train_s,
                                  X_val, Tpl_val, RR_val, y_val, cw)

    print("\n" + "=" * 70)
    print(f"So sanh ty le LOT LUOI (True S/V/F -> du doan N, cang THAP cang tot):")
    print(f"  Khong detach (A): {leak_a*100:.2f}%")
    print(f"  Co detach    (B): {leak_b*100:.2f}%")
    print(f"  => {'FIX co tac dung (giam lot luoi)' if leak_b < leak_a else 'FIX KHONG giup (khong giam lot luoi)'}")
    print(f"F1_macro(val): A={f1_a:.4f}  B={f1_b:.4f}")


if __name__ == "__main__":
    main()
