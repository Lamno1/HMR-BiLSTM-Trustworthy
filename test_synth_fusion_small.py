"""
Kiem chung NHANH (subset nho, vai epoch, GPU) huong synthetic-fusion: tong
hop mau F gia lap bang cach tron N that + V that (cung tap con, khong dung
du lieu ngoai subset -- de la phep so sanh cong bang), roi xem co giup F1-F
tot hon so voi CHINH tap con do KHONG tang cuong hay khong.

Dung CUNG phuong phap luan voi test_hier_detach_hypothesis.py (tap con nho:
N=3000, V=1500, giu HET S=836, F=384) de so sanh cong bang giua cac huong da
thu trong session nay. Muc tieu: co bang chung THAT truoc khi quyet dinh co
dang hoan thien lai synthetic-fusion tren GPU day du hay khong.

Bien the A (control): model flat 5-lop, CHI du lieu that trong subset (khong
  tang cuong), dung DSTALiteClassifierSynthFusion nhung N_SYNTH=0 (alpha loss
  = 0 vi khong co mau synth) -- de kien truc giong het bien the B, chi khac
  co/khong co du lieu tong hop.
Bien the B: them N_SYNTH mau F gia lap (tron N+V TU CHINH SUBSET, khong dung
  ngoai) vao tap train.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from dsta_c_synth_fusion import (
    DSTALiteClassifierSynthFusion, SynthFusionLoss, synthesize_fusion_batch,
    class_weights_from, NUM_CLASSES, CLASS_NAMES, N_IDX, S_IDX, V_IDX, F_IDX,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH_SIZE = 64
SEED = 42
N_TRAIN_N, N_TRAIN_V = 3000, 1500
N_SYNTH = 300
ALPHA_LO, ALPHA_HI = 0.3, 0.7

DATA_DIR_SPLITS = "data/processed/splits"


def load_split(name):
    d = np.load(f"{DATA_DIR_SPLITS}/inter_{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


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


def train_variant(name, X_aug, y_aug, alpha_aug, is_synth_aug, X_val, y_val, cw):
    torch.manual_seed(SEED)
    model = DSTALiteClassifierSynthFusion(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=NUM_CLASSES,
        cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2, rank=4,
    ).to(DEVICE)
    criterion = SynthFusionLoss(lambda_smooth=0.003, class_weights=cw.to(DEVICE), focal_gamma=1.5, lambda_alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(X_aug)
    ytr_t = torch.from_numpy(y_aug)
    alphatr_t = torch.from_numpy(alpha_aug)
    is_synth_t = torch.from_numpy(is_synth_aug)
    n_train = len(Xtr_t)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(DEVICE)
            yb = ytr_t[idx].to(DEVICE)
            alphab = alphatr_t[idx].to(DEVICE)
            is_synth_b = is_synth_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals, alphab, is_synth_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        print(f"    [{name}] epoch {epoch} loss={ep_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(X_val).to(DEVICE)
        preds = model(Xva_t).argmax(-1).cpu().numpy()

    f1_macro = f1_score(y_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    print(f"  [{name}] F1_macro(val)={f1_macro:.4f}")
    for i, cn in enumerate(["N", "S", "V", "F"]):
        print(f"  [{name}] {cn}: precision={prec[i]:.4f} recall={rec[i]:.4f} f1={f1[i]:.4f}")
    return f1_macro, f1[F_IDX]


def main():
    rng = np.random.default_rng(SEED)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    X_train_s, y_train_s = make_subset(X_train, y_train, rng)
    print(f"Subset train: {X_train_s.shape}  nhan: " +
          "  ".join(f"{n}={int((y_train_s==i).sum())}" for i, n in enumerate(['N','S','V','F'])))

    cw = class_weights_from(y_train_s)
    print(f"Class weights (subset, chua tinh synth): {cw.numpy()}")

    print("\n=== Bien the A: KHONG tang cuong (control) ===")
    alpha_a = np.zeros(len(X_train_s), dtype=np.float32)
    is_synth_a = np.zeros(len(X_train_s), dtype=bool)
    f1_macro_a, f1_f_a = train_variant("A-control", X_train_s, y_train_s, alpha_a, is_synth_a,
                                        X_val, y_val, cw)

    print(f"\n=== Bien the B: THEM {N_SYNTH} mau F gia lap (tron N+V TU SUBSET) ===")
    rng2 = np.random.default_rng(SEED)
    X_synth, y_synth, alpha_synth = synthesize_fusion_batch(
        X_train_s, y_train_s, n_synth=N_SYNTH, rng=rng2, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    X_aug_b = np.concatenate([X_train_s, X_synth], axis=0)
    y_aug_b = np.concatenate([y_train_s, y_synth], axis=0)
    alpha_aug_b = np.concatenate([np.zeros(len(X_train_s), dtype=np.float32), alpha_synth], axis=0)
    is_synth_aug_b = np.concatenate(
        [np.zeros(len(X_train_s), dtype=bool), np.ones(N_SYNTH, dtype=bool)], axis=0)
    cw_b = class_weights_from(y_aug_b)
    print(f"Class weights (subset + synth): {cw_b.numpy()}")
    print(f"F count: that={int((y_train_s==F_IDX).sum())} -> sau tang cuong={int((y_aug_b==F_IDX).sum())}")
    f1_macro_b, f1_f_b = train_variant("B-synth", X_aug_b, y_aug_b, alpha_aug_b, is_synth_aug_b,
                                        X_val, y_val, cw_b)

    print("\n" + "=" * 70)
    print("So sanh F1-F (cang CAO cang tot):")
    print(f"  A (khong tang cuong):  F1-F={f1_f_a:.4f}  F1_macro={f1_macro_a:.4f}")
    print(f"  B (co tang cuong synth): F1-F={f1_f_b:.4f}  F1_macro={f1_macro_b:.4f}")
    print(f"  => {'synthetic-fusion CO giup (F1-F tang)' if f1_f_b > f1_f_a else 'synthetic-fusion KHONG giup'}")


if __name__ == "__main__":
    main()
