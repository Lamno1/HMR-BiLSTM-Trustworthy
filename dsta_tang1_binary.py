"""
Tang 1 cua kien truc cay phan tang (KHONG code Tang 2a/2b -- chi test rieng
Tang 1 truoc, dung nhu da thong nhat, vi neu Tang 1 sai thi moi thu phia sau
vo nghia).

Cau hoi nhi phan: QRS HEP ({N,S}=0) hay QRS RONG ({V,F}=1)?

Backbone: giu nguyen DSTA-C (per_dim blend, extra_edge, gamma softmax --
KHONG doi) + 4 dac trung phu tro (RR-ratio x3 + QRS-width x1) noi shallow
truoc classifier (dung co che da kiem chung o DSTALiteClassifierRRShallow),
chi doi dau ra tu 5 lop -> 2 lop.

Muc tieu: do AUC/accuracy THAT cua rieng quyet dinh nay tren du lieu that,
so voi tran da do bang dac trung tay + logistic regression don gian
(AUC=0.7306 tren val). Neu backbone thuc su hoc duoc bieu dien tot hon,
ky vong vuot qua muc nay ro ret. Neu khong vuot qua duoc nhieu, kien truc
cay se gap dung rui ro da canh bao (loi Tang 1 khong the cuu duoc o tang sau).

Lop Q (hiem, 8 mau trong train) bi LOAI KHOI TRAIN/VAL/TEST cua thi nghiem
nay -- qua it de co y nghia thong ke, tranh gay nhieu.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor
from hmr_dsta_lite import BiDSTALite

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

N_IDX, S_IDX, V_IDX, F_IDX, Q_IDX = 0, 1, 2, 3, 4
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = 42
EVAL_BATCH_SIZE = 256
QRS_THRESH_FRAC = 0.6

DATA_DIR = Path("data/processed/splits_rr")

# Tran tham chieu da do bang dac trung tay + Logistic Regression tren VAL that
LOGREG_VAL_AUC = 0.7306


class DSTATang1Classifier(nn.Module):
    """DSTA-C backbone + 4 dac trung phu tro shallow-concat -- dau ra NHI PHAN
    (hep=0 / rong=1) thay vi 5 lop."""

    def __init__(self, input_size=1, hidden_size=96, dropout=0.25,
                 cnn_out_channels=64, num_layers=2, attn_dim=24, num_heads=2,
                 rank=4, rr_dim=4):
        super().__init__()
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        self.birlstm = BiDSTALite(
            cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=True, gamma_mode="softmax", blend_mode="per_dim",
        )
        embed_dim = 2 * hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, 32), nn.GELU(), nn.Linear(32, 32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 32, hidden_size),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x, rr):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, _ = self.attention_pool(h_seq)
        h_pooled = self.dropout(self.layer_norm(h_pooled))
        rr_feat = self.rr_encoder(rr)
        combined = torch.cat([h_pooled, rr_feat], dim=-1)
        logits = self.classifier(combined)
        return logits, r_fwd, r_bwd


def qrs_width_batch(X, thresh_frac=QRS_THRESH_FRAC):
    x = X.squeeze(-1) if X.ndim == 3 else X
    n, T = x.shape
    peak_idx = np.argmax(np.abs(x), axis=1)
    peak_amp = np.abs(x[np.arange(n), peak_idx])
    thresh = (thresh_frac * peak_amp)[:, None]
    above = np.abs(x) > thresh
    widths = np.zeros(n, dtype=np.float32)
    for i in range(n):
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
    qrs = np.clip(qrs, 1, 60)
    if fit_stats:
        qrs_mean = qrs.mean()
        qrs_std = qrs.std() + 1e-8
    qrs_norm = (qrs - qrs_mean) / qrs_std
    return np.concatenate([rr, qrs_norm[:, None].astype(np.float32)], axis=-1), qrs_mean, qrs_std


def load_split_binary(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    X, rr, y = d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)
    keep = y != Q_IDX  # loai Q -- qua hiem, khong co y nghia thong ke
    X, rr, y = X[keep], rr[keep], y[keep]
    y_bin = np.isin(y, [V_IDX, F_IDX]).astype(np.int64)  # 0=hep{N,S} 1=rong{V,F}
    return X, rr, y_bin, y  # tra ca y goc (5 lop) de sau nay xem F1 tung lop bi anh huong the nao


def temporal_smoothness_loss(r_seq):
    if r_seq.size(1) < 2:
        return r_seq.sum() * 0.0
    diff = r_seq[:, 1:, :] - r_seq[:, :-1, :]
    return (diff ** 2).sum(dim=-1).mean()


@torch.no_grad()
def predict_proba_in_batches(model, X_t, RR_t, device):
    model.eval()
    probs = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        rrb = RR_t[i:i + EVAL_BATCH_SIZE].to(device)
        logits, _, _ = model(xb, rrb)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, RR_train, ybin_train, y5_train = load_split_binary("train")
    X_val, RR_val, ybin_val, y5_val = load_split_binary("val")
    X_test, RR_test, ybin_test, y5_test = load_split_binary("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")
    print(f"Train nhan: hep(0)={int((ybin_train==0).sum())}  rong(1)={int((ybin_train==1).sum())}")

    print("Tinh dac trung QRS-width...")
    RR4_train, qm, qs = build_rr4(X_train, RR_train, None, None, fit_stats=True)
    RR4_val, _, _ = build_rr4(X_val, RR_val, qm, qs)
    RR4_test, _, _ = build_rr4(X_test, RR_test, qm, qs)

    torch.manual_seed(SEED)
    counts = np.bincount(ybin_train, minlength=2).astype(np.float64)
    cw = torch.from_numpy((counts.sum() / (2 * counts)).astype(np.float32)).to(device)
    print(f"Class weights (hep, rong): {cw.cpu().numpy()}")

    model = DSTATang1Classifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(X_train)
    RRtr_t = torch.from_numpy(RR4_train)
    ytr_t = torch.from_numpy(ybin_train)
    Xva_t, RRva_t = torch.from_numpy(X_val), torch.from_numpy(RR4_val)
    Xte_t, RRte_t = torch.from_numpy(X_test), torch.from_numpy(RR4_test)

    n_train = len(Xtr_t)
    best_auc, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()
    lambda_smooth = 0.003

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            rrb = RRtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            logits, r_fwd, r_bwd = model(xb, rrb)
            l_task = criterion(logits, yb)
            l_smooth = (temporal_smoothness_loss(r_fwd) + temporal_smoothness_loss(r_bwd)) / 2.0
            loss = l_task + lambda_smooth * l_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, rrb, yb, logits, loss

        val_proba = predict_proba_in_batches(model, Xva_t, RRva_t, device)
        val_auc = roc_auc_score(ybin_val, val_proba)
        val_acc = accuracy_score(ybin_val, (val_proba > 0.5).astype(int))
        print(f"  epoch {epoch:2d} | val AUC = {val_auc:.4f} | val acc = {val_acc:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if val_auc > best_auc:
            best_auc = val_auc
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
    test_proba = predict_proba_in_batches(model, Xte_t, RRte_t, device)
    test_pred = (test_proba > 0.5).astype(int)

    test_auc = roc_auc_score(ybin_test, test_proba)
    test_acc = accuracy_score(ybin_test, test_pred)
    test_f1 = f1_score(ybin_test, test_pred)

    print(f"\n  best val AUC={best_auc:.4f}  train_time={train_time:.1f}s")
    print(f"  TEST AUC={test_auc:.4f}  accuracy={test_acc:.4f}  F1(rong)={test_f1:.4f}")
    print(classification_report(ybin_test, test_pred, target_names=["hep{N,S}", "rong{V,F}"], digits=4))

    # Xem loi dinh tuyen anh huong tung lop 5-chieu the nao (khong danh gia
    # phan loai cuoi, chi xem % moi lop bi dinh tuyen SAI o Tang 1)
    print("\n  Ty le dinh tuyen SAI theo tung lop that (5 lop):")
    for cls_idx, cls_name in [(N_IDX, "N"), (S_IDX, "S"), (V_IDX, "V"), (F_IDX, "F")]:
        mask = y5_test == cls_idx
        if mask.sum() == 0:
            continue
        expected_bin = 1 if cls_idx in (V_IDX, F_IDX) else 0
        wrong = (test_pred[mask] != expected_bin).mean()
        print(f"    {cls_name}: {wrong*100:.2f}% bi dinh tuyen sai  (n={int(mask.sum())})")

    result = {
        "n_params": n_params, "test_auc": float(test_auc), "test_acc": float(test_acc),
        "test_f1_rong": float(test_f1), "train_time_sec": train_time, "best_val_auc": float(best_auc),
        "logreg_val_auc_reference": LOGREG_VAL_AUC,
    }
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_tang1_binary_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")
    print(f"\nSo voi tran Logistic Regression (dac trung tay, AUC val={LOGREG_VAL_AUC}): "
          f"backbone that dat AUC val={best_auc:.4f} ({'VUOT' if best_auc > LOGREG_VAL_AUC else 'KHONG VUOT'})")


if __name__ == "__main__":
    main()
