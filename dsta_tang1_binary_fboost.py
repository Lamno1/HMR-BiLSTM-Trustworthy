"""
Tang 1 + trong so uu tien rieng cho mau F (F_BOOST) trong loss.

Van de phat hien duoc o lan chay Tang 1 dau tien: ty le dinh tuyen SAI theo
tung lop that -- N=4.84%, S=4.19%, V=4.35%, F=26.03%. F la lop duy nhat bi
anh huong nang -- vi F la ban chat "hoa tron" nen do rong QRS cua no nam
GIUA N/S (hep) va V (rong), khien no de bi lac sang nhanh "hep" trong bai
toan nhi phan hien tai (chi co 2 trong so hep/rong THEO NHOM, khong phan
biet duoc F kho hon V trong CUNG nhom "rong").

Sua: nhan them 1 he so F_BOOST cho MOI mau CO NHAN GOC LA F (du no thuoc
nhom "rong" trong bai toan nhi phan) -- ep gradient chu y hon toi viec
khong bo sot F, chap nhan danh doi it precision cua nhom "rong" noi chung
(co the them N/S bi keo nham sang "rong").

Dung F_BOOST=3.0 lam gia tri thu nghiem dau tien (co the dieu chinh).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from dsta_tang1_binary import (
    DSTATang1Classifier, build_rr4, load_split_binary, temporal_smoothness_loss,
    predict_proba_in_batches, N_IDX, S_IDX, V_IDX, F_IDX, LOGREG_VAL_AUC,
)

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = 42
F_BOOST = 3.0

TANG1_BASELINE = {"test_auc": 0.9872, "f_misroute_rate": 0.2603}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  F_BOOST={F_BOOST}")

    X_train, RR_train, ybin_train, y5_train = load_split_binary("train")
    X_val, RR_val, ybin_val, y5_val = load_split_binary("val")
    X_test, RR_test, ybin_test, y5_test = load_split_binary("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    print("Tinh dac trung QRS-width...")
    RR4_train, qm, qs = build_rr4(X_train, RR_train, None, None, fit_stats=True)
    RR4_val, _, _ = build_rr4(X_val, RR_val, qm, qs)
    RR4_test, _, _ = build_rr4(X_test, RR_test, qm, qs)

    torch.manual_seed(SEED)
    counts = np.bincount(ybin_train, minlength=2).astype(np.float64)
    cw = torch.from_numpy((counts.sum() / (2 * counts)).astype(np.float32)).to(device)
    print(f"Class weights (hep, rong): {cw.cpu().numpy()}")

    # Trong so PER-SAMPLE: base = cw[nhan nhi phan] , nhan them F_BOOST neu
    # nhan GOC (5 lop) la F.
    sample_w_train = cw.cpu().numpy()[ybin_train].astype(np.float32)
    sample_w_train[y5_train == F_IDX] *= F_BOOST
    sample_w_train_t = torch.from_numpy(sample_w_train)
    print(f"So mau F trong train: {(y5_train==F_IDX).sum()}  "
          f"(trong so trung binh cho F: {sample_w_train[y5_train==F_IDX].mean():.2f}, "
          f"cho V: {sample_w_train[y5_train==V_IDX].mean():.2f})")

    model = DSTATang1Classifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    ce_none = nn.CrossEntropyLoss(reduction="none")
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
            wb = sample_w_train_t[idx].to(device)
            optimizer.zero_grad()
            logits, r_fwd, r_bwd = model(xb, rrb)
            l_task = (ce_none(logits, yb) * wb).mean()
            l_smooth = (temporal_smoothness_loss(r_fwd) + temporal_smoothness_loss(r_bwd)) / 2.0
            loss = l_task + lambda_smooth * l_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, rrb, yb, wb, logits, loss

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

    print("\n  Ty le dinh tuyen SAI theo tung lop that (5 lop):")
    misroute = {}
    for cls_idx, cls_name in [(N_IDX, "N"), (S_IDX, "S"), (V_IDX, "V"), (F_IDX, "F")]:
        mask = y5_test == cls_idx
        if mask.sum() == 0:
            continue
        expected_bin = 1 if cls_idx in (V_IDX, F_IDX) else 0
        wrong = float((test_pred[mask] != expected_bin).mean())
        misroute[cls_name] = wrong
        marker = "  <-- so voi baseline (khong F_BOOST): 26.03%" if cls_name == "F" else ""
        print(f"    {cls_name}: {wrong*100:.2f}% bi dinh tuyen sai  (n={int(mask.sum())}){marker}")

    result = {
        "n_params": n_params, "test_auc": float(test_auc), "test_acc": float(test_acc),
        "test_f1_rong": float(test_f1), "train_time_sec": train_time, "best_val_auc": float(best_auc),
        "f_boost": F_BOOST, "misroute_by_class": misroute,
        "baseline_no_fboost": TANG1_BASELINE,
    }
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/dsta_tang1_binary_fboost_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")
    print(f"\nSo sanh: AUC {TANG1_BASELINE['test_auc']:.4f} -> {test_auc:.4f}   "
          f"F misroute {TANG1_BASELINE['f_misroute_rate']*100:.2f}% -> {misroute.get('F',0)*100:.2f}%")


if __name__ == "__main__":
    main()
