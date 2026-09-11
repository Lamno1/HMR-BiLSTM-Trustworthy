"""
Huan luyen DSTAHierNFirst (Tang 1: N vs {S,V,F} bang template-corr; Tang 2:
S vs V vs F bang RR-ratio) -- xem dsta_hier_n_first.py de biet ly do thiet ke
(rut ra tu y van AAMI EC57 + cac bai bao ve cascade N-first).

Lop Q bi loai (giong Tang1 binary truoc do) -- qua hiem (8 mau trong train).

So sanh voi:
  - ResMemECG chinh thuc (flat 5-lop): F1_macro=0.6279
  - DSTA-C khong RR (flat, F1-F tot nhat tung co): F1_macro=0.6357, F1-F=0.4322
  - DSTA-C + RR(3) + Template(1) (flat, 3 seed): F1_macro trung binh 0.5808,
    F1-S trung binh 0.5315 (tot nhat tung co) nhung F1-F trung binh 0.0606
    (sup do -- ly do chinh dan den kien truc phan tang nay)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from dsta_hier_n_first import DSTAHierNFirst, HierLoss, hier_predict, N_IDX, S_IDX, V_IDX, F_IDX, Q_IDX

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--out-suffix", type=str, default="")
_parser.add_argument("--weight-scheme", type=str, default="invfreq", choices=["invfreq", "effnum"])
_parser.add_argument("--effnum-beta", type=float, default=0.9999)
_args = _parser.parse_args()

EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
SEED = _args.seed
EVAL_BATCH_SIZE = 256
CLASS_NAMES_4 = ["N", "S", "V", "F"]

DATA_DIR = Path("data/processed/splits_rr")

OFFICIAL = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}
C_NO_RR = {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}}
C_RR_TEMPLATE_AVG3SEED = {"f1_macro": 0.5808, "f1_per_class": {"S": 0.5315, "F": 0.0606}}


def load_split_4class(name):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    X, rr, y = d["X"].astype(np.float32), d["rr"].astype(np.float32), d["y"].astype(np.int64)
    keep = y != Q_IDX
    return X[keep], rr[keep], y[keep]  # y da la 0=N,1=S,2=V,3=F dung san


def class_weights_4(y_train):
    counts = np.bincount(y_train, minlength=4).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (4 * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def class_weights_effnum(y_train, beta=0.9999):
    """Class-balanced weighting theo "effective number of samples"
    (Cui et al. 2019, https://arxiv.org/abs/1901.05555):
        E_n = (1 - beta^n) / (1 - beta)
        w_c = 1 / E_n , chuan hoa lai de trung binh trong so = 1 (giu thang
        do tuong tu inv-freq cu, de so sanh cong bang).
    Khac inv-freq thong thuong: beta<1 lam "bao hoa" tac dung khi n lon --
    tranh overcorrect cho lop da so (N) hay gay nhieu cho N/V nhu da nghi
    ngo khi dung inv-freq tho."""
    counts = np.bincount(y_train, minlength=4).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff_num
    w = w / w.mean()
    w = np.clip(w, 0.1, 100.0).astype(np.float32)
    return torch.from_numpy(w)


def compute_template(X_train, y_train, n_idx=N_IDX):
    x = X_train.squeeze(-1) if X_train.ndim == 3 else X_train
    template = x[y_train == n_idx].mean(axis=0)
    return (template / (np.linalg.norm(template) + 1e-8)).astype(np.float32)


def template_corr_batch(X, template_norm):
    x = X.squeeze(-1) if X.ndim == 3 else X
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return (xn @ template_norm).astype(np.float32)


def build_template_feat(X, template_norm, corr_mean, corr_std, fit_stats=False):
    corr = template_corr_batch(X, template_norm)
    if fit_stats:
        corr_mean = corr.mean()
        corr_std = corr.std() + 1e-8
    corr_norm = (corr - corr_mean) / corr_std
    return corr_norm[:, None].astype(np.float32), corr_mean, corr_std


@torch.no_grad()
def predict_in_batches(model, X_t, Tpl_t, RR_t, device):
    model.eval()
    preds = []
    for i in range(0, len(X_t), EVAL_BATCH_SIZE):
        xb = X_t[i:i + EVAL_BATCH_SIZE].to(device)
        tb = Tpl_t[i:i + EVAL_BATCH_SIZE].to(device)
        rrb = RR_t[i:i + EVAL_BATCH_SIZE].to(device)
        logit_n, logits_svf = model(xb, tb, rrb)
        preds.append(hier_predict(logit_n, logits_svf).cpu())
    return torch.cat(preds).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  SEED={SEED}")

    X_train, RR_train, y_train = load_split_4class("train")
    X_val, RR_val, y_val = load_split_4class("val")
    X_test, RR_test, y_test = load_split_4class("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")
    print(f"Train nhan (4 lop): " + "  ".join(
        f"{CLASS_NAMES_4[i]}={int((y_train==i).sum())}" for i in range(4)))

    template_norm = compute_template(X_train, y_train)
    Tpl_train, corr_mean, corr_std = build_template_feat(X_train, template_norm, None, None, fit_stats=True)
    Tpl_val, _, _ = build_template_feat(X_val, template_norm, corr_mean, corr_std)
    Tpl_test, _, _ = build_template_feat(X_test, template_norm, corr_mean, corr_std)
    print(f"template-corr train stats: mean={corr_mean:.4f} std={corr_std:.4f}")

    torch.manual_seed(SEED)
    if _args.weight_scheme == "effnum":
        cw = class_weights_effnum(y_train, beta=_args.effnum_beta).to(device)
    else:
        cw = class_weights_4(y_train).to(device)
    print(f"Weight scheme: {_args.weight_scheme}")
    print(f"Class weights (N,S,V,F): {cw.cpu().numpy()}")

    n_pos = int((y_train == N_IDX).sum())
    n_neg = len(y_train) - n_pos
    bce_pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)
    print(f"BCE pos_weight (tang 1, N vs rest): {bce_pos_weight.item():.4f}")

    model = DSTAHierNFirst(
        input_size=1, hidden_size=96, dropout=0.25, cnn_out_channels=64,
        num_layers=2, attn_dim=24, num_heads=2, rank=4, template_dim=1, rr_dim=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = HierLoss(class_weights_4=cw, focal_gamma=1.5, lambda_aux_stage1=0.3,
                          lambda_smooth=0.003, bce_pos_weight=bce_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr_t = torch.from_numpy(X_train)
    Tpltr_t = torch.from_numpy(Tpl_train)
    RRtr_t = torch.from_numpy(RR_train)
    ytr_t = torch.from_numpy(y_train)
    Xva_t, Tplva_t, RRva_t = torch.from_numpy(X_val), torch.from_numpy(Tpl_val), torch.from_numpy(RR_val)
    Xte_t, Tplte_t, RRte_t = torch.from_numpy(X_test), torch.from_numpy(Tpl_test), torch.from_numpy(RR_test)

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            tb = Tpltr_t[idx].to(device)
            rrb = RRtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            logit_n, logits_svf, internals = model(xb, tb, rrb, return_internals=True)
            loss, _ = criterion(logit_n, logits_svf, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, tb, rrb, yb, logit_n, logits_svf, internals, loss

        val_preds = predict_in_batches(model, Xva_t, Tplva_t, RRva_t, device)
        val_f1 = f1_score(y_val, val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        val_f1_s = f1_score(y_val, val_preds, labels=[S_IDX], average=None, zero_division=0)[0]
        val_f1_f = f1_score(y_val, val_preds, labels=[F_IDX], average=None, zero_division=0)[0]
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f} | val F1-S = {val_f1_s:.4f} | val F1-F = {val_f1_f:.4f}")

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
    test_preds = predict_in_batches(model, Xte_t, Tplte_t, RRte_t, device)

    acc = accuracy_score(y_test, test_preds)
    f1_macro = f1_score(y_test, test_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    f1_per_class = f1_score(y_test, test_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    print(f"\n  best val F1_macro={best_f1:.4f}  train_time={train_time:.1f}s")
    print(f"  TEST accuracy={acc:.4f}  F1_macro={f1_macro:.4f}")
    print(f"  TEST per-class F1: " + "  ".join(f"{CLASS_NAMES_4[i]}={f1_per_class[i]:.4f}" for i in range(4)))
    print(classification_report(y_test, test_preds, labels=[0, 1, 2, 3],
                                 target_names=CLASS_NAMES_4, zero_division=0, digits=4))

    result = {
        "n_params": n_params, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES_4[i]: float(f1_per_class[i]) for i in range(4)},
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1), "seed": SEED,
    }

    print("\n" + "=" * 100)
    print(" SUMMARY: Hier N-first (Tang1: N vs rest bang Template | Tang2: S/V/F bang RR) ")
    print("=" * 100)
    print(f"{'Model':<38} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<38} {OFFICIAL['f1_macro']:>10.4f} "
          f"{OFFICIAL['f1_per_class']['S']:>8.4f} {OFFICIAL['f1_per_class']['F']:>8.4f}")
    print(f"{'C (khong RR, flat)':<38} {C_NO_RR['f1_macro']:>10.4f} "
          f"{C_NO_RR['f1_per_class']['S']:>8.4f} {C_NO_RR['f1_per_class']['F']:>8.4f}")
    print(f"{'C+RR+Template (flat, TB 3 seed)':<38} {C_RR_TEMPLATE_AVG3SEED['f1_macro']:>10.4f} "
          f"{C_RR_TEMPLATE_AVG3SEED['f1_per_class']['S']:>8.4f} {C_RR_TEMPLATE_AVG3SEED['f1_per_class']['F']:>8.4f}")
    print(f"{'Hier N-first seed=' + str(SEED):<38} {result['f1_macro']:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    suffix = f"_{_args.out_suffix}" if _args.out_suffix else ""
    out_path = Path(f"results/logs/dsta_hier_n_first_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": OFFICIAL, "C_no_rr": C_NO_RR,
                    "C_rr_template_avg3seed": C_RR_TEMPLATE_AVG3SEED, "hier_n_first": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
