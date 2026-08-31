"""
train_lopo_cv.py  —  grouped K-fold CV over RECORDS for the rare classes (F, S).
=============================================================================
See docs/RARE_CLASS_PROGRESS.md. Replaces the fixed DS1/DS2 split as the PRIMARY
F/S evaluation: fold unit = record, so F1-F/F1-S become a distribution over
held-out patients with proper error bars, not a single-patient point estimate.

Protocol (locked — see the review in project memory):
  * K folds over records (default 5). Greedy-balanced on F-record then S-record
    count then source, so every test fold has F and S records from both DBs.
  * NO record's beats in both train and test of any fold.
  * Inner val = ~15% of the train-fold RECORDS (record-disjoint), greedy-balanced
    on F/S/source. Used for model selection ONLY — never touches outer test.
  * Per-source z-score, fit on TRAIN-FOLD records only.
  * Model selection = best inner-val macro-F1 over {S,V,F} (labels 1,2,3).
  * SWA over the tail epochs -> final model (attacks the epoch-lottery, B2).
  * Report: per-fold F1-F/F1-S/macro, mean +- std over folds, split by database.
  * Recipe = official (cosine LR, FGSM adv, focal + class weights). Epochs
    reduced (pool is ~3x DS1); train-N subsampled to keep per-epoch cost sane.

Resumable per fold. One GPU job at a time.

    python train_lopo_cv.py --arch flat --k 5 --epochs 25
    python train_lopo_cv.py --arch flat --folds 0 1        # subset
    python train_lopo_cv.py --arch flat --summarize-only
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import f1_score, classification_report, accuracy_score

from hmr_bilstm import RLSTMClassifier, RLSTMLoss
from train_inter_patient import CONFIG, set_seed, cosine_lr, fgsm_attack_train

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
POOL = Path("data/processed/lopo/pool.npz")
RECMETA = Path("data/processed/lopo/records.json")

# Fixed sentinel screening fold (see docs/LOPO_PROTOCOL.md). Held-out test =
# the two canonical hard MIT-BIH patients + the largest cross-DB INCART F/S
# records. A rung must beat sentinel F1(213-F) and F1(232-S) WITHOUT dropping
# N/V before it earns a full 5-fold LOPO run. 208 stays in train here.
SENTINEL_TEST = ["mitdb/213", "mitdb/232", "incart/I18", "incart/I74",
                 "incart/I33", "incart/I62"]


# --------------------------------------------------------------------------- folds
def assign_folds(rec_cls, rec_src, k, seed):
    """Snake-draft records into k folds, ordered by class-of-interest beat count
    descending, so the few huge F/S records (MIT-BIH 208/213/232) land in
    DIFFERENT folds and every fold gets a mix of big + small + INCART F/S
    records. F-records first, then S-only, then the rest."""
    rids = sorted(rec_cls)
    rng = np.random.default_rng(seed)
    assign = {}
    counter = [0]

    def snake(pool_rids, key):
        ordered = sorted(pool_rids, key=lambda r: -rec_cls[r][key])
        for r in ordered:
            i = counter[0]
            f = i % k if (i // k) % 2 == 0 else k - 1 - (i % k)   # boustrophedon
            assign[r] = f
            counter[0] += 1

    f_recs = [r for r in rids if rec_cls[r][3] > 0]
    s_recs = [r for r in rids if rec_cls[r][3] == 0 and rec_cls[r][1] > 0]
    rest = [r for r in rids if r not in f_recs and r not in s_recs]
    rng.shuffle(rest)
    snake(f_recs, 3)
    snake(s_recs, 1)
    snake(rest, 0)
    return assign


def carve_inner_val(train_rids, rec_cls, rec_src, frac, seed):
    """~frac of train records for inner val, greedy on F then S then source."""
    rng = np.random.default_rng(seed)
    n_target = max(2, int(round(frac * len(train_rids))))
    f_recs = [r for r in train_rids if rec_cls[r][3] > 0]
    s_recs = [r for r in train_rids if rec_cls[r][3] == 0 and rec_cls[r][1] > 0]
    rng.shuffle(f_recs); rng.shuffle(s_recs)
    val = []
    # take ~1/k share of F records and S records, then pad with random
    for r in f_recs[:max(1, len(f_recs) // 5)]:
        val.append(r)
    for r in s_recs[:max(1, len(s_recs) // 5)]:
        if r not in val:
            val.append(r)
    pool = [r for r in train_rids if r not in val]
    rng.shuffle(pool)
    while len(val) < n_target and pool:
        val.append(pool.pop())
    return set(val)


# --------------------------------------------------------------------------- norm
def per_source_zscore(X, src, stats=None):
    """X (N,187,1), src (N,). stats = {0:(m,s), 1:(m,s)} fit on train if None."""
    X = X.copy()
    if stats is None:
        stats = {}
        for s in (0, 1):
            m = src == s
            if m.any():
                stats[s] = (float(X[m].mean()), float(X[m].std() + 1e-8))
    for s, (m_, s_) in stats.items():
        mask = src == s
        X[mask] = (X[mask] - m_) / s_
    return X, stats


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def score(y, p, src):
    def block(mask):
        if mask.sum() == 0:
            return None
        fpc = f1_score(y[mask], p[mask], labels=[0, 1, 2, 3], average=None, zero_division=0)
        return {"n": int(mask.sum()),
                "f1_macro": float(f1_score(y[mask], p[mask], labels=[0, 1, 2, 3],
                                           average="macro", zero_division=0)),
                **{CLASS_NAMES[i]: float(fpc[i]) for i in range(4)}}
    return {"all": block(np.ones(len(y), bool)),
            "mitbih": block(src == 0), "incart": block(src == 1)}


# --------------------------------------------------------------------------- train
def class_weights(y, k=5):
    c = np.bincount(y, minlength=k).astype(np.float64)
    c = np.where(c == 0, 1e-9, c)
    return torch.from_numpy(np.clip(c.sum() / (k * c), 0.5, 50.0).astype(np.float32))


def train_fold(fold, idx, X, y, src, rr, device, args, ckpt_dir):
    tr_i, va_i, te_i = idx
    Xtr, Xtr_stats = per_source_zscore(X[tr_i], src[tr_i])
    Xva, _ = per_source_zscore(X[va_i], src[va_i], Xtr_stats)
    Xte, _ = per_source_zscore(X[te_i], src[te_i], Xtr_stats)
    ytr, yva, yte = y[tr_i], y[va_i], y[te_i]

    # subsample train N to keep per-epoch cost ~ DS1
    fold_off = fold if isinstance(fold, int) else 99
    rng = np.random.default_rng(args.seed + fold_off)
    n_i = np.where(ytr == 0)[0]
    if len(n_i) > args.train_n_cap:
        drop = rng.choice(n_i, len(n_i) - args.train_n_cap, replace=False)
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"  fold {fold}: train {len(ytr)} {np.bincount(ytr, minlength=5)} | "
          f"val {len(yva)} {np.bincount(yva, minlength=5)} | "
          f"test {len(yte)} {np.bincount(yte, minlength=5)}")

    set_seed(args.seed)
    cw = class_weights(ytr).to(device)
    model = RLSTMClassifier(input_size=1, hidden_size=CONFIG["hidden_size"],
                            dropout=CONFIG["dropout"], num_classes=5,
                            cnn_out_channels=CONFIG["cnn_out_channels"], num_layers=2).to(device)
    crit = RLSTMLoss(lambda_smooth=CONFIG["lambda_smooth"], class_weights=cw,
                     use_focal=True, focal_gamma=CONFIG["focal_gamma"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    use_amp = getattr(args, "amp", 1) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    tr_dl = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    swa_model = AveragedModel(model)
    swa_start = int(args.epochs * 0.6)
    state_path = ckpt_dir / f"lopo_{args.arch}_fold{fold}_state.pt"
    start_ep, best_v, best_state, n_swa = 1, -1.0, None, 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        swa_model.load_state_dict(st["swa"]); n_swa = st["n_swa"]
        start_ep, best_v, best_state = st["ep"] + 1, st["best_v"], st["best_state"]
        if st.get("scaler") is not None:
            scaler.load_state_dict(st["scaler"])
        print(f"  [resume] fold {fold} ep {start_ep}")

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        lr = cosine_lr(ep - 1, args.epochs, CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            split = int(len(xb) * (1 - CONFIG["adv_ratio"]))
            if split < len(xb):
                xa = fgsm_attack_train(model, xb[split:], yb[split:], CONFIG["adv_epsilon"], crit)
                xb = torch.cat([xb[:split], xa]); yb = torch.cat([yb[:split], yb[split:]])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit, internal = model(xb, return_internals=True)
                loss, _ = crit(logit, yb, internal["r_fwd"], internal["r_bwd"])
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt)
            scaler.update()
        vp = predict(model, Xva, device)
        v_svf = f1_score(yva, vp, labels=[1, 2, 3], average="macro", zero_division=0)
        if v_svf > best_v:
            best_v = v_svf
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep >= swa_start:
            swa_model.update_parameters(model); n_swa += 1
        print(f"  [f{fold}] ep{ep:2d} lr={lr:.5f} | inner-val F1(S,V,F)={v_svf:.4f}"
              f"{'  <-- best' if v_svf == best_v else ''}{'  +swa' if ep >= swa_start else ''}")
        torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "swa": swa_model.state_dict(), "n_swa": n_swa,
                    "best_v": best_v, "best_state": best_state,
                    "scaler": scaler.state_dict() if use_amp else None}, state_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    res = {"fold": fold, "n_swa": n_swa, "best_inner_val_svf": float(best_v),
           "train_time_sec": time.time() - t0}

    # best-epoch model
    model.load_state_dict(best_state)
    res["best_epoch_model"] = score(yte, predict(model, Xte, device), src[te_i])
    # SWA model (recompute BN stats on train)
    if n_swa > 0:
        update_bn(tr_dl, swa_model, device=device)
        sp = predict(swa_model, Xte, device)
        res["swa_model"] = score(yte, sp, src[te_i])
        print(classification_report(yte, sp, labels=[0, 1, 2, 3],
                                    target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return res


def summarize(out_dir, arch, k):
    rows = [json.load(open(out_dir / f"fold_{i}.json"))
            for i in range(k) if (out_dir / f"fold_{i}.json").exists()]
    if not rows:
        print("[summary] no folds done"); return
    print(f"\n{'='*88}\n  LOPO CV [{arch}]  --  {len(rows)}/{k} folds\n{'='*88}")
    for mdl in ["best_epoch_model", "swa_model"]:
        if mdl not in rows[0]:
            continue
        print(f"\n--- {mdl} ---")
        for blk in ["all", "mitbih", "incart"]:
            vals = {m: np.array([r[mdl][blk][m] for r in rows if r[mdl].get(blk)])
                    for m in ["f1_macro", "N", "S", "V", "F"]}
            if not len(vals["F"]):
                continue
            print(f"  [{blk}]")
            for m in ["f1_macro", "S", "V", "F"]:
                v = vals[m]
                print(f"    {m:<9} mean {v.mean():.4f}  std {v.std():.4f}  "
                      f"min {v.min():.4f}  max {v.max():.4f}  (per-fold: "
                      + " ".join(f"{x:.3f}" for x in v) + ")")
    json.dump({"arch": arch, "folds": [r["fold"] for r in rows], "per_fold": rows},
              open(out_dir / f"summary_{arch}.json", "w"), indent=2)
    print(f"\n[OK] {out_dir}/summary_{arch}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="flat", choices=["flat"])
    ap.add_argument("--mode", default="kfold", choices=["kfold", "sentinel"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--inner-val-frac", type=float, default=0.15)
    ap.add_argument("--train-n-cap", type=int, default=35000)
    ap.add_argument("--amp", type=int, default=1, help="1=mixed precision (CUDA), 0=fp32")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--out-dir", default="results/lopo_cv")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir, args.arch, args.k if args.mode == "kfold" else 1); return

    d = np.load(POOL)
    X, y, rr, record, source = d["X"], d["y"], d["rr"], d["record"], d["source"]
    rec_cls = {int(r): np.bincount(y[record == r], minlength=5).astype(float)
               for r in np.unique(record)}
    rec_src = {int(r): int(source[record == r][0]) for r in np.unique(record)}

    # ---- SENTINEL mode: one fixed screening fold, benchmark wall-time here ----
    if args.mode == "sentinel":
        recmeta = json.load(open(RECMETA))["records"]
        name2id = {v["name"]: int(k) for k, v in recmeta.items()}
        te_recs = {name2id[n] for n in SENTINEL_TEST if n in name2id}
        missing = [n for n in SENTINEL_TEST if n not in name2id]
        if missing:
            print(f"[warn] sentinel records not in pool: {missing}")
        trval = [r for r in rec_cls if r not in te_recs]
        va_recs = carve_inner_val(trval, rec_cls, rec_src, args.inner_val_frac, args.fold_seed)
        tr_recs = set(trval) - va_recs
        idx = (np.where(np.isin(record, list(tr_recs)))[0],
               np.where(np.isin(record, list(va_recs)))[0],
               np.where(np.isin(record, list(te_recs)))[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}\n=== SENTINEL fold  test={sorted(SENTINEL_TEST)} ===")
        rp = out_dir / f"sentinel_{args.arch}.json"
        res = train_fold("sentinel", idx, X, y, source, rr, device, args, ckpt_dir)
        json.dump(res, open(rp, "w"), indent=2)
        print(f"\n[OK] {rp}")
        print(f"[BENCHMARK] one fold ({args.epochs} ep) = {res['train_time_sec']/60:.0f} min "
              f"=> full 5-fold ~ {5*res['train_time_sec']/3600:.1f} h/seed")
        m = res.get("swa_model", res["best_epoch_model"])
        print(f"[SENTINEL] all: F1_F={m['all']['F']:.3f} F1_S={m['all']['S']:.3f} "
              f"N={m['all']['N']:.3f} V={m['all']['V']:.3f} | "
              f"mitbih F1_F={m['mitbih']['F']:.3f} F1_S={m['mitbih']['S']:.3f}")
        return

    fold_of = assign_folds(rec_cls, rec_src, args.k, args.fold_seed)
    fold_recs = defaultdict(list)
    for r, f in fold_of.items():
        fold_recs[f].append(r)
    print("Fold record / F-record / S-record / INCART-record counts:")
    for f in range(args.k):
        rs = fold_recs[f]
        fr = sum(1 for r in rs if rec_cls[r][3] > 0)
        sr = sum(1 for r in rs if rec_cls[r][1] > 0)
        ir = sum(1 for r in rs if rec_src[r] == 1)
        fb = int(sum(rec_cls[r][3] for r in rs)); sb = int(sum(rec_cls[r][1] for r in rs))
        print(f"  fold {f}: {len(rs):2d} recs | F-rec {fr} ({fb} beats) | S-rec {sr} ({sb} beats) | INCART {ir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    todo = args.folds if args.folds is not None else list(range(args.k))
    for f in todo:
        rp = out_dir / f"fold_{f}.json"
        if rp.exists():
            print(f"[skip] fold {f} done"); continue
        te_recs = set(fold_recs[f])
        trval_recs = [r for r in fold_of if r not in te_recs]
        va_recs = carve_inner_val(trval_recs, rec_cls, rec_src, args.inner_val_frac,
                                  args.fold_seed + f)
        tr_recs = set(trval_recs) - va_recs
        tr_i = np.where(np.isin(record, list(tr_recs)))[0]
        va_i = np.where(np.isin(record, list(va_recs)))[0]
        te_i = np.where(np.isin(record, list(te_recs)))[0]
        print(f"\n{'='*70}\n  FOLD {f}  (test records: {sorted(te_recs)})\n{'='*70}")
        res = train_fold(f, (tr_i, va_i, te_i), X, y, source, rr, device, args, ckpt_dir)
        json.dump(res, open(rp, "w"), indent=2)
        print(f"[OK] {rp}")
    summarize(out_dir, args.arch, args.k)


if __name__ == "__main__":
    main()
