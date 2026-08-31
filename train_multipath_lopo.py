"""
train_multipath_lopo.py  —  HMR-MultiPath ablation rungs on the record-level CV.
=============================================================================
Shares fold construction / normalization / sentinel / scoring with
train_lopo_cv.py (imported). Adds: 2nd-lead + RR + record inputs, GRL lambda
ramp, manual SWA (AveragedModel can't handle the multi-arg forward).

Rungs (docs/RARE_CLASS_PROGRESS.md §3):
  rung1  --v1
  rung2  --v1 --rhythm
  rung3  --v1 --rhythm --grl
  rung4  --v1 --rhythm --grl --loss balanced

    python train_multipath_lopo.py --mode sentinel --v1 --epochs 25
    python train_multipath_lopo.py --mode kfold --v1 --rhythm --grl --k 5 --seeds 42 123
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

from hmr_multipath import HMRMultiPath, MultiPathLoss
from train_inter_patient import CONFIG, set_seed, cosine_lr
from train_lopo_cv import (
    POOL, RECMETA, SENTINEL_TEST, CLASS_NAMES,
    assign_folds, carve_inner_val, carve_conformal_s_cal, per_source_zscore, score, class_weights,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def rr_zscore(rr, src, stats=None):
    rr = np.clip(rr, 0.2, 3.0)
    if stats is None:
        stats = {}
        for s in (0, 1):
            m = src == s
            if m.any():
                stats[s] = (rr[m].mean(0), rr[m].std(0) + 1e-8)
    out = rr.copy()
    for s, (mu, sd) in stats.items():
        out[src == s] = (rr[src == s] - mu) / sd
    return out.astype(np.float32), stats


@torch.no_grad()
def predict_mp_logits(model, X, X2, RR, device, bs=512, amp=True):
    """Return pre-softmax logits in input order for post-hoc decision analysis."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        sl = slice(i, i + bs)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(torch.from_numpy(X[sl]).to(device),
                           torch.from_numpy(X2[sl]).to(device),
                           torch.from_numpy(RR[sl]).to(device))
        # float32 on the way out: AMP emits fp16 logits, but P2 calibration and
        # conformal quantiles must not be fp16-quantised.
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def predict_mp(model, X, X2, RR, device, bs=512, amp=True):
    return predict_mp_logits(model, X, X2, RR, device, bs=bs, amp=amp).argmax(-1)


def fgsm_primary(model, xb, x2b, rrb, yb, eps, amp=True):
    xr = xb.clone().detach().requires_grad_(True)
    with torch.cuda.amp.autocast(enabled=amp and xb.device.type == "cuda"):
        logit = model(xr, x2b, rrb)
        loss = F.cross_entropy(logit, yb)
    g = torch.autograd.grad(loss, xr)[0]
    return (xr + eps * g.sign()).detach()


def recompute_bn(model, loader, device):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        for xb, x2b, rrb, yb, rcb in loader:
            model(xb.to(device), x2b.to(device), rrb.to(device))
    model.eval()


def train_fold(fold, idx, data, device, args, ckpt_dir):
    X, X2, y, RR, record, source = data
    tr_i, va_i, te_i, tu_i, ca_i = idx
    Xtr, st = per_source_zscore(X[tr_i], source[tr_i])
    Xva, _ = per_source_zscore(X[va_i], source[va_i], st)
    Xte, _ = per_source_zscore(X[te_i], source[te_i], st)
    X2tr, st2 = per_source_zscore(X2[tr_i], source[tr_i])
    X2va, _ = per_source_zscore(X2[va_i], source[va_i], st2)
    X2te, _ = per_source_zscore(X2[te_i], source[te_i], st2)
    RRtr, rst = rr_zscore(RR[tr_i], source[tr_i])
    RRva, _ = rr_zscore(RR[va_i], source[va_i], rst)
    RRte, _ = rr_zscore(RR[te_i], source[te_i], rst)
    Xtu, _ = per_source_zscore(X[tu_i], source[tu_i], st)
    Xca, _ = per_source_zscore(X[ca_i], source[ca_i], st)
    X2tu, _ = per_source_zscore(X2[tu_i], source[tu_i], st2)
    X2ca, _ = per_source_zscore(X2[ca_i], source[ca_i], st2)
    RRtu, _ = rr_zscore(RR[tu_i], source[tu_i], rst)
    RRca, _ = rr_zscore(RR[ca_i], source[ca_i], rst)
    ytr, yva, yte = y[tr_i], y[va_i], y[te_i]
    rec_tr = record[tr_i]

    # remap train record ids -> contiguous for the adversary
    uniq = sorted(set(rec_tr.tolist()))
    r2c = {r: i for i, r in enumerate(uniq)}
    rec_tr_c = np.array([r2c[r] for r in rec_tr], np.int64)
    n_rec = len(uniq)

    fo = fold if isinstance(fold, int) else 99
    rng = np.random.default_rng(args.seed + fo)
    n_idx = np.where(ytr == 0)[0]
    if len(n_idx) > args.train_n_cap:
        drop = set(rng.choice(n_idx, len(n_idx) - args.train_n_cap, replace=False).tolist())
        keep = np.array([i not in drop for i in range(len(ytr))])
        Xtr, X2tr, RRtr, ytr, rec_tr_c = Xtr[keep], X2tr[keep], RRtr[keep], ytr[keep], rec_tr_c[keep]
    print(f"  fold {fold}: train {len(ytr)} {np.bincount(ytr, minlength=5)} | "
          f"val {len(yva)} {np.bincount(yva, minlength=5)} | test {len(yte)} {np.bincount(yte, minlength=5)} "
          f"| n_rec {n_rec}")

    set_seed(args.seed)
    cw = class_weights(ytr).to(device)
    prior = torch.from_numpy((np.bincount(ytr, minlength=5) / len(ytr)).astype(np.float32)).to(device)
    model = HMRMultiPath(hidden_size=CONFIG["hidden_size"], dropout=CONFIG["dropout"],
                         cnn_out_channels=CONFIG["cnn_out_channels"], num_layers=2,
                         use_v1=args.v1, use_rhythm=args.rhythm, use_grl=args.grl,
                         n_records=n_rec, rr_dim=RR.shape[1]).to(device)
    crit = MultiPathLoss(class_weights=cw, lambda_smooth=CONFIG["lambda_smooth"],
                         loss=args.loss, class_prior=prior if args.loss == "balanced" else None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(X2tr),
                                  torch.from_numpy(RRtr), torch.from_numpy(ytr),
                                  torch.from_numpy(rec_tr_c)),
                    batch_size=CONFIG["batch_size"], shuffle=True)

    # last ~20% of epochs. 0.6 was too wide — the rhythm S-specialist keeps
    # shifting the S/F boundary late, so a wide SWA window over-favoured S and
    # tanked F (rung 2: SWA mitbih F .34 vs best-epoch .56).
    swa_start = int(args.epochs * 0.8)
    swa_sum, swa_n = None, 0
    tag = f"{'v' if args.v1 else ''}{'r' if args.rhythm else ''}{'g' if args.grl else ''}{args.loss}"
    # A checkpoint is state for one *seeded* optimization trajectory.  Omitting
    # the seed silently made a later seed resume the earlier one.
    state_path = ckpt_dir / f"mp_{tag}_fold{fold}_seed{args.seed}_state.pt"
    start_ep, best_v, best_state = 1, -1.0, None
    if state_path.exists():
        stt = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(stt["model"]); opt.load_state_dict(stt["opt"])
        start_ep, best_v, best_state = stt["ep"] + 1, stt["best_v"], stt["best_state"]
        swa_sum, swa_n = stt["swa_sum"], stt["swa_n"]
        if stt.get("scaler") is not None:
            scaler.load_state_dict(stt["scaler"])
        print(f"  [resume] fold {fold} ep {start_ep}")

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        lr = cosine_lr(ep - 1, args.epochs, CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        lam = args.grl_max * min(1.0, ep / max(1, args.grl_ramp)) if args.grl else 0.0
        model.train()
        for xb, x2b, rrb, yb, rcb in dl:
            xb, x2b, rrb, yb, rcb = (t.to(device) for t in (xb, x2b, rrb, yb, rcb))
            sp = int(len(xb) * (1 - CONFIG["adv_ratio"]))
            if sp < len(xb):
                xa = fgsm_primary(model, xb[sp:], x2b[sp:], rrb[sp:], yb[sp:], CONFIG["adv_epsilon"], use_amp)
                xb = torch.cat([xb[:sp], xa])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit, intr = model(xb, x2b, rrb, yb, grl_lambda=lam, return_internals=True)
                loss, comp = crit(logit, yb, intr, rcb)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt)
            scaler.update()
        vp = predict_mp(model, Xva, X2va, RRva, device, amp=use_amp)
        v_svf = f1_score(yva, vp, labels=[1, 2, 3], average="macro", zero_division=0)
        if v_svf > best_v:
            best_v = v_svf
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep >= swa_start:
            pd = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
            if swa_sum is None:
                swa_sum = pd
            else:
                for k in swa_sum:
                    swa_sum[k] += pd[k]
            swa_n += 1
        print(f"  [f{fold} {tag}] ep{ep:2d} lr={lr:.5f} lam={lam:.3f} | "
              f"inner-val F1(S,V,F)={v_svf:.4f} adv={float(comp['adv']):.3f}"
              f"{'  <-- best' if v_svf == best_v else ''}{'  +swa' if ep >= swa_start else ''}")
        torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_v": best_v, "best_state": best_state,
                    "swa_sum": swa_sum, "swa_n": swa_n,
                    "scaler": scaler.state_dict() if use_amp else None}, state_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    res = {"fold": fold, "tag": tag, "n_swa": swa_n, "best_inner_val_svf": float(best_v),
           "train_time_sec": time.time() - t0,
           "n_params": sum(p.numel() for p in model.parameters())}
    model.load_state_dict(best_state)
    te_logits = predict_mp_logits(model, Xte, X2te, RRte, device, amp=use_amp)
    va_logits = predict_mp_logits(model, Xva, X2va, RRva, device, amp=use_amp)
    tu_logits = predict_mp_logits(model, Xtu, X2tu, RRtu, device, amp=use_amp)
    ca_logits = predict_mp_logits(model, Xca, X2ca, RRca, device, amp=use_amp)
    res["best_epoch_model"] = score(yte, te_logits.argmax(-1), source[te_i])
    if swa_n > 0:
        swa_model = copy.deepcopy(model)
        swa_model.load_state_dict(best_state)          # start from best (gets all buffers)
        with torch.no_grad():
            for k, p in swa_model.named_parameters():
                if k in swa_sum:
                    p.copy_((swa_sum[k] / swa_n).to(device))
        recompute_bn(swa_model, dl, device)
        swa_te_logits = predict_mp_logits(swa_model, Xte, X2te, RRte, device, amp=use_amp)
        sp2 = swa_te_logits.argmax(-1)
        res["swa_model"] = score(yte, sp2, source[te_i])
        print(classification_report(yte, sp2, labels=[0, 1, 2, 3],
                                    target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    # Keep the artifact independent of the data pool ordering.  RR is raw
    # (pre-z-score) so downstream code can audit every transform; logits are
    # from the fold-specific normalisation above.
    artifact_dir = Path(args.out_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"fold{fold}_seed{args.seed}_{tag}_preds.npz"
    meta = {
        "fold": fold, "seed": args.seed, "tag": tag, "tuning_split": "inner_val",
        "best_inner_val_svf": float(best_v),
        "train_indices": tr_i.tolist(), "val_indices": va_i.tolist(), "test_indices": te_i.tolist(),
        "tuning_indices": tu_i.tolist(), "conformal_cal_indices": ca_i.tolist(),
        "x_zscore": {str(k): [float(v[0]), float(v[1])] for k, v in st.items()},
        "x2_zscore": {str(k): [float(v[0]), float(v[1])] for k, v in st2.items()},
        "rr_zscore": {str(k): [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()] for k, v in rst.items()},
    }
    payload = {
        "te_logits": te_logits, "te_y": yte, "te_record": record[te_i], "te_source": source[te_i],
        "te_rr_raw": RR[te_i], "te_pred": te_logits.argmax(-1),
        "va_logits": va_logits, "va_y": yva, "va_record": record[va_i], "va_source": source[va_i],
        "va_rr_raw": RR[va_i], "va_pred": va_logits.argmax(-1),
        "tuning_logits": tu_logits, "tuning_y": y[tu_i], "tuning_record": record[tu_i],
        "tuning_source": source[tu_i], "tuning_rr_raw": RR[tu_i], "tuning_pred": tu_logits.argmax(-1),
        "conformal_cal_logits": ca_logits, "conformal_cal_y": y[ca_i], "conformal_cal_record": record[ca_i],
        "conformal_cal_source": source[ca_i], "conformal_cal_rr_raw": RR[ca_i],
        "conformal_cal_pred": ca_logits.argmax(-1),
        "meta_json": np.array(json.dumps(meta)),
    }
    if swa_n > 0:
        payload["swa_te_logits"] = swa_te_logits
    np.savez_compressed(artifact_path, **payload)
    model_path = artifact_dir / f"fold{fold}_seed{args.seed}_{tag}_model.pt"
    torch.save({"model_state": best_state, "fold": fold, "seed": args.seed, "tag": tag,
                "config": {"use_v1": args.v1, "use_rhythm": args.rhythm, "use_grl": args.grl,
                           "rr_dim": int(RR.shape[1])}}, model_path)
    res["prediction_artifact"] = str(artifact_path)
    res["model_artifact"] = str(model_path)
    return res


def build_sentinel_idx(record, rec_cls, rec_src, args):
    recmeta = json.load(open(RECMETA))["records"]
    name2id = {v["name"]: int(k) for k, v in recmeta.items()}
    te = {name2id[n] for n in SENTINEL_TEST if n in name2id}
    trval = [r for r in rec_cls if r not in te]
    va = carve_inner_val(trval, rec_cls, rec_src, args.inner_val_frac, args.fold_seed)
    tr = set(trval) - va
    return (np.where(np.isin(record, list(tr)))[0],
            np.where(np.isin(record, list(va)))[0],
            np.where(np.isin(record, list(te)))[0])


def summarize(out_dir, tag, k):
    rows = [json.load(open(p)) for p in sorted(out_dir.glob(f"*_{tag}.json"))]
    if not rows:
        print(f"[summary] no {tag} results"); return
    print(f"\n{'='*84}\n  MultiPath [{tag}]  {len(rows)} run(s)\n{'='*84}")
    for mdl in ["best_epoch_model", "swa_model"]:
        if mdl not in rows[0]:
            continue
        print(f"\n--- {mdl} ---")
        for blk in ["all", "mitbih", "incart"]:
            v = {m: np.array([r[mdl][blk][m] for r in rows if r[mdl].get(blk)])
                 for m in ["f1_macro", "S", "V", "F"]}
            if not len(v["F"]):
                continue
            print(f"  [{blk}] " + "  ".join(
                f"{m} {v[m].mean():.3f}±{v[m].std():.3f}" for m in ["f1_macro", "S", "V", "F"]))
    json.dump({"tag": tag, "runs": rows}, open(out_dir / f"summary_{tag}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sentinel", choices=["sentinel", "kfold"])
    ap.add_argument("--v1", action="store_true")
    ap.add_argument("--rhythm", action="store_true")
    ap.add_argument("--grl", action="store_true")
    ap.add_argument("--loss", default="ce", choices=["ce", "balanced"])
    ap.add_argument("--grl-max", type=float, default=0.2)
    ap.add_argument("--grl-ramp", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--inner-val-frac", type=float, default=0.15)
    ap.add_argument("--tuning-frac", type=float, default=0.0,
                    help="Deprecated: tuning is the inner-val split; no extra records are held out.")
    ap.add_argument("--conformal-cal-frac", type=float, default=0.15,
                    help="Record fraction held out exclusively for conformal/risk calibration.")
    ap.add_argument("--train-n-cap", type=int, default=35000)
    ap.add_argument("--amp", type=int, default=1, help="1=mixed precision (CUDA), 0=fp32")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--out-dir", default="results/mp_lopo")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ckpt = out_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    tag = f"{'v' if args.v1 else ''}{'r' if args.rhythm else ''}{'g' if args.grl else ''}{args.loss}"

    d = np.load(POOL)
    data = (d["X"], d["X2"], d["y"], d["rr"], d["record"], d["source"])
    y, record, source = d["y"], d["record"], d["source"]
    rec_cls = {int(r): np.bincount(y[record == r], minlength=5).astype(float) for r in np.unique(record)}
    rec_src = {int(r): int(source[record == r][0]) for r in np.unique(record)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  tag: {tag}")

    if args.mode == "sentinel":
        tr_i, va_i, te_i = build_sentinel_idx(record, rec_cls, rec_src, args)
        # Sentinel is screening only; still make the decision splits explicit.
        tr_records = sorted(set(record[tr_i].tolist()))
        conformal = carve_conformal_s_cal(tr_records, rec_cls, args.conformal_cal_frac, args.fold_seed + 200)
        train = set(tr_records) - conformal
        idx = (np.where(np.isin(record, list(train)))[0], va_i, te_i,
               va_i, np.where(np.isin(record, list(conformal)))[0])
        res = train_fold("sentinel", idx, data, device, args, ckpt)
        json.dump(res, open(out_dir / f"sentinel_{tag}.json", "w"), indent=2)
        print(f"\n[BENCHMARK] one fold ({args.epochs} ep) = {res['train_time_sec']/60:.0f} min")
        m = res.get("swa_model", res["best_epoch_model"])
        print(f"[SENTINEL {tag}] all F1_F={m['all']['F']:.3f} F1_S={m['all']['S']:.3f} "
              f"N={m['all']['N']:.3f} V={m['all']['V']:.3f} | "
              f"mitbih F1_F={m['mitbih']['F']:.3f} F1_S={m['mitbih']['S']:.3f}")
        return

    fold_of = assign_folds(rec_cls, rec_src, args.k, args.fold_seed)
    from collections import defaultdict
    fr = defaultdict(list)
    for r, f in fold_of.items():
        fr[f].append(r)
    seeds = args.seeds or [args.seed]
    todo = args.folds if args.folds is not None else list(range(args.k))
    for seed in seeds:
        args.seed = seed
        for f in todo:
            rp = out_dir / f"fold{f}_seed{seed}_{tag}.json"
            if rp.exists():
                print(f"[skip] fold {f} seed {seed}"); continue
            te = set(fr[f]); trval = [r for r in fold_of if r not in te]
            va = carve_inner_val(trval, rec_cls, rec_src, args.inner_val_frac, args.fold_seed + f)
            tr = set(trval) - va
            conformal = carve_conformal_s_cal(sorted(tr), rec_cls, args.conformal_cal_frac,
                                               args.fold_seed + 2000 + f)
            train = tr - conformal
            idx = (np.where(np.isin(record, list(train)))[0],
                   np.where(np.isin(record, list(va)))[0],
                   np.where(np.isin(record, list(te)))[0],
                   np.where(np.isin(record, list(va)))[0],
                   np.where(np.isin(record, list(conformal)))[0])
            print(f"\n{'='*70}\n  FOLD {f} SEED {seed}\n{'='*70}")
            res = train_fold(f, idx, data, device, args, ckpt)
            json.dump(res, open(rp, "w"), indent=2)
    summarize(out_dir, tag, args.k)


if __name__ == "__main__":
    main()
