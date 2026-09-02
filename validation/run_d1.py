"""D1 - HMR-MultiPath R2 (v1+rhythm) on the official inter-patient DS1/DS2 split,
3 seeds, for the literature comparison table.

Data: data/processed/splits_rr_2ch/inter_{train,val,test}.npz
      X (n,187,2): ch0 = MLII (model x), ch1 = V1 (model x2); already z-scored.
      rr (n,3): already z-scored. y (n,): AAMI 0..4.

Recipe: reuses the exact training helpers the LOPO trainer uses
(`train_multipath_lopo.fgsm_primary` / `predict_mp_logits` / `recompute_bn`,
`train_inter_patient.CONFIG` / `set_seed` / `cosine_lr`). The epoch loop is the
same as `train_multipath_lopo.train_fold`: FGSM adv on the last `adv_ratio` of
each batch, MultiPathLoss (CE + smoothness), grad clip, cosine LR, best-epoch
model selection on inner-val {S,V,F} macro-F1, tail-epoch SWA reported alongside.

Run: python -u -m validation.run_d1
Out: results/d1_ds12/seed{N}.json, results/d1_ds12/summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr_multipath import HMRMultiPath, MultiPathLoss  # noqa: E402
from train_inter_patient import CONFIG, set_seed, cosine_lr  # noqa: E402
from train_multipath_lopo import fgsm_primary, predict_mp_logits, recompute_bn  # noqa: E402

DATA = Path("data/processed/splits_rr_2ch")
OUT = Path("results/d1_ds12")
CLS = ["N", "S", "V", "F", "Q"]
BASELINE = {"N": 0.964, "S": 0.43, "V": 0.87, "F": 0.24}  # DS1/DS2 flat baseline, context only


def load(split):
    d = np.load(DATA / f"inter_{split}.npz")
    X = d["X"].astype(np.float32)
    return X[:, :, 0:1], X[:, :, 1:2], d["rr"].astype(np.float32), d["y"].astype(np.int64)


def class_weights(y):
    c = np.bincount(y, minlength=5).astype(np.float64)
    c[c == 0] = 1e-9
    return torch.tensor(np.clip(c.sum() / (5 * c), 0.5, 50.0), dtype=torch.float32)


def score(y, p):
    fpc = f1_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {"f1_macro": float(f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            **{CLS[i]: float(fpc[i]) for i in range(4)}}


def run_seed(seed, epochs, device):
    set_seed(seed)
    xtr, x2tr, rrtr, ytr = load("train")
    xva, x2va, rrva, yva = load("val")
    xte, x2te, rrte, yte = load("test")
    use_amp = device.type == "cuda"
    cw = class_weights(ytr).to(device)
    model = HMRMultiPath(hidden_size=CONFIG["hidden_size"], dropout=CONFIG["dropout"],
                         cnn_out_channels=CONFIG["cnn_out_channels"], num_layers=2,
                         use_v1=True, use_rhythm=True, use_grl=False, rr_dim=3).to(device)
    crit = MultiPathLoss(class_weights=cw, lambda_smooth=CONFIG["lambda_smooth"], loss="ce").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(x2tr),
                                  torch.from_numpy(rrtr), torch.from_numpy(ytr)),
                    batch_size=CONFIG["batch_size"], shuffle=True)

    swa_start = int(epochs * 0.8)
    swa_sum, swa_n = None, 0
    best_v, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        lr = cosine_lr(ep - 1, epochs, CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        for xb, x2b, rrb, yb in dl:
            xb, x2b, rrb, yb = (t.to(device) for t in (xb, x2b, rrb, yb))
            sp = int(len(xb) * (1 - CONFIG["adv_ratio"]))
            if sp < len(xb):
                xa = fgsm_primary(model, xb[sp:], x2b[sp:], rrb[sp:], yb[sp:],
                                  CONFIG["adv_epsilon"], use_amp)
                xb = torch.cat([xb[:sp], xa])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit, intr = model(xb, x2b, rrb, yb, grl_lambda=0.0, return_internals=True)
                loss, _ = crit(logit, yb, intr, None)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt)
            scaler.update()
        vp = predict_mp_logits(model, xva, x2va, rrva, device, amp=use_amp).argmax(1)
        vs = f1_score(yva, vp, labels=[1, 2, 3], average="macro", zero_division=0)
        if vs > best_v:
            best_v, best_state = vs, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep >= swa_start:
            pd = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
            swa_sum = pd if swa_sum is None else {k: swa_sum[k] + pd[k] for k in swa_sum}
            swa_n += 1
        print(f"  seed{seed} ep{ep:2d} lr={lr:.5f} inner-val F1(S,V,F)={vs:.4f}"
              f"{'  <-- best' if vs == best_v else ''}", flush=True)

    model.load_state_dict(best_state)
    res = {"seed": seed, "best_inner_val_svf": float(best_v),
           "train_time_h": (time.time() - t0) / 3600,
           "best_epoch_model": score(yte, predict_mp_logits(model, xte, x2te, rrte, device,
                                                            amp=use_amp).argmax(1))}
    if swa_n > 0:
        swa = copy.deepcopy(model)
        swa.load_state_dict(best_state)
        with torch.no_grad():
            for k, p in swa.named_parameters():
                if k in swa_sum:
                    p.copy_((swa_sum[k] / swa_n).to(device))
        recompute_bn(swa, ((torch.from_numpy(xtr[i:i + 512]), torch.from_numpy(x2tr[i:i + 512]),
                            torch.from_numpy(rrtr[i:i + 512]), None, None)
                           for i in range(0, len(xtr), 512)), device)
        res["swa_model"] = score(yte, predict_mp_logits(swa, xte, x2te, rrte, device,
                                                        amp=use_amp).argmax(1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=45)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for s in args.seeds:
        rp = OUT / f"seed{s}.json"
        if rp.exists():
            rows.append(json.loads(rp.read_text()))
            print(f"[skip] seed {s}")
            continue
        print(f"==== D1 seed {s} ====", flush=True)
        r = run_seed(s, args.epochs, device)
        rp.write_text(json.dumps(r, indent=2))
        rows.append(r)

    agg = {}
    for k in ("f1_macro", "N", "S", "V", "F"):
        v = np.array([r["best_epoch_model"][k] for r in rows])
        agg[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                  "per_seed": [round(float(x), 4) for x in v]}
    (OUT / "summary.json").write_text(json.dumps(
        {"n_seeds": len(rows), "epochs": args.epochs, "agg": agg,
         "ds12_flat_baseline_context": BASELINE}, indent=2))
    print("\n=== D1 SUMMARY (R2 v1+rhythm, official DS1/DS2, best-epoch, 3 seeds) ===")
    for k in ("f1_macro", "N", "S", "V", "F"):
        b = f"  (DS1/DS2 flat ~{BASELINE[k]})" if k in BASELINE else ""
        print(f"  {k:9s} {agg[k]['mean']:.4f} +- {agg[k]['std']:.4f}{b}")
    print(f"[OK] {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
