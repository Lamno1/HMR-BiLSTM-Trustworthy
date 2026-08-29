"""
train_hd_ladder.py  —  ladder step 1-2 of the F/S breakthrough plan.
===================================================================
Same data (data/processed/splits/, the OFFICIAL plain inter-patient split) and
same recipe as train_inter_patient.py (functions imported directly: cosine LR,
FGSM adv-training, epochs=45, patience=15, lambda_smooth, cudnn.deterministic).

SINGLE VARIABLE vs the official baseline: the decision HEAD.
  --variant hier     : flat 5-way softmax  ->  soft hierarchical head
                       (loss kept identical: focal gamma=1.5 + class weights)
  --variant hier_la  : hierarchical head + balanced-softmax / logit adjustment
                       on the [S,V,F,Q] sub-head (tau*log prior), focal OFF,
                       class weights OFF  -> the principled prior correction

Official baseline (data/processed/splits/, full recipe), per seed:
  42:  macro .6279  N .9643  S .4335  V .8728  F .2411
  123: macro .6137  N .9653  S .3600  V .8609  F .2688
  456: macro .5109  N .9580  S .1940  V .8543  F .0371

Resumable at epoch granularity (per-epoch state ckpt) — machine reboots.

    python train_hd_ladder.py --variant hier    --seeds 42 123 456
    python train_hd_ladder.py --variant hier_la --seeds 42 123 456
    python train_hd_ladder.py --variant hier --summarize-only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hmr_hd import RLSTMClassifierHD, HDLoss
from train_inter_patient import CONFIG, ECGDataset, set_seed, cosine_lr, train_one_epoch, evaluate

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
DATA_DIR = Path("data/processed/splits")

OFFICIAL_BASELINE = {
    42:  {"f1_macro": 0.6279, "N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411},
    123: {"f1_macro": 0.6137, "N": 0.9653, "S": 0.3600, "V": 0.8609, "F": 0.2688},
    456: {"f1_macro": 0.5109, "N": 0.9580, "S": 0.1940, "V": 0.8543, "F": 0.0371},
}


def class_weights_from(y, k=5):
    c = np.bincount(y, minlength=k).astype(np.float64)
    c = np.where(c == 0, 1e-9, c)
    return torch.from_numpy(np.clip(c.sum() / (k * c), 0.5, 50.0).astype(np.float32))


def sub_log_prior(y):
    """log prior over [S,V,F,Q] (labels 1..4) from train y — for logit adjustment."""
    c = np.bincount(y, minlength=5).astype(np.float64)[1:5]
    p = c / c.sum()
    return torch.from_numpy(np.log(np.clip(p, 1e-8, 1)).astype(np.float32))


@torch.no_grad()
def eval_test(model, loader, device, tag):
    model.eval()
    preds, ys = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    fm = f1_score(ys, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    fpc = f1_score(ys, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    pc = {CLASS_NAMES[i]: float(fpc[i]) for i in range(4)}
    print(f"  [{tag}] acc={accuracy_score(ys, preds):.4f}  F1_macro={fm:.4f}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in pc.items()))
    print(classification_report(ys, preds, labels=[0, 1, 2, 3],
                                target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {"f1_macro": float(fm), "f1_per_class": pc}


def train_one_seed(seed, variant, loaders, device, ckpt_dir):
    cfg = CONFIG.copy()
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    ytr = train_loader.dataset.y.numpy()

    if variant == "hier":
        cw = class_weights_from(ytr).to(device)
        crit = HDLoss(lambda_smooth=cfg["lambda_smooth"], class_weights=cw,
                      focal_gamma=cfg["focal_gamma"], logit_adjust=None).to(device)
        print(f"  variant=hier  class weights={cw.cpu().numpy().round(2)}")
    elif variant == "hier_la":
        la = sub_log_prior(ytr).to(device)
        crit = HDLoss(lambda_smooth=cfg["lambda_smooth"], class_weights=None,
                      focal_gamma=0.0, logit_adjust=la).to(device)
        print(f"  variant=hier_la  sub log-prior [S,V,F,Q]={la.cpu().numpy().round(3)}")
    else:
        raise ValueError(variant)

    model = RLSTMClassifierHD(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
        num_classes=5, cnn_out_channels=cfg["cnn_out_channels"], num_layers=cfg["num_layers"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])

    state_path = ckpt_dir / f"hd_{variant}_seed{seed}_state.pt"
    start_epoch, best_f1, best_epoch, patience_cnt, best_state, elapsed_prev = 1, 0.0, 0, 0, None, 0.0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state"]); opt.load_state_dict(st["opt_state"])
        start_epoch = st["epoch"] + 1
        best_f1, best_epoch, patience_cnt = st["best_f1"], st["best_epoch"], st["patience_cnt"]
        best_state, elapsed_prev = st["best_state"], st.get("elapsed", 0.0)
        print(f"  [resume] {variant} seed {seed} from epoch {start_epoch} (best {best_f1:.4f} @ep{best_epoch})")

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        ts = train_one_epoch(model, train_loader, opt, crit, device, cfg["grad_clip"],
                             adv_training=cfg["adversarial_training"],
                             adv_epsilon=cfg["adv_epsilon"], adv_ratio=cfg["adv_ratio"])
        vm = evaluate(model, val_loader, device, 5)
        vf1 = vm["f1_macro"]
        improved = vf1 > best_f1
        if improved:
            best_f1, best_epoch, patience_cnt = vf1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        print(f"  [{variant} s{seed}] ep{epoch:2d} | lr={lr:.5f} | loss={ts['loss']:.4f} "
              f"| val F1_macro={vf1:.4f}{'  <-- best' if improved else ''}")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "best_f1": best_f1, "best_epoch": best_epoch, "patience_cnt": patience_cnt,
                    "best_state": best_state, "seed": seed, "n_params": n_params,
                    "elapsed": elapsed_prev + (time.time() - t0)}, state_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"  [{variant} s{seed}] early stop ep{epoch} (best ep{best_epoch})")
            break

    train_time = elapsed_prev + (time.time() - t0)
    model.load_state_dict(best_state)
    print(f"\n  [{variant} s{seed}] best val F1_macro={best_f1:.4f} @ep{best_epoch}  {train_time:.0f}s")
    res = {"seed": seed, "variant": variant, "n_params": n_params, "best_epoch": best_epoch,
           "best_val_f1_macro": float(best_f1), "train_time_sec": train_time}
    res["test"] = eval_test(model, test_loader, device, f"{variant}_test_ds2")
    return res


def summarize(out_dir, variant, seeds):
    rows = [json.load(open(out_dir / f"{variant}_seed_{s}.json"))
            for s in seeds if (out_dir / f"{variant}_seed_{s}.json").exists()]
    if not rows:
        print(f"[summary] no {variant} seeds done"); return
    print(f"\n{'='*90}\n  HD ladder [{variant}]  --  {len(rows)}/{len(seeds)} seeds  (DS2, data/processed/splits)\n{'='*90}")
    print(f"{'metric':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'spread':>8}   official mean[min,max] spread")
    for k in ["f1_macro", "N", "S", "V", "F"]:
        v = np.array([r["test"]["f1_macro"] if k == "f1_macro" else r["test"]["f1_per_class"][k] for r in rows])
        ref = ""
        if all(r["seed"] in OFFICIAL_BASELINE for r in rows):
            b = np.array([OFFICIAL_BASELINE[r["seed"]][k] for r in rows])
            ref = f"   {b.mean():.4f}[{b.min():.3f},{b.max():.3f}] {b.max()-b.min():.3f}"
        print(f"{k:<10} {v.mean():8.4f} {v.std():8.4f} {v.min():8.4f} {v.max():8.4f} {v.max()-v.min():8.4f}{ref}")
    json.dump({"variant": variant, "seeds": [r["seed"] for r in rows],
               "per_seed": {r["seed"]: r["test"] for r in rows},
               "official_baseline": OFFICIAL_BASELINE},
              open(out_dir / f"summary_{variant}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["hier", "hier_la"], required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    ap.add_argument("--out-dir", default="results/hd_ladder")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir, args.variant, args.seeds); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = CONFIG["batch_size"]
    tr = ECGDataset(DATA_DIR / "inter_train.npz")
    va = ECGDataset(DATA_DIR / "inter_val.npz")
    te = ECGDataset(DATA_DIR / "inter_test.npz")
    print(f"Device: {device}")
    print(f"train {tuple(tr.X.shape)} {np.bincount(tr.y.numpy())}")
    print(f"val   {tuple(va.X.shape)} {np.bincount(va.y.numpy())}")
    print(f"test  {tuple(te.X.shape)} {np.bincount(te.y.numpy())}")
    loaders = (DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0),
               DataLoader(va, batch_size=bs, shuffle=False, num_workers=0),
               DataLoader(te, batch_size=bs, shuffle=False, num_workers=0))

    for s in args.seeds:
        rp = out_dir / f"{args.variant}_seed_{s}.json"
        if rp.exists():
            print(f"[skip] {args.variant} seed {s} done"); continue
        print(f"\n{'='*70}\n  {args.variant}  seed {s}\n{'='*70}")
        res = train_one_seed(s, args.variant, loaders, device, ckpt_dir)
        json.dump(res, open(rp, "w"), indent=2)
        print(f"[OK] {rp}")
    summarize(out_dir, args.variant, args.seeds)


if __name__ == "__main__":
    main()
