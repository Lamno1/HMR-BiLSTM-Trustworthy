"""
train_crossdb_multiseed.py  (v2, 2026-08-28)
===========================================
Stage A of the cross-database plan (see memory project_fclass_root_cause_diagnostics).

Plain HMR-BiLSTM, architecture and recipe BYTE-FOR-BYTE the official
train_inter_patient.py (same functions imported directly), trained on the fixed-v2
cross-database split (data/processed/splits_rr_crossdb/), model-selected on the
patient-diverse validation set, evaluated on BOTH:
  - MIT-BIH DS2   (primary, re-extracted, numerically identical to the paper's test set)
  - INCART held-out patients (secondary cross-database generalization)

SINGLE VARIABLE vs the original-split baseline: the training + validation DATA.
Everything else — model, cosine LR, FGSM adv-training (deterministic clean/adv split,
task-loss-only gradient, single concatenated forward/backward), Focal gamma=1.5,
class weights, lambda_smooth, epochs=45, patience=15 — comes from train_inter_patient.

Success criterion: DS2 F1-F / F1-S seed-to-seed VARIANCE shrinks vs the wild
orig-split spread (F .04-.24, S .19-.43) with the mean held or improved.

Resumable at EPOCH granularity (machine reboots have killed 2 earlier launches):
each epoch writes results/crossdb_multiseed/checkpoints/crossdb_seed{S}_state.pt;
a re-run picks up from the last completed epoch.

    python train_crossdb_multiseed.py --seeds 42 123 456
    python train_crossdb_multiseed.py --summarize-only
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

from hmr_bilstm import RLSTMClassifier, RLSTMLoss
from train_inter_patient import (
    CONFIG, ECGDataset, set_seed, cosine_lr, train_one_epoch, evaluate,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
DATA_DIR = Path("data/processed/splits_rr_crossdb")

# original-split baseline (full official recipe), per seed -- results/logs + ensemble logs
ORIG_SPLIT_BASELINE = {
    42:  {"f1_macro": 0.6279, "N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411},
    123: {"f1_macro": 0.6137, "N": 0.9653, "S": 0.3600, "V": 0.8609, "F": 0.2688},
    456: {"f1_macro": 0.5109, "N": 0.9580, "S": 0.1940, "V": 0.8543, "F": 0.0371},
}


def class_weights_from(y, num_classes=5):
    c = np.bincount(y, minlength=num_classes).astype(np.float64)
    c = np.where(c == 0, 1e-9, c)
    w = np.clip(c.sum() / (num_classes * c), 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


@torch.no_grad()
def eval_test(model, loader, device, tag):
    model.eval()
    preds, ys = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    acc = accuracy_score(ys, preds)
    fm = f1_score(ys, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    fpc = f1_score(ys, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    pc = {CLASS_NAMES[i]: float(fpc[i]) for i in range(4)}
    print(f"  [{tag}] acc={acc:.4f}  F1_macro={fm:.4f}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in pc.items()))
    print(classification_report(ys, preds, labels=[0, 1, 2, 3],
                                target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {"accuracy": float(acc), "f1_macro": float(fm), "f1_per_class": pc}


def train_one_seed(seed, loaders, device, ckpt_dir):
    cfg = CONFIG.copy()
    set_seed(seed)
    train_loader, val_loader, ds2_loader, inc_loader = loaders

    cw = class_weights_from(train_loader.dataset.y.numpy(), cfg["num_classes"]).to(device)
    print(f"  class weights: {cw.cpu().numpy().round(3)}")

    model = RLSTMClassifier(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"], num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"], num_layers=cfg["num_layers"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = RLSTMLoss(lambda_smooth=cfg["lambda_smooth"], class_weights=cw,
                          use_focal=cfg["use_focal_loss"], focal_gamma=cfg["focal_gamma"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])

    state_path = ckpt_dir / f"crossdb_seed{seed}_state.pt"
    start_epoch, best_f1, best_epoch, patience_cnt = 1, 0.0, 0, 0
    best_state = None
    elapsed_prev = 0.0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state"])
        opt.load_state_dict(st["opt_state"])
        start_epoch = st["epoch"] + 1
        best_f1, best_epoch, patience_cnt = st["best_f1"], st["best_epoch"], st["patience_cnt"]
        best_state = st["best_state"]
        elapsed_prev = st.get("elapsed", 0.0)
        print(f"  [resume] seed {seed} from epoch {start_epoch} "
              f"(best F1_macro={best_f1:.4f} @ep{best_epoch})")

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr

        tstats = train_one_epoch(
            model, train_loader, opt, criterion, device, cfg["grad_clip"],
            adv_training=cfg["adversarial_training"],
            adv_epsilon=cfg["adv_epsilon"], adv_ratio=cfg["adv_ratio"],
        )
        vm = evaluate(model, val_loader, device, cfg["num_classes"])
        vf1 = vm["f1_macro"]
        improved = vf1 > best_f1
        if improved:
            best_f1, best_epoch, patience_cnt = vf1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        print(f"  [seed={seed}] ep{epoch:2d} | lr={lr:.5f} | loss={tstats['loss']:.4f} "
              f"| val F1_macro={vf1:.4f} auc={vm.get('auc_ovr', float('nan')):.4f}"
              f"{'  <-- best' if improved else ''}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "best_f1": best_f1, "best_epoch": best_epoch, "patience_cnt": patience_cnt,
            "best_state": best_state, "seed": seed, "n_params": n_params,
            "elapsed": elapsed_prev + (time.time() - t0),
        }, state_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"  [seed={seed}] early stop ep{epoch} (best ep{best_epoch})")
            break

    train_time = elapsed_prev + (time.time() - t0)
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "seed": seed, "epoch": best_epoch,
                "val_f1_macro": best_f1, "n_params": n_params},
               ckpt_dir / f"crossdb_hmr_seed{seed}.pt")
    print(f"\n  [seed={seed}] best val F1_macro={best_f1:.4f} @ep{best_epoch}  "
          f"train_time={train_time:.0f}s")

    results = {"seed": seed, "n_params": n_params, "best_epoch": best_epoch,
               "best_val_f1_macro": float(best_f1), "train_time_sec": train_time}
    results["test_mitbih_ds2"] = eval_test(model, ds2_loader, device, "test_mitbih_ds2")
    results["test_incart"] = eval_test(model, inc_loader, device, "test_incart")
    return results


def summarize(out_dir, seeds):
    rows = [json.load(open(out_dir / f"seed_{s}.json"))
            for s in seeds if (out_dir / f"seed_{s}.json").exists()]
    if not rows:
        print("[summary] no seeds done")
        return
    print(f"\n{'='*94}\n  CROSS-DB HMR-BiLSTM (v2)  --  {len(rows)}/{len(seeds)} seeds\n{'='*94}")
    for tag in ["test_mitbih_ds2", "test_incart"]:
        if tag not in rows[0]:
            continue
        print(f"\n--- {tag} ---")
        print(f"{'metric':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}   "
              f"(orig-split baseline mean [min,max])")
        for k in ["f1_macro", "N", "S", "V", "F"]:
            vals = np.array([r[tag]["f1_macro"] if k == "f1_macro"
                             else r[tag]["f1_per_class"][k] for r in rows])
            ref = ""
            if tag == "test_mitbih_ds2" and all(r["seed"] in ORIG_SPLIT_BASELINE for r in rows):
                b = np.array([ORIG_SPLIT_BASELINE[r["seed"]][k] for r in rows])
                ref = f"   {b.mean():.4f} [{b.min():.3f},{b.max():.3f}]  (spread {b.max()-b.min():.3f})"
            print(f"{k:<10} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} "
                  f"{vals.max():8.4f}{ref}")
        if tag == "test_mitbih_ds2":
            fvar = np.array([r[tag]["f1_per_class"]["F"] for r in rows])
            svar = np.array([r[tag]["f1_per_class"]["S"] for r in rows])
            print(f"\n  >>> F1-F spread this run = {fvar.max()-fvar.min():.3f} "
                  f"(orig-split spread 0.232) | F1-S spread = {svar.max()-svar.min():.3f} "
                  f"(orig-split spread 0.240)")
    summary = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
               "per_seed": {r["seed"]: {k: r.get(k) for k in
                            ["test_mitbih_ds2", "test_incart", "best_val_f1_macro", "best_epoch"]}
                            for r in rows},
               "orig_split_baseline": ORIG_SPLIT_BASELINE}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\n[OK] {out_dir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    ap.add_argument("--out-dir", default="results/crossdb_multiseed")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir, args.seeds)
        return

    if not (DATA_DIR / "inter_train.npz").exists():
        print(f"[ERROR] {DATA_DIR}/inter_train.npz missing -- run "
              f"validation/preprocess_aami_rr_crossdb.py first")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = CONFIG["batch_size"]
    train_ds = ECGDataset(DATA_DIR / "inter_train.npz")
    val_ds = ECGDataset(DATA_DIR / "inter_val.npz")
    ds2_ds = ECGDataset(DATA_DIR / "inter_test.npz")
    inc_ds = ECGDataset(DATA_DIR / "incart_test.npz")
    print(f"Device: {device}")
    print(f"train {tuple(train_ds.X.shape)} {np.bincount(train_ds.y.numpy())}")
    print(f"val   {tuple(val_ds.X.shape)} {np.bincount(val_ds.y.numpy())}")
    print(f"test DS2 {tuple(ds2_ds.X.shape)} {np.bincount(ds2_ds.y.numpy())}")
    print(f"test INCART {tuple(inc_ds.X.shape)} {np.bincount(inc_ds.y.numpy())}")

    loaders = (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False),
        DataLoader(ds2_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False),
        DataLoader(inc_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False),
    )

    for s in args.seeds:
        rp = out_dir / f"seed_{s}.json"
        if rp.exists():
            print(f"[skip] seed {s} done")
            continue
        print(f"\n{'='*70}\n  seed {s}\n{'='*70}")
        res = train_one_seed(s, loaders, device, ckpt_dir)
        json.dump(res, open(rp, "w"), indent=2)
        print(f"[OK] {rp}")

    summarize(out_dir, args.seeds)


if __name__ == "__main__":
    main()
