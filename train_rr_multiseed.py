"""
train_rr_multiseed.py
=======================
Multi-seed reliability check for the HMR-BiLSTM + RR-ratio prototype.

Trains the SAME config as train_rr_prototype.py (data/processed/splits_rr,
same hyperparameters) across several seeds, then reports mean +/- std for
F1(S), F1(V), F1(F), F1-macro, AUC, Accuracy — so the single-run numbers
(F1(S)=0.4314 etc.) can be reported as a trustworthy range instead of one
possibly-lucky draw.

Resumable: each seed's result is saved to its own file immediately after
that seed finishes. Re-running the script skips seeds that already have a
saved result, so an interrupted run (e.g. killed background process) can
just be re-invoked to pick up where it left off.

Cach chay:
    # Train all 5 default seeds (skips any already completed):
    python train_rr_multiseed.py

    # Custom seed list:
    python train_rr_multiseed.py --seeds 42 123 456

    # Only recompute the mean/std summary from whatever seeds are done so far
    # (no training), useful to check progress on a partially-completed run:
    python train_rr_multiseed.py --summarize-only

Results:
    results/rr_prototype_multiseed/seed_{seed}_results.json   (per seed)
    results/rr_prototype_multiseed/checkpoints/best_rlstm_rr_seed{seed}.pt
    results/rr_prototype_multiseed/summary.json               (mean/std)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from hmr_bilstm_rr import RLSTMClassifierRR
from hmr_bilstm_ablation import RLSTMLoss
from train_rr_prototype import (
    CONFIG, set_seed, cosine_lr, load_data, evaluate, train_one_epoch,
)

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

BASELINE_NO_RR = {"f1_S": 0.2674, "f1_V": 0.8689, "f1_F": 0.2958, "f1_macro": 0.5964, "auc_ovr": 0.9213}
SINGLE_RUN_RR_RATIO = {"f1_S": 0.4314, "f1_V": 0.8457, "f1_F": 0.1841, "f1_macro": 0.6043, "auc_ovr": 0.9522}


def train_one_seed(seed, cfg, device, train_loader, val_loader, test_loader,
                    class_weights, ckpt_dir):
    set_seed(seed)

    model = RLSTMClassifierRR(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
        n_rr_features=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = RLSTMLoss(
        lambda_smooth=cfg["lambda_smooth"],
        class_weights=class_weights,
        use_focal=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )

    ckpt_path = ckpt_dir / f"best_rlstm_rr_seed{seed}.pt"
    best_f1, best_epoch, patience_cnt = 0.0, 0, 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        marker = ""
        if val_m["f1_macro"] > best_f1:
            best_f1, best_epoch, patience_cnt = val_m["f1_macro"], epoch, 0
            torch.save({
                "model_state": model.state_dict(), "config": cfg, "seed": seed,
                "epoch": epoch, "val_f1_macro": best_f1, "n_params": n_params,
            }, ckpt_path)
            marker = " <-- best"
        else:
            patience_cnt += 1

        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "val_f1_macro": val_m["f1_macro"], "val_accuracy": val_m["accuracy"],
        })
        print(f"  [seed={seed}] ep{epoch:3d} | loss={train_loss:.4f} | "
              f"val_f1={val_m['f1_macro']:.4f} acc={val_m['accuracy']:.4f} | "
              f"{elapsed:.0f}s{marker}")

        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"  [seed={seed}] [Early stop] ep={epoch} best={best_epoch} F1={best_f1:.4f}")
            break

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    test_m = evaluate(model, test_loader, device)
    report = classification_report(
        test_m["_y_true"], test_m["_preds"],
        target_names=["N", "S", "V", "F", "Q"], zero_division=0, digits=4,
    )
    clean_test = {k: v for k, v in test_m.items() if not k.startswith("_")}

    return {
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_metrics": clean_test,
        "report": report,
        "history": history,
    }


def summarize(out_dir, seeds):
    rows = []
    for seed in seeds:
        p = out_dir / f"seed_{seed}_results.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))

    if not rows:
        print("[SUMMARY] No completed seeds yet.")
        return None

    keys = ["accuracy", "f1_macro", "auc_ovr", "f1_S", "f1_V", "f1_F"]
    agg = {k: np.array([r["test_metrics"][k] for r in rows]) for k in keys}

    print(f"\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY  ({len(rows)}/{len(seeds)} seeds completed)")
    print(f"{'='*70}")
    print(f"{'Metric':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  | baseline(no-RR) | single-run(RR)")
    for k in keys:
        vals = agg[k]
        base_key = "auc_ovr" if k == "auc_ovr" else k
        base_val = BASELINE_NO_RR.get(base_key, float("nan"))
        single_val = SINGLE_RUN_RR_RATIO.get(base_key, float("nan"))
        print(f"{k:<12} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} {vals.max():8.4f}  | "
              f"{base_val:14.4f} | {single_val:12.4f}")
    print(f"\nSeeds included: {[r['seed'] for r in rows]}")

    summary = {
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "mean": {k: float(agg[k].mean()) for k in keys},
        "std": {k: float(agg[k].std()) for k in keys},
        "min": {k: float(agg[k].min()) for k in keys},
        "max": {k: float(agg[k].max()) for k in keys},
        "per_seed": {r["seed"]: r["test_metrics"] for r in rows},
        "baseline_no_rr": BASELINE_NO_RR,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved -> {out_dir / 'summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed reliability check for RR-ratio prototype")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", default="results/rr_prototype_multiseed")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Skip training; just aggregate whatever seed results already exist.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize(out_dir, args.seeds)
        return

    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds to run: {args.seeds}")

    data_dir = Path(cfg["data_dir"])
    if not (data_dir / "inter_train.npz").exists():
        print(f"[ERROR] {data_dir}/inter_train.npz not found. "
              f"Run: python validation/preprocess_aami_rr.py first.")
        return

    print("\n[Loading data once, shared across all seeds]")
    train_loader, val_loader, test_loader = load_data(cfg["data_dir"], cfg["batch_size"])

    class_weights = None
    if cfg["use_class_weights"]:
        y_inter_tr = np.load(f"{cfg['data_dir']}/inter_train.npz")["y"]
        counts = np.bincount(y_inter_tr, minlength=cfg["num_classes"]).astype(np.float64)
        counts = np.where(counts == 0, 1e-9, counts)
        cw_arr = counts.sum() / (float(cfg["num_classes"]) * counts)
        cw_arr = np.clip(cw_arr, 0.5, 50.0).astype(np.float32)
        class_weights = torch.from_numpy(cw_arr).float().to(device)
        print(f"Class weights (inter_train): {cw_arr}")

    for seed in args.seeds:
        result_path = out_dir / f"seed_{seed}_results.json"
        if result_path.exists():
            print(f"\n[SKIP] seed={seed} already done -> {result_path}")
            continue

        print(f"\n{'='*70}\n  Training seed={seed}\n{'='*70}")
        result = train_one_seed(seed, cfg, device, train_loader, val_loader, test_loader,
                                 class_weights, ckpt_dir)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[OK] seed={seed} -> F1(S)={result['test_metrics']['f1_S']:.4f}  "
              f"F1(V)={result['test_metrics']['f1_V']:.4f}  "
              f"F1(F)={result['test_metrics']['f1_F']:.4f}  "
              f"F1-macro={result['test_metrics']['f1_macro']:.4f}")
        print(f"[OK] Saved -> {result_path}")

    summarize(out_dir, args.seeds)


if __name__ == "__main__":
    main()
