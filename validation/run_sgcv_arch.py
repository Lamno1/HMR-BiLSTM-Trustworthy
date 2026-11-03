"""Matched-recipe recurrent cores under the controlled subject-grouped CV (C1_MIT_SUBJECT_CV_V1).

Does validation blindness on DS1/DS2 (inner-validation macro-F1 does not rank models by test
macro-F1) persist when both validation and test are subject-grouped and repeated over five
folds? Folds, inner validation groups, N control and normalisation are rebuilt with the
locked protocol of run_controlled_mit_subject_cv.py (verified against its manifest); the
recurrent core and recipe override come from train_matched_recipe.ARCHS, so `bilstm`,
`normc`, `normc_nodrop`, `bilstm_nocw`, ... are the same models as in the DS1/DS2 controls.
The existing flat runs in results/controlled_mit_subject_cv are the `full` arch.

Saves the best-state logits for both the inner-validation and the outer-test beats.
Resumable per epoch.

    python -m validation.run_sgcv_arch --arch bilstm
    python -m validation.run_sgcv_arch --arch normc_nodrop --seeds 42 --folds 0 --smoke-batches 3 --out-dir <tmp>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from hmr_bilstm import RLSTMLoss
from train_inter_patient import CONFIG, cosine_lr, set_seed, train_one_epoch
from train_matched_recipe import ARCHS, BUILDERS
from validation.run_controlled_mit_subject_cv import (PROTOCOL_ID, build_protocol, class_weights,
                                                      clone_state_dict, controlled_train_indices,
                                                      metrics, normalize_from_train,
                                                      predict_logits)

ROOT = Path(__file__).resolve().parents[1]
LOCKED_MANIFEST = ROOT / "results" / "controlled_mit_subject_cv" / "protocol_manifest.json"


def check_protocol(manifest: dict) -> None:
    locked = json.load(open(LOCKED_MANIFEST, encoding="utf-8"))
    for a, b in zip(manifest["folds"], locked["folds"]):
        for key in ("train_groups", "val_groups", "test_groups", "val_counts", "test_counts"):
            if a[key] != b[key]:
                raise AssertionError(f"Fold {a['fold']} {key} differs from the locked protocol")


def run_fold(arch, manifest, arrays, fold_row, seed, out_dir, smoke_batches):
    fold = int(fold_row["fold"])
    builder, override = ARCHS[arch]
    cfg = {**CONFIG, **override}
    X, y = arrays["X"], arrays["y"]
    train_idx = controlled_train_indices(fold_row["train_indices"], y,
                                         manifest["official_ds1_model_train_n_target"], seed, fold)
    val_idx, test_idx = fold_row["val_indices"], fold_row["test_indices"]
    (Xtr, Xva, Xte), norm = normalize_from_train(X[train_idx], X[val_idx], X[test_idx])
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BUILDERS[builder](cfg).to(device)
    cw = class_weights(ytr).to(device) if cfg["use_class_weights"] else None
    criterion = RLSTMLoss(lambda_smooth=cfg["lambda_smooth"], class_weights=cw,
                          use_focal=cfg["use_focal_loss"], focal_gamma=cfg["focal_gamma"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        generator=generator)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / f"fold{fold}_seed{seed}_state.pt"
    start, best_f1, best_epoch, bad, best_state, history = 1, -1.0, 0, 0, None, []
    if state_path.exists() and not smoke_batches:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best_f1, best_epoch = st["epoch"] + 1, st["best_f1"], st["best_epoch"]
        bad, best_state, history = st["bad"], st["best_state"], st["history"]
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["np_rng"])
        generator.set_state(st["loader_rng"])
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"[resume] {arch} fold {fold} seed {seed} from epoch {start}", flush=True)

    epochs = 1 if smoke_batches else cfg["epochs"]
    for epoch in range(start, epochs + 1):
        if bad >= cfg["early_stopping_patience"]:
            break
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        batches = [b for _, b in zip(range(smoke_batches), loader)] if smoke_batches else loader
        ts = train_one_epoch(model, batches, opt, criterion, device, cfg["grad_clip"],
                             adv_training=cfg["adversarial_training"],
                             adv_epsilon=cfg["adv_epsilon"], adv_ratio=cfg["adv_ratio"])
        val_f1 = float(f1_score(yva, predict_logits(model, Xva, device).argmax(1),
                                labels=[0, 1, 2, 3], average="macro", zero_division=0))
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = clone_state_dict(model)
        else:
            bad += 1
        history.append({"epoch": epoch, "lr": lr, "train_loss": ts["loss"],
                        "val_f1_macro": val_f1, "sec": time.time() - t0})
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad,
                    "best_state": best_state, "history": history,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                    "loader_rng": generator.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   state_path)
        print(f"{arch} f{fold} s{seed} ep{epoch:2d} lr={lr:.5f} loss={ts['loss']:.4f} "
              f"val_f1={val_f1:.4f} best={best_f1:.4f}@{best_epoch} bad={bad} "
              f"{time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    val_logits, test_logits = predict_logits(model, Xva, device), predict_logits(model, Xte, device)
    np.savez_compressed(out_dir / f"fold{fold}_seed{seed}_preds.npz",
                        val_logits=val_logits, val_y=yva, val_record=arrays["record"][val_idx],
                        logits=test_logits, y=yte, record=arrays["record"][test_idx],
                        subject_group=arrays["beat_group"][test_idx])
    torch.save({"model_state": best_state, "arch": arch, "fold": fold, "seed": seed,
                "config": cfg, "protocol_id": PROTOCOL_ID}, out_dir / f"fold{fold}_seed{seed}_model.pt")
    return {"protocol_id": PROTOCOL_ID, "arch": arch, "fold": fold, "seed": seed,
            "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
            "epochs_run": history[-1]["epoch"], "normalization": norm,
            "train_counts_after_n_control": np.bincount(ytr, minlength=5).astype(int).tolist(),
            "val_metrics": metrics(yva, val_logits), "test_metrics": metrics(yte, test_logits),
            "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke-batches", type=int, default=None)
    args = ap.parse_args()
    manifest, arrays = build_protocol(5, 0, 0.15)
    check_protocol(manifest)
    out_dir = Path(args.out_dir or ROOT / "results" / "sgcv_arch" / args.arch)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for fold in args.folds:
            rp = out_dir / f"fold{fold}_seed{seed}.json"
            if rp.exists() and not args.smoke_batches:
                print(f"[skip] {rp}", flush=True)
                continue
            res = run_fold(args.arch, manifest, arrays, arrays["folds"][fold], seed, out_dir,
                           args.smoke_batches)
            rp.write_text(json.dumps(res, indent=2))
            t = res["test_metrics"]
            print(f"[done] {args.arch} f{fold} s{seed} val={res['best_val_f1_macro']:.4f} "
                  f"test={t['f1_macro']:.4f} "
                  + " ".join(f"{c}={t['f1_per_class'][c]:.3f}" for c in "NSVF"), flush=True)


if __name__ == "__main__":
    main()
