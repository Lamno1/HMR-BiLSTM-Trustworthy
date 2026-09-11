"""Controlled MIT-BIH-only subject-grouped cross-validation for flat ResMemECG.

This is a new, isolated experiment. It does not reuse or overwrite the historical
record-grouped RG-CV artifacts. Controls relative to the conventional DS1/DS2
reference are: MIT-BIH only, flat RLSTMClassifier, official 45-epoch cosine/FGSM
recipe, patience=15, and an N cap equal to the official DS1 model-train N count.

The split unit is a subject group. MIT-BIH records 201 and 202 are explicitly
bound to one group. The 44 non-paced records present in the historical pool thus
form 43 subject groups. Every epoch is resumable.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmr_bilstm import RLSTMClassifier, RLSTMLoss
from train_inter_patient import CONFIG, cosine_lr, fgsm_attack_train, set_seed


POOL = Path("data/processed/lopo/pool.npz")
RECORDS = Path("data/processed/lopo/records.json")
DS1_TRAIN = Path("data/processed/splits/inter_train.npz")
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
PROTOCOL_ID = "C1_MIT_SUBJECT_CV_V1"


def subject_key(record_name: str) -> str:
    """Return the locked subject-group key for a MIT-BIH record."""
    rec = record_name.split("/")[-1]
    return "mitdb/subject_201_202" if rec in {"201", "202"} else f"mitdb/subject_{rec}"


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assign_subject_folds(group_counts: dict[int, np.ndarray], k: int, seed: int) -> dict[int, int]:
    """Snake-draft subject groups by F, then S-only, then remaining N support."""
    rng = np.random.default_rng(seed)
    gids = sorted(group_counts)
    assignment: dict[int, int] = {}
    cursor = 0

    def snake(pool: list[int], cls: int) -> None:
        nonlocal cursor
        # Random tie breaker is generated once and retained in the manifest.
        tie = {g: float(rng.random()) for g in pool}
        ordered = sorted(pool, key=lambda g: (-group_counts[g][cls], tie[g]))
        for g in ordered:
            row, col = divmod(cursor, k)
            fold = col if row % 2 == 0 else k - 1 - col
            assignment[g] = fold
            cursor += 1

    f_groups = [g for g in gids if group_counts[g][3] > 0]
    s_groups = [g for g in gids if group_counts[g][3] == 0 and group_counts[g][1] > 0]
    rest = [g for g in gids if g not in f_groups and g not in s_groups]
    snake(f_groups, 3)
    snake(s_groups, 1)
    snake(rest, 0)
    return assignment


def choose_inner_val(train_groups: list[int], group_counts: dict[int, np.ndarray],
                     fraction: float, seed: int) -> set[int]:
    """Greedy subject-level validation split with F/S coverage where possible."""
    rng = np.random.default_rng(seed)
    target = max(2, int(round(fraction * len(train_groups))))
    remaining = set(train_groups)
    selected: list[int] = []
    for cls in (3, 1, 2, 0):
        candidates = [g for g in remaining if group_counts[g][cls] > 0]
        rng.shuffle(candidates)
        candidates.sort(key=lambda g: -group_counts[g][cls])
        if candidates and len(selected) < target:
            g = candidates[0]
            selected.append(g)
            remaining.remove(g)
    tail = list(remaining)
    rng.shuffle(tail)
    selected.extend(tail[: max(0, target - len(selected))])
    return set(selected[:target])


def normalize_from_train(X_train: np.ndarray, *others: np.ndarray):
    mean = float(X_train.mean())
    std = float(np.sqrt(X_train.var()) + 1e-8)
    arrays = tuple(((x - mean) / std).astype(np.float32) for x in (X_train,) + others)
    return arrays, {"mean": mean, "std": std}


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=5).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    values = np.clip(counts.sum() / (5.0 * counts), 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(values)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        chunks.append(model(xb).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    per_class = f1_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, labels=[0, 1, 2, 3], average="weighted", zero_division=0)),
        "f1_per_class": {CLASS_NAMES[i]: float(per_class[i]) for i in range(4)},
        "confusion_counts": {},
    }
    for true_cls in range(4):
        for pred_cls in range(4):
            out["confusion_counts"][f"{CLASS_NAMES[true_cls]}->{CLASS_NAMES[pred_cls]}"] = int(
                np.sum((y == true_cls) & (pred == pred_cls)))
    aucs = {}
    for cls in range(4):
        target = (y == cls).astype(np.int8)
        aucs[CLASS_NAMES[cls]] = (float(roc_auc_score(target, probs[:, cls]))
                                  if 0 < target.sum() < len(target) else None)
    out["auc_per_class"] = aucs
    return out


def build_protocol(k: int, fold_seed: int, inner_val_fraction: float) -> tuple[dict, dict]:
    data = np.load(POOL)
    source_mask = data["source"] == 0
    X = data["X"][source_mask].astype(np.float32)
    y = data["y"][source_mask].astype(np.int64)
    record = data["record"][source_mask].astype(np.int64)

    metadata = json.load(open(RECORDS, encoding="utf-8"))["records"]
    record_name = {int(rid): row["name"] for rid, row in metadata.items() if int(row["source"]) == 0}
    missing_names = sorted(set(map(int, np.unique(record))) - set(record_name))
    if missing_names:
        raise RuntimeError(f"Missing MIT record metadata for IDs: {missing_names}")

    keys = sorted({subject_key(record_name[int(r)]) for r in np.unique(record)})
    key_to_gid = {key: idx for idx, key in enumerate(keys)}
    rid_to_gid = {rid: key_to_gid[subject_key(name)] for rid, name in record_name.items()}
    beat_group = np.array([rid_to_gid[int(r)] for r in record], dtype=np.int64)
    group_counts = {g: np.bincount(y[beat_group == g], minlength=5).astype(np.int64)
                    for g in np.unique(beat_group)}
    fold_of = assign_subject_folds(group_counts, k, fold_seed)

    ds1 = np.load(DS1_TRAIN)
    ds1_y = ds1["y"].astype(np.int64)
    official_n_target = int(np.sum(ds1_y == 0))
    folds = []
    for fold in range(k):
        test_groups = {g for g, f in fold_of.items() if f == fold}
        development_groups = [g for g in fold_of if g not in test_groups]
        val_groups = choose_inner_val(development_groups, group_counts, inner_val_fraction,
                                      fold_seed + 1000 + fold)
        train_groups = set(development_groups) - val_groups
        tr = np.where(np.isin(beat_group, list(train_groups)))[0]
        va = np.where(np.isin(beat_group, list(val_groups)))[0]
        te = np.where(np.isin(beat_group, list(test_groups)))[0]
        if set(beat_group[tr]) & set(beat_group[va]) or set(beat_group[tr]) & set(beat_group[te]) or set(beat_group[va]) & set(beat_group[te]):
            raise AssertionError("Subject-group leakage detected")
        n_available = int(np.sum(y[tr] == 0))
        folds.append({
            "fold": fold,
            "train_groups": sorted(map(int, train_groups)),
            "val_groups": sorted(map(int, val_groups)),
            "test_groups": sorted(map(int, test_groups)),
            "train_indices": tr,
            "val_indices": va,
            "test_indices": te,
            "train_counts_before_n_control": np.bincount(y[tr], minlength=5).astype(int).tolist(),
            "val_counts": np.bincount(y[va], minlength=5).astype(int).tolist(),
            "test_counts": np.bincount(y[te], minlength=5).astype(int).tolist(),
            "n_available": n_available,
            "n_target": official_n_target,
        })

    gid_to_records = defaultdict(list)
    for rid, gid in rid_to_gid.items():
        if rid in set(map(int, np.unique(record))):
            gid_to_records[gid].append(record_name[rid])
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "description": "MIT-BIH-only five-fold subject-grouped cross-validation of flat RLSTMClassifier",
        "dataset": "MIT-BIH only; source==0 from data/processed/lopo/pool.npz",
        "split_unit": "subject_group",
        "n_mit_records": int(len(np.unique(record))),
        "n_subject_groups": int(len(keys)),
        "special_binding": {"mitdb/subject_201_202": ["mitdb/201", "mitdb/202"]},
        "group_records": {str(g): sorted(v) for g, v in gid_to_records.items()},
        "k": k,
        "fold_seed": fold_seed,
        "inner_val_fraction": inner_val_fraction,
        "official_ds1_model_train_n_target": official_n_target,
        "official_ds1_model_train_counts": np.bincount(ds1_y, minlength=5).astype(int).tolist(),
        "model": "hmr_bilstm.RLSTMClassifier (flat, num_layers=2)",
        "selection": "best inner-validation macro-F1 over AAMI N/S/V/F",
        "recipe": {key: CONFIG[key] for key in [
            "epochs", "early_stopping_patience", "batch_size", "learning_rate", "min_lr",
            "weight_decay", "grad_clip", "use_focal_loss", "focal_gamma",
            "lambda_smooth", "adversarial_training", "adv_epsilon", "adv_ratio",
            "hidden_size", "dropout", "cnn_out_channels", "num_layers"]},
        "folds": [{key: value for key, value in row.items() if not key.endswith("indices")}
                  for row in folds],
    }
    arrays = {"X": X, "y": y, "record": record, "beat_group": beat_group, "folds": folds}
    return manifest, arrays


def controlled_train_indices(indices: np.ndarray, y: np.ndarray, n_target: int,
                             seed: int, fold: int) -> np.ndarray:
    n_idx = indices[y[indices] == 0]
    other_idx = indices[y[indices] != 0]
    if len(n_idx) < n_target:
        raise RuntimeError(f"Fold {fold} has only {len(n_idx)} training N beats; target={n_target}")
    rng = np.random.default_rng(seed * 1009 + fold)
    kept_n = rng.choice(n_idx, n_target, replace=False)
    combined = np.concatenate([kept_n, other_idx])
    rng.shuffle(combined)
    return combined


def train_one_fold(manifest: dict, arrays: dict, fold_row: dict, seed: int,
                   out_dir: Path, epochs: int, patience: int, smoke_batches: int | None) -> dict:
    fold = int(fold_row["fold"])
    X, y = arrays["X"], arrays["y"]
    train_idx = controlled_train_indices(fold_row["train_indices"], y,
                                         manifest["official_ds1_model_train_n_target"], seed, fold)
    val_idx, test_idx = fold_row["val_indices"], fold_row["test_indices"]
    (Xtr, Xva, Xte), norm = normalize_from_train(X[train_idx], X[val_idx], X[test_idx])
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLSTMClassifier(input_size=1, hidden_size=CONFIG["hidden_size"],
                            dropout=CONFIG["dropout"], num_classes=5,
                            cnn_out_channels=CONFIG["cnn_out_channels"], num_layers=2).to(device)
    criterion = RLSTMLoss(lambda_smooth=CONFIG["lambda_smooth"],
                          class_weights=class_weights(ytr).to(device), use_focal=True,
                          focal_gamma=CONFIG["focal_gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0,
                        pin_memory=False, generator=generator)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / f"fold{fold}_seed{seed}_state.pt"
    start_epoch, best_epoch, best_value, bad_epochs = 1, 0, -1.0, 0
    best_state, history = None, []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Refusing incompatible checkpoint: {state_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch, best_value = int(state["best_epoch"]), float(state["best_value"])
        bad_epochs, best_state = int(state["bad_epochs"]), state["best_state"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        print(f"[resume] fold={fold} seed={seed} epoch={start_epoch}", flush=True)

    run_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        lr = cosine_lr(epoch - 1, epochs, CONFIG["learning_rate"], CONFIG["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses, seen = 0.0, 0
        for batch_no, (xb, yb) in enumerate(loader):
            if smoke_batches is not None and batch_no >= smoke_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            split = int(len(xb) * (1 - CONFIG["adv_ratio"]))
            if split < len(xb):
                xa = fgsm_attack_train(model, xb[split:], yb[split:], CONFIG["adv_epsilon"], criterion)
                xb = torch.cat([xb[:split], xa])
                yb = torch.cat([yb[:split], yb[split:]])
            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss fold={fold} seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            losses += float(loss.item()) * len(xb)
            seen += len(xb)

        val_logits = predict_logits(model, Xva, device)
        val_value = float(f1_score(yva, val_logits.argmax(1), labels=[0, 1, 2, 3],
                                   average="macro", zero_division=0))
        improved = val_value > best_value
        if improved:
            best_value, best_epoch, bad_epochs = val_value, epoch, 0
            best_state = clone_state_dict(model)
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "lr": lr, "train_loss": losses / max(seen, 1),
               "train_examples_seen": seen, "val_f1_macro": val_value,
               "time_sec": time.time() - epoch_start}
        history.append(row)
        torch.save({
            "protocol_id": PROTOCOL_ID, "fold": fold, "seed": seed, "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch, "best_value": best_value, "bad_epochs": bad_epochs,
            "best_state": best_state, "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "loader_rng": generator.get_state(),
        }, state_path)
        print(f"fold={fold} seed={seed} epoch={epoch}/{epochs} loss={row['train_loss']:.4f} "
              f"val_macro={val_value:.4f} best={best_value:.4f}@{best_epoch} "
              f"bad={bad_epochs}/{patience} sec={row['time_sec']:.1f}", flush=True)
        if smoke_batches is None and bad_epochs >= patience:
            print(f"[early-stop] fold={fold} seed={seed}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid best state was produced")
    model.load_state_dict(best_state)
    test_logits = predict_logits(model, Xte, device)
    result = {
        "protocol_id": PROTOCOL_ID, "fold": fold, "seed": seed,
        "best_epoch": best_epoch, "best_val_f1_macro": best_value,
        "epochs_completed": history[-1]["epoch"], "stopped_early": history[-1]["epoch"] < epochs,
        "train_time_this_invocation_sec": time.time() - run_start,
        "train_counts_after_n_control": np.bincount(ytr, minlength=5).astype(int).tolist(),
        "val_counts": np.bincount(yva, minlength=5).astype(int).tolist(),
        "test_counts": np.bincount(yte, minlength=5).astype(int).tolist(),
        "normalization": norm, "history": history, "test_metrics": metrics(yte, test_logits),
        "classification_report": classification_report(yte, test_logits.argmax(1),
            labels=[0, 1, 2, 3, 4], target_names=CLASS_NAMES, zero_division=0, digits=4),
    }
    np.savez_compressed(out_dir / f"fold{fold}_seed{seed}_flat_preds.npz",
                        logits=test_logits, y=yte, record=arrays["record"][test_idx],
                        subject_group=arrays["beat_group"][test_idx],
                        protocol_id=np.array(PROTOCOL_ID))
    torch.save({"model_state": best_state, "protocol_id": PROTOCOL_ID, "fold": fold,
                "seed": seed, "config": {**CONFIG, "epochs": epochs,
                                           "early_stopping_patience": patience}},
               out_dir / f"fold{fold}_seed{seed}_flat_model.pt")
    return result


def summarize(out_dir: Path) -> dict:
    rows = [json.load(open(path, encoding="utf-8"))
            for path in sorted(out_dir.glob("fold*_seed*_flat.json"))]
    summary = {"protocol_id": PROTOCOL_ID, "n_runs": len(rows), "runs": rows}
    if rows:
        per_fold = []
        for fold in sorted({int(row["fold"]) for row in rows}):
            fold_rows = [row for row in rows if int(row["fold"]) == fold]
            per_fold.append({
                "fold": fold, "seeds": [int(row["seed"]) for row in fold_rows],
                "mean": {metric: float(np.mean([row["test_metrics"]["f1_per_class"][metric]
                                                  for row in fold_rows]))
                         for metric in ["N", "S", "V", "F"]},
                "seed_values": {metric: [row["test_metrics"]["f1_per_class"][metric]
                                          for row in fold_rows]
                                for metric in ["N", "S", "V", "F"]},
            })
        summary["per_fold"] = per_fold
        complete = len(rows) == 10 and all(len(row["seeds"]) == 2 for row in per_fold)
        summary["complete_5fold_2seed"] = complete
        if complete:
            summary["fold_mean"] = {metric: float(np.mean([row["mean"][metric]
                                                            for row in per_fold]))
                                    for metric in ["N", "S", "V", "F"]}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--inner-val-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=CONFIG["early_stopping_patience"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--out-dir", default="results/controlled_mit_subject_cv")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=None)
    args = parser.parse_args()
    if args.k != 5:
        raise ValueError("The controlled protocol is locked to five folds")
    if args.smoke_batches is None and (args.epochs != 45 or args.patience != 15):
        raise ValueError("Full controlled runs are locked to epochs=45 and patience=15")

    manifest, arrays = build_protocol(args.k, args.fold_seed, args.inner_val_frac)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "protocol_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    if any(row["n_available"] < row["n_target"] for row in manifest["folds"]):
        raise RuntimeError("At least one fold cannot meet the locked N target")
    if args.audit_only:
        return

    todo = args.folds if args.folds is not None else list(range(5))
    for seed in args.seeds:
        for fold in todo:
            result_path = out_dir / f"fold{fold}_seed{seed}_flat.json"
            if result_path.exists() and args.smoke_batches is None:
                print(f"[skip-complete] {result_path}", flush=True)
                continue
            fold_row = arrays["folds"][fold]
            result = train_one_fold(manifest, arrays, fold_row, seed, out_dir,
                                    args.epochs, args.patience, args.smoke_batches)
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            summarize(out_dir)
    print(json.dumps(summarize(out_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
