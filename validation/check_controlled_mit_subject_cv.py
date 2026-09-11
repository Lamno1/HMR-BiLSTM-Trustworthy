"""Integrity and metric audit for C1 MIT-only subject-grouped CV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score


CLASSES = ("N", "S", "V", "F", "Q")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/controlled_mit_subject_cv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.results

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    rows = []
    errors = []

    for seed in (42, 123):
        for fold in range(5):
            stem = f"fold{fold}_seed{seed}_flat"
            paths = {
                "json": root / f"{stem}.json",
                "npz": root / f"{stem}_preds.npz",
            }
            model_path = root / f"{stem}_model.pt"
            for path in (*paths.values(), model_path):
                if not path.exists() or path.stat().st_size == 0:
                    errors.append(f"missing/empty: {path}")
            if errors and any(stem in item for item in errors):
                continue

            run = json.loads(paths["json"].read_text(encoding="utf-8"))
            with np.load(paths["npz"], allow_pickle=False) as data:
                keys = set(data.files)
                y_key = next((k for k in ("y_true", "labels", "y") if k in keys), None)
                p_key = next((k for k in ("probs", "probabilities", "y_prob", "logits") if k in keys), None)
                pred_key = next((k for k in ("y_pred", "preds", "predictions") if k in keys), None)
                if y_key is None or (p_key is None and pred_key is None):
                    errors.append(f"unexpected NPZ keys for {stem}: {sorted(keys)}")
                    continue
                y = np.asarray(data[y_key]).astype(int)
                if p_key is not None:
                    probs = np.asarray(data[p_key])
                    if p_key == "logits":
                        shifted = probs - probs.max(axis=1, keepdims=True)
                        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                    pred = probs.argmax(axis=1)
                else:
                    probs = None
                    pred = np.asarray(data[pred_key]).astype(int)

            report = classification_report(
                y, pred, labels=list(range(5)), target_names=CLASSES,
                output_dict=True, zero_division=0,
            )
            row = {
                "fold": fold,
                "seed": seed,
                "n_test": int(len(y)),
                "f1_N": report["N"]["f1-score"],
                "f1_S": report["S"]["f1-score"],
                "f1_V": report["V"]["f1-score"],
                "f1_F": report["F"]["f1-score"],
                "macro_SVF": f1_score(y, pred, labels=[1, 2, 3], average="macro", zero_division=0),
                "best_epoch": run.get("best_epoch"),
                "npz_sha256": sha256(paths["npz"]),
            }
            stored = run.get("test_metrics", {})
            stored_pc = stored.get("f1_per_class", {})
            for name in CLASSES[:4]:
                if name in stored_pc and not np.isclose(row[f"f1_{name}"], stored_pc[name], atol=1e-12):
                    errors.append(f"stored/recomputed F1 mismatch: {stem} class {name}")
            if probs is not None:
                for idx, name in ((1, "S"), (3, "F")):
                    binary = (y == idx).astype(int)
                    row[f"auc_{name}"] = (
                        roc_auc_score(binary, probs[:, idx]) if len(np.unique(binary)) == 2 else None
                    )
            rows.append(row)

    if summary.get("protocol_id") != "C1_MIT_SUBJECT_CV_V1":
        errors.append(f"unexpected protocol_id: {summary.get('protocol_id')}")
    if summary.get("n_runs") != 10:
        errors.append(f"summary n_runs={summary.get('n_runs')}, expected 10")

    per_fold = []
    for fold in range(5):
        fold_rows = [row for row in rows if row["fold"] == fold]
        per_fold.append({
            "fold": fold,
            "seeds": [row["seed"] for row in fold_rows],
            "n_test": fold_rows[0]["n_test"] if fold_rows else None,
            **{
                f"mean_{metric}": float(np.mean([row[metric] for row in fold_rows]))
                for metric in ("f1_S", "f1_F", "auc_S", "auc_F") if fold_rows
            },
        })

    fold_first = {
        metric: float(np.mean([row[f"mean_{metric}"] for row in per_fold]))
        for metric in ("f1_S", "f1_F", "auc_S", "auc_F")
    } if len(per_fold) == 5 else {}

    conventional_path = Path("results/logs/baseline_results.json")
    conventional = None
    if conventional_path.exists():
        baseline = json.loads(conventional_path.read_text(encoding="utf-8")).get("hmr_bilstm", {})
        conventional = {
            "artifact": str(conventional_path),
            "f1_S": baseline.get("f1_S"),
            "f1_F": baseline.get("f1_F"),
            "auc_S": baseline.get("auc_S"),
            "auc_F": baseline.get("auc_F"),
        }

    result = {
        "status": "PASS" if not errors and len(rows) == 10 else "FAIL",
        "protocol_id": summary.get("protocol_id"),
        "summary_n_runs": summary.get("n_runs"),
        "manifest_protocol_id": manifest.get("protocol_id"),
        "errors": errors,
        "fold_first_mean": fold_first,
        "per_fold_two_seed_mean": per_fold,
        "conventional_ds1_ds2_reference": conventional,
        "runs": rows,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
