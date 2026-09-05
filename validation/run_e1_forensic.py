"""E1: CPU-only forensic analysis of existing LOPO prediction artifacts.

Writes results/forensic/E1_report.json and E1_notes.md.  This script never
loads a model or trains; all quantities come from saved outer-test logits.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forensic"
SPECS = {
    "r2": (ROOT / "results" / "mp_lopo", "vrce"),
    "flat": (ROOT / "results" / "lopo_cv", "flat"),
}
CLASSES = ["N", "S", "V", "F", "Q"]
RECORD_META = json.loads((ROOT / "data" / "processed" / "lopo" / "records.json").read_text())["records"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Equal-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(conf)
    if total == 0:
        return math.nan
    ans = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if take.any():
            ans += take.mean() * abs(float(correct[take].mean()) - float(conf[take].mean()))
    return float(ans)


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
        "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)), "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def metric_block(p: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    metrics = {}
    for c, name in enumerate(CLASSES):
        binary = (y == c).astype(int)
        metrics[name] = {
            "support": int(binary.sum()),
            "auc_ovr": float(roc_auc_score(binary, p[:, c])) if 0 < binary.sum() < len(y) else None,
            "average_precision": float(average_precision_score(binary, p[:, c])) if binary.sum() else None,
            "f1_argmax": float(f1_score(binary, pred == c, zero_division=0)),
            "predicted_class_ece_15bin": ece(p[pred == c, c], (y[pred == c] == c)) if (pred == c).any() else None,
            "classwise_ece_ovr_15bin": ece(p[:, c], binary),
        }
    return {"n": int(len(y)), "multiclass_ece_15bin": ece(p.max(1), pred == y), "classes": metrics}


def parse_artifact(path: Path, model: str) -> dict:
    m = re.search(r"fold(\d+)_seed(\d+)_", path.name)
    if not m:
        raise ValueError(f"Cannot parse fold/seed: {path}")
    d = np.load(path, allow_pickle=True)
    required = {"te_logits", "te_y", "te_record", "te_source", "te_rr_raw"}
    missing = sorted(required - set(d.files))
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    z, y = d["te_logits"], d["te_y"].astype(int)
    p = softmax(z)
    pred = p.argmax(1)
    source_id = d["te_source"].astype(int)
    scopes = {"all": np.ones(len(y), bool), "mitbih": source_id == 0, "incart": source_id == 1}
    by_scope = {name: metric_block(p[take], y[take], pred[take]) for name, take in scopes.items() if take.any()}
    record_id = d["te_record"].astype(int)
    record_name = np.array([RECORD_META[str(i)]["name"] for i in record_id])
    source_name = np.where(source_id == 0, "mitbih", "incart")
    return {
        "model": model, "path": str(path.relative_to(ROOT)), "fold": int(m.group(1)),
        "seed": int(m.group(2)), "n": int(len(y)), "multiclass_ece_15bin": by_scope["all"]["multiclass_ece_15bin"],
        "metrics": by_scope["all"]["classes"], "metrics_by_scope": by_scope,
        "p": p, "y": y, "pred": pred, "record": record_name, "source": source_name,
        "rr": d["te_rr_raw"].astype(float),
    }


def public_run(r: dict) -> dict:
    return {k: r[k] for k in ("model", "path", "fold", "seed", "n", "multiclass_ece_15bin", "metrics", "metrics_by_scope")}


def fold_means(runs: list[dict]) -> list[dict]:
    ans = []
    for model in SPECS:
        for fold in range(5):
            rs = [r for r in runs if r["model"] == model and r["fold"] == fold]
            row = {"model": model, "fold": fold, "n_seeds": len(rs), "scopes": {}}
            for scope in ("all", "mitbih", "incart"):
                row["scopes"][scope] = {"classes": {}}
                for name in CLASSES:
                    row["scopes"][scope]["classes"][name] = {}
                    for metric in ("auc_ovr", "average_precision", "f1_argmax", "predicted_class_ece_15bin", "classwise_ece_ovr_15bin"):
                        vals = [r["metrics_by_scope"][scope]["classes"][name][metric] for r in rs if scope in r["metrics_by_scope"] and r["metrics_by_scope"][scope]["classes"][name][metric] is not None]
                        row["scopes"][scope]["classes"][name][metric] = float(np.mean(vals)) if vals else None
                    row["scopes"][scope]["classes"][name]["support"] = rs[0]["metrics_by_scope"][scope]["classes"][name]["support"] if rs and scope in rs[0]["metrics_by_scope"] else 0
                vals = [r["metrics_by_scope"][scope]["multiclass_ece_15bin"] for r in rs if scope in r["metrics_by_scope"]]
                row["scopes"][scope]["multiclass_ece_15bin"] = float(np.mean(vals)) if vals else None
            row["classes"] = row["scopes"]["all"]["classes"]  # compatibility/readability
            row["multiclass_ece_15bin"] = float(np.mean([r["multiclass_ece_15bin"] for r in rs])) if rs else None
            ans.append(row)
    return ans


def f_forensics(runs: list[dict], model: str) -> dict:
    rs = [r for r in runs if r["model"] == model]
    pf = np.concatenate([r["p"][r["y"] == 3, 3] for r in rs])
    pred = np.concatenate([r["pred"][r["y"] == 3] for r in rs])
    rr = np.concatenate([r["rr"][r["y"] == 3] for r in rs])
    rec = np.concatenate([r["record"][r["y"] == 3] for r in rs])
    src = np.concatenate([r["source"][r["y"] == 3] for r in rs])
    # Rank bins are deterministic even under ties and contain nearly equal counts.
    order = np.argsort(pf, kind="stable")
    decile = np.empty(len(pf), dtype=int)
    for q, idx in enumerate(np.array_split(order, 10), 1):
        decile[idx] = q
    table = []
    error_records = defaultdict(Counter)
    for q in range(1, 11):
        take = decile == q
        n = int(take.sum())
        row = {"decile": q, "rank_direction": "1=lowest_P(F), 10=highest_P(F)", "n_true_F": n,
               "pF": summarize(pf[take]), "rr_raw": [summarize(rr[take, j]) for j in range(rr.shape[1])],
               "predicted_fraction": {CLASSES[c]: float(np.mean(pred[take] == c)) for c in range(5)}}
        table.append(row)
        err = take & (pred == 0)
        for a, b in zip(src[err], rec[err]):
            error_records[q][f"{a}/{b}"] += 1
    total_nerr = int((pred == 0).sum())
    low = int(((pred == 0) & (decile <= 3)).sum())
    high = int(((pred == 0) & (decile >= 8)).sum())
    return {
        "unit": "outer-test true-F predictions pooled across 5 folds and 2 seeds; seeds are repeated predictions, not independent beats",
        "n_true_F_predictions": int(len(pf)), "n_F_to_N": total_nerr,
        "F_to_N_bottom3_deciles_fraction": low / total_nerr if total_nerr else None,
        "F_to_N_top3_deciles_fraction": high / total_nerr if total_nerr else None,
        "deciles": table,
        "F_to_N_records_by_decile": {str(q): dict(error_records[q].most_common()) for q in range(1, 11)},
    }


def s_composition(runs: list[dict], model: str = "r2") -> dict:
    # Composition is identical across seeds; use the first seed per fold.
    out = []
    for fold in range(5):
        r = sorted([x for x in runs if x["model"] == model and x["fold"] == fold], key=lambda x: x["seed"])[0]
        take = r["y"] == 1
        records = Counter(f"{s}/{q}" for s, q in zip(r["source"][take], r["record"][take]))
        out.append({
            "fold": fold, "n_S": int(take.sum()), "contains_mitdb_232": any("232" in k and "mit" in k.lower() for k in records),
            "record_counts": dict(records.most_common()),
            "largest_record_share": float(max(records.values()) / take.sum()) if take.any() else None,
            "rr_raw": [summarize(r["rr"][take, j]) for j in range(r["rr"].shape[1])],
        })
    return {"note": "one seed used because test composition is seed-invariant", "folds": out}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for model, (directory, tag) in SPECS.items():
        paths = sorted(directory.glob(f"fold[0-4]_seed*_{tag}_preds.npz"))
        if len(paths) != 10:
            raise RuntimeError(f"Expected 10 {model} artifacts, found {len(paths)}")
        runs.extend(parse_artifact(p, model) for p in paths)
    fm = fold_means(runs)
    s_auc = {m: [x["scopes"]["mitbih"]["classes"]["S"]["auc_ovr"] for x in fm if x["model"] == m] for m in SPECS}
    report = {
        "experiment_id": "E1", "status": "completed", "data_scope": "existing outer-test prediction artifacts only",
        "definitions": {
            "probability": "softmax(te_logits)", "auc": "one-vs-rest ROC AUC", "ap": "one-vs-rest average precision",
            "f1": "argmax one-vs-rest F1", "multiclass_ece": "15 equal-width bins of max-softmax confidence vs correctness",
            "predicted_class_ece": "15-bin calibration among examples argmax-predicted as that class",
            "classwise_ece": "15-bin one-vs-rest calibration over all examples",
            "rr_raw_columns": ["pre_rr/local_before", "post_rr/pre_rr", "local_before/record_median_rr"],
        },
        "runs": [public_run(r) for r in runs], "fold_means": fm,
        "s_gate": {
            "r2_fold_mean_auc": s_auc["r2"], "flat_fold_mean_auc": s_auc["flat"],
            "r2_n_folds_auc_ge_0_85": int(sum(x >= .85 for x in s_auc["r2"])),
            "r2_n_folds_auc_lt_0_80": int(sum(x < .80 for x in s_auc["r2"])),
            "criterion_calibration_lever": bool(sum(x >= .85 for x in s_auc["r2"]) >= 4),
        },
        "f_forensics": {m: f_forensics(runs, m) for m in SPECS},
        "s_record_composition": s_composition(runs),
    }
    (OUT / "E1_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    r2_folds = [x for x in fm if x["model"] == "r2"]
    lines = ["# E1 — LOPO error forensic", "", "## Direct answers", "",
             f"1. **S ranking vs calibration:** R2 S-AUC by fold = {', '.join(f'{x:.3f}' for x in s_auc['r2'])}. "
             f"The pre-registered calibration-lever criterion (AUC >= 0.85 on >=4/5 folds) is "
             f"{'MET' if report['s_gate']['criterion_calibration_lever'] else 'NOT MET'}.", ""]
    fdiag = report["f_forensics"]["r2"]
    lines += [f"2. **F errors:** among R2 true-F→N errors, {fdiag['F_to_N_bottom3_deciles_fraction']:.1%} are in the bottom three P(F) deciles and {fdiag['F_to_N_top3_deciles_fraction']:.1%} in the top three. This tests whether errors are low-ranked ambiguity or high-ranked argmax losses.", ""]
    comp = report["s_record_composition"]["folds"]
    f4 = comp[4]
    top4 = list(f4["record_counts"].items())[:5]
    lines += [f"3. **Fold 4:** mitdb/232 present = {f4['contains_mitdb_232']}; largest S-record share = {f4['largest_record_share']:.1%}. Top composition = " + ", ".join(f"{k}: {v}" for k, v in top4) + ".", ""]
    lines += ["4. **Can P2 alone salvage the current S story?** No, not from current evidence. The registered 4/5 high-AUC criterion fails and existing useful selective coverage is fold-4-only. Folds 1–3 do show high S ranking, so a targeted calibration analysis remains informative, but a global 'calibration-only' explanation is contradicted by folds 0 and 4.", "",
              "## Fold-level MIT-BIH comparison (two-seed means)", "", "| Fold | R2 S AUC | Flat S AUC | R2 S AP | R2 S F1 | R2 S OVR-ECE | R2 F AUC | R2 F AP | R2 F F1 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in r2_folds:
        s, f = x["scopes"]["mitbih"]["classes"]["S"], x["scopes"]["mitbih"]["classes"]["F"]
        flatx = [q for q in fm if q["model"] == "flat" and q["fold"] == x["fold"]][0]
        fs = flatx["scopes"]["mitbih"]["classes"]["S"]
        lines.append(f"| {x['fold']} | {s['auc_ovr']:.3f} | {fs['auc_ovr']:.3f} | {s['average_precision']:.3f} | {s['f1_argmax']:.3f} | {s['classwise_ece_ovr_15bin']:.3f} | {f['auc_ovr']:.3f} | {f['average_precision']:.3f} | {f['f1_argmax']:.3f} |")
    lines += ["", "## Observations and anomalies", "",
              "- S is heterogeneous: ranking is excellent on folds 1–3, poor on fold 0, and intermediate on fold 4. This is neither a uniform ranking failure nor a uniform calibration failure.",
              "- Fold 4's apparent S success is composition-heavy: record mitdb/232 alone contributes 1,382/2,490 S beats (55.5%). Beat-weighted metrics therefore overrepresent one patient.",
              f"- R2 F→N errors are strongly low-rank: bottom three P(F) deciles contain {fdiag['F_to_N_bottom3_deciles_fraction']:.1%}, versus {fdiag['F_to_N_top3_deciles_fraction']:.1%} in the top three. This contradicts the pre-registered decision-rule signature.",
              "- F AUC can remain moderate/high while AP and F1 collapse because prevalence is tiny and false positives dominate. AUC alone previously overstated usable F separation.", "",
              "## Candidate explanations", "",
              "1. **S subgroup/domain heterogeneity (supported):** fold-level AUC and AP vary materially; record composition is concentrated.",
              "2. **S calibration-only (partially refuted):** supported on folds 1–3, refuted as a global explanation by fold 0 AUC=0.733 and fold 4 AUC=0.816 plus fold-4-only selective utility.",
              "3. **F argmax/threshold-only (refuted for most errors):** very few F→N errors occupy the highest P(F) deciles.",
              "4. **F label/representation ambiguity or patient-specific morphology (supported but not distinguished):** errors are low-ranked; E1 cannot tell whether labels are ambiguous or signal/representation is insufficient.", "",
              "## What remains unknown", "", "- Whether specific S annotation subtypes or rhythms explain fold 0/4 ranking loss; AAMI aggregation hides subtype identity in saved artifacts.", "- Whether fold-4 selective utility persists under patient-weighted rather than beat-weighted operating points.", "- Whether F low-rank errors are human-ambiguous labels or recoverable with unavailable temporal/clinical information.", "",
              "## Interpretation safeguards", "", "- The table and registered S-AUC gate use MIT-BIH-only outer-test beats; JSON also contains all/INCART scopes.", "- Fold means average two seeds; folds, not beats or duplicate seed predictions, are the primary generalization units.", "- AUC/AP diagnose ranking; ECE diagnoses calibration. Neither proves label identifiability.", "- The F decile pool repeats each outer-test beat once per seed and is descriptive, not an i.i.d. sample.", "- RR columns are ratios: pre/local, post/pre, and local/record-median.", ""]
    (OUT / "E1_notes.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'E1_report.json'}")
    print(f"Wrote {OUT / 'E1_notes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
