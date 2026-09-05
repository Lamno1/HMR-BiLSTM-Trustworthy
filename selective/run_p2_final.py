"""Final CPU-only selective analysis over frozen LOPO prediction artifacts.

This intentionally does not import or modify ``selective.run_p2``.  It preserves
the first-cut policy (an S-calibrated conformal threshold applied to all class
probabilities), expands it over a pre-registered alpha grid, and forces every
singleton-F decision to referral.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from selective.run_p2 import CLASSES, F, S, conformal_quantile, fit_class_temperature, meta_of, softmax

ALPHAS = np.round(np.arange(0.005, 0.5001, 0.005), 3)
FIXED = {0.01, 0.05, 0.10, 0.20, 0.30}


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def source_name(x) -> str:
    s = str(x).lower()
    # Frozen LOPO artifacts use the preprocessing registry: 0=MIT-BIH, 1=INCART.
    if s in {"0", "0.0"}:
        return "mitbih"
    if s in {"1", "1.0"}:
        return "incart"
    return "mitbih" if "mit" in s else ("incart" if "incart" in s else s)


def artifact_rows(root: Path, tag: str):
    paths = sorted(root.glob(f"fold*_seed*_{tag}_preds.npz"))
    if len(paths) != 10:
        raise RuntimeError(f"expected 10 {tag} artifacts in {root}, found {len(paths)}")
    for path in paths:
        d = np.load(path, allow_pickle=True)
        required = {"va_logits", "va_y", "conformal_cal_logits", "conformal_cal_y",
                    "te_logits", "te_y", "te_record", "te_source"}
        missing = required.difference(d.files)
        if missing:
            raise RuntimeError(f"{path}: missing {sorted(missing)}")
        meta = meta_of({k: d[k] for k in d.files})
        yield path, d, int(meta["fold"]), int(meta["seed"])


def counts(y, pred, auto, set_size, in_s):
    n = len(y)
    true_s = y == S
    auto_s = auto & (pred == S)
    true_f = y == F
    accepted_n = int(auto.sum())
    correct_n = int((auto & (pred == y)).sum())
    auto_s_n = int(auto_s.sum())
    auto_s_tp = int((auto_s & true_s).sum())
    return {
        "n": n, "accepted_n": accepted_n, "correct_n": correct_n,
        "true_s_n": int(true_s.sum()), "in_set_s_true_n": int((in_s & true_s).sum()),
        "auto_s_n": auto_s_n, "auto_s_tp": auto_s_tp,
        "true_f_n": int(true_f.sum()), "f_auto_leak_n": int((auto & true_f).sum()),
        "set_empty_n": int((set_size == 0).sum()),
        "set_singleton_n": int((set_size == 1).sum()),
        "set_ambiguous_n": int((set_size > 1).sum()),
    }


def metrics(c):
    n, accepted = c["n"], c["accepted_n"]
    true_s, auto_s = c["true_s_n"], c["auto_s_n"]
    true_f = c["true_f_n"]
    return {
        "overall_coverage": finite(accepted / n if n else None),
        "overall_selective_risk": finite(1 - c["correct_n"] / accepted if accepted else None),
        "referral_fraction": finite(1 - accepted / n if n else None),
        "auto_S_coverage": finite(c["auto_s_tp"] / true_s if true_s else None),
        "auto_S_precision": finite(c["auto_s_tp"] / auto_s if auto_s else None),
        "realised_S_coverage": finite(c["in_set_s_true_n"] / true_s if true_s else None),
        "F_referral_fraction": finite(1 - c["f_auto_leak_n"] / true_f if true_f else None),
        "F_to_auto_leakage": finite(c["f_auto_leak_n"] / true_f if true_f else None),
        "empty_rate": finite(c["set_empty_n"] / n if n else None),
        "singleton_rate": finite(c["set_singleton_n"] / n if n else None),
        "ambiguous_rate": finite(c["set_ambiguous_n"] / n if n else None),
        "n_auto": accepted, "n_auto_S": auto_s, "n_true_S": true_s, "n_true_F": true_f,
    }


def full_coverage(y, pred):
    true_s, pred_s = y == S, pred == S
    tp = int((true_s & pred_s).sum())
    return {
        "full_coverage_error": finite(np.mean(y != pred)),
        "full_S_precision": finite(tp / pred_s.sum() if pred_s.any() else None),
        "full_S_recall": finite(tp / true_s.sum() if true_s.any() else None),
    }


def binary_ece(prob, truth, bins=15):
    edges = np.linspace(0, 1, bins + 1); total = len(prob); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            value += mask.mean() * abs(float(prob[mask].mean()) - float(truth[mask].mean()))
    return value if total else None


def sum_counts(rows):
    keys = ["n", "accepted_n", "correct_n", "true_s_n", "in_set_s_true_n",
            "auto_s_n", "auto_s_tp", "true_f_n", "f_auto_leak_n",
            "set_empty_n", "set_singleton_n", "set_ambiguous_n"]
    return {k: sum(int(r[k]) for r in rows) for k in keys}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def bootstrap_patient(patient_rows, reps, seed):
    """Macro patient bootstrap; model seeds remain nested within each record."""
    rng = np.random.default_rng(seed)
    vals = np.array([[r.get(k, np.nan) if r.get(k) is not None else np.nan
                      for k in ("overall_coverage", "overall_selective_risk", "auto_S_coverage",
                                "auto_S_precision", "referral_fraction", "F_referral_fraction")]
                     for r in patient_rows], dtype=float)
    if not len(vals):
        return {}
    sims = np.empty((reps, vals.shape[1]))
    for i in range(reps):
        sample = vals[rng.integers(0, len(vals), len(vals))]
        for j in range(vals.shape[1]):
            good = sample[:, j][np.isfinite(sample[:, j])]
            sims[i, j] = np.mean(good) if len(good) else np.nan
    out = {}
    for j, key in enumerate(("overall_coverage", "overall_selective_risk", "auto_S_coverage",
                             "auto_S_precision", "referral_fraction", "F_referral_fraction")):
        good = sims[:, j][np.isfinite(sims[:, j])]
        observed = vals[:, j][np.isfinite(vals[:, j])]
        out[key] = {"mean": finite(np.mean(observed)) if len(observed) else None,
                    "ci95": ([finite(np.quantile(good, .025)), finite(np.quantile(good, .975))]
                             if len(good) else [None, None])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-dir", type=Path, required=True)
    ap.add_argument("--flat-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-seed", type=int, default=20260905)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve, patients, full_rows, calibration_rows, manifest_inputs = [], [], [], [], []
    for model, root, tag in (("r2", args.r2_dir, "vrce"), ("flat", args.flat_dir, "flat")):
        for path, d, fold, seed in artifact_rows(root, tag):
            manifest_inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            T = fit_class_temperature(d["va_logits"].astype(float), d["va_y"].astype(int))
            p = softmax(d["te_logits"].astype(float) / T)
            pcal = softmax(d["conformal_cal_logits"].astype(float) / T)
            yall = d["te_y"].astype(int); recall = d["te_record"].astype(str)
            srcall = np.array([source_name(x) for x in d["te_source"]])
            nonconf = 1 - pcal[d["conformal_cal_y"].astype(int) == S, S]
            for dataset in ("mitbih", "incart"):
                dm = srcall == dataset
                if not dm.any(): continue
                y, pp, records = yall[dm], p[dm], recall[dm]
                pred = pp.argmax(1)
                base = {"dataset": dataset, "model": model, "fold": fold, "seed": seed}
                full_rows.append({**base, **full_coverage(y, pred), "n": len(y)})
                rawp = softmax(d["te_logits"].astype(float))[dm]
                calibration_rows.append({**base,
                    "S_ECE_before": binary_ece(rawp[:, S], y == S),
                    "S_ECE_after": binary_ece(pp[:, S], y == S)})
                for alpha in ALPHAS:
                    q = conformal_quantile(nonconf, float(alpha)); threshold = 1 - q
                    inset = pp >= threshold; sizes = inset.sum(1)
                    singleton_class = np.where(sizes == 1, inset.argmax(1), -1)
                    auto = (sizes == 1) & (singleton_class != F)
                    c = counts(y, singleton_class, auto, sizes, inset[:, S])
                    row = {**base, "alpha": float(alpha), "nominal_S_coverage": 1-float(alpha),
                           "threshold": float(threshold), **c, **metrics(c), **full_coverage(y, pred)}
                    curve.append(row)
                    if float(alpha) in FIXED:
                        for record in sorted(np.unique(records)):
                            rm = records == record
                            rc = counts(y[rm], singleton_class[rm], auto[rm], sizes[rm], inset[rm, S])
                            patients.append({**base, "alpha": float(alpha), "record": record,
                                             **rc, **metrics(rc), **full_coverage(y[rm], pred[rm])})

    # Fold summaries pool counts only across the two model runs, then expose seed spread separately.
    fold_rows = []
    for key in sorted({(r["dataset"],r["model"],r["fold"],r["alpha"]) for r in curve}):
        ds, model, fold, alpha = key
        rr = [r for r in curve if (r["dataset"],r["model"],r["fold"],r["alpha"]) == key]
        c = sum_counts(rr); m = metrics(c)
        risk_vals = [r["overall_selective_risk"] for r in rr if r["overall_selective_risk"] is not None]
        fold_rows.append({"dataset":ds,"model":model,"fold":fold,"seed":"seed-pooled-counts",
                          "alpha":alpha,"nominal_S_coverage":1-alpha,**m,
                          "risk_seed_min":min(risk_vals) if risk_vals else None,
                          "risk_seed_max":max(risk_vals) if risk_vals else None})

    # Locked GO: highest S coverage among pre-registered grid points satisfying risk <= .02.
    go_rows = []
    for model in ("r2", "flat"):
        for fold in range(5):
            eligible = [r for r in fold_rows if r["dataset"]=="mitbih" and r["model"]==model and
                        r["fold"]==fold and r["overall_selective_risk"] is not None and
                        r["overall_selective_risk"] <= .02]
            best = max(eligible, key=lambda r: (r["auto_S_coverage"] or -1, r["overall_coverage"] or -1)) if eligible else None
            reliable = fold in {1,2,3}
            qualifies = bool(best and reliable and (best["auto_S_coverage"] or 0)>=.15 and
                             (best["auto_S_precision"] or 0)>=.90)
            go_rows.append({"dataset":"mitbih","model":model,"fold":fold,"seed":"seed-pooled-counts",
                            "alpha":best["alpha"] if best else None,
                            "overall_selective_risk":best["overall_selective_risk"] if best else None,
                            "auto_S_coverage":best["auto_S_coverage"] if best else None,
                            "auto_S_precision":best["auto_S_precision"] if best else None,
                            "referral_fraction":best["referral_fraction"] if best else None,
                            "ranking_reliable":reliable,"qualifies":qualifies})
    n_go = sum(r["qualifies"] for r in go_rows if r["model"]=="r2")

    # Structural check: any S singleton at any alpha, plus fixed alpha details, folds 1-3.
    structural = []
    for model in ("r2","flat"):
        for fold in (1,2,3):
            rr = [r for r in curve if r["dataset"]=="mitbih" and r["model"]==model and r["fold"]==fold]
            cr = [r for r in calibration_rows if r["dataset"]=="mitbih" and r["model"]==model and r["fold"]==fold]
            before = float(np.mean([r["S_ECE_before"] for r in cr])); after = float(np.mean([r["S_ECE_after"] for r in cr]))
            structural.append({"dataset":"mitbih","model":model,"fold":fold,"seed":"both",
                               "alpha":"all-grid","any_S_singleton":any(r["n_auto_S"]>0 for r in rr),
                               "max_auto_S_coverage":max((r["auto_S_coverage"] or 0) for r in rr),
                               "n_grid_points_zero_auto_S":sum(r["n_auto_S"]==0 for r in rr),
                               "n_grid_points":len(rr),"S_ECE_before":before,"S_ECE_after":after,
                               "S_ECE_reduced":after < before})

    uncertainty = {}
    for key in sorted({(r["dataset"],r["model"],r["fold"],r["alpha"]) for r in patients}):
        rr0 = [r for r in patients if (r["dataset"],r["model"],r["fold"],r["alpha"])==key]
        # Average model-seed metrics within each patient before patient resampling.
        byrec=[]
        for record in sorted({r["record"] for r in rr0}):
            rs=[r for r in rr0 if r["record"]==record]; pr={"record":record}
            for k in ("overall_coverage","overall_selective_risk","auto_S_coverage","auto_S_precision","referral_fraction","F_referral_fraction"):
                vv=[r[k] for r in rs if r[k] is not None]; pr[k]=float(np.mean(vv)) if vv else None
            byrec.append(pr)
        uncertainty["|".join(map(str,key))] = bootstrap_patient(byrec,args.bootstrap_replicates,args.bootstrap_seed)

    write_csv(args.out_dir/"continuous_s_risk_coverage.csv", curve)
    write_csv(args.out_dir/"fixed_operating_points.csv", [r for r in curve if r["alpha"] in FIXED])
    write_csv(args.out_dir/"prediction_set_sizes.csv", [{k:v for k,v in r.items() if k in
              {"dataset","model","fold","seed","alpha","empty_rate","singleton_rate","ambiguous_rate","n_auto_S"}}
              for r in curve])
    write_csv(args.out_dir/"patient_level_metrics.csv", patients)
    write_csv(args.out_dir/"fold_level_metrics.csv", fold_rows)
    write_csv(args.out_dir/"f_referral_diagnostics.csv", [{k:v for k,v in r.items() if k in
              {"dataset","model","fold","seed","alpha","F_referral_fraction","F_to_auto_leakage","n_true_F"}}
              for r in curve])
    write_csv(args.out_dir/"go_criterion.csv", go_rows)
    summary={"locked_verdict":"GO" if n_go>=2 else "NO-GO","n_qualifying_reliable_mit_folds":n_go,
             "go_criterion":go_rows,"structural_failure_check":structural}
    (args.out_dir/"continuous_s_risk_coverage.json").write_text(json.dumps(curve,indent=2,allow_nan=False),encoding="utf-8")
    fixed_rows=[r for r in curve if r["alpha"] in FIXED]
    (args.out_dir/"fixed_operating_points.json").write_text(json.dumps(fixed_rows,indent=2,allow_nan=False),encoding="utf-8")
    (args.out_dir/"fold_level_metrics.json").write_text(json.dumps(fold_rows,indent=2,allow_nan=False),encoding="utf-8")
    (args.out_dir/"f_referral_diagnostics.json").write_text(json.dumps(
        [{k:v for k,v in r.items() if k in {"dataset","model","fold","seed","alpha","F_referral_fraction","F_to_auto_leakage","n_true_F"}} for r in curve],
        indent=2,allow_nan=False),encoding="utf-8")
    (args.out_dir/"patient_cluster_uncertainty.json").write_text(json.dumps(uncertainty,indent=2,allow_nan=False),encoding="utf-8")
    (args.out_dir/"r2_vs_flat_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    manifest={"alpha_grid":[float(x) for x in ALPHAS],"fixed_alphas":sorted(FIXED),
              "bootstrap_unit":"record/patient","bootstrap_replicates":args.bootstrap_replicates,
              "bootstrap_seed":args.bootstrap_seed,"policy":"S-calibrated singleton; singleton F forced referral",
              "inputs":manifest_inputs}
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# P2 Final Report","",f"Locked verdict: **{summary['locked_verdict']}**",
           "",f"Qualifying ranking-reliable MIT-BIH folds: **{n_go}/3**","",
           "## Locked GO criterion","",
           "| model | fold | alpha | risk | S coverage | S precision | referral | qualifies |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in go_rows:
        def fmt(x): return "NA" if x is None else f"{x:.4f}" if isinstance(x,float) else str(x)
        lines.append(f"| {r['model']} | {r['fold']} | {fmt(r['alpha'])} | {fmt(r['overall_selective_risk'])} | {fmt(r['auto_S_coverage'])} | {fmt(r['auto_S_precision'])} | {fmt(r['referral_fraction'])} | {r['qualifies']} |")
    lines += ["","## Structural-failure check","",
              "S-ECE is binary classwise ECE on the outer fold; it is diagnostic, not fitted there.",""]
    for r in structural:
        lines.append(f"- {r['model']} fold {r['fold']}: any S singleton={r['any_S_singleton']}; max S coverage={r['max_auto_S_coverage']:.4f}; S-ECE {r['S_ECE_before']:.4f}->{r['S_ECE_after']:.4f}.")
    (args.out_dir/"P2_FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
