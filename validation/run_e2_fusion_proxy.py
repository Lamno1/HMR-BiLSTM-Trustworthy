"""E2 CPU-only transparent rule-vs-label proxy for MIT-BIH fusion beats.

This is a construct-validity diagnostic, not an inter-annotator ceiling. It uses
only the primary MLII beat window already stored in the LOPO pool; no model is
loaded or trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forensic"


def corr_rows(a: np.ndarray, template: np.ndarray) -> np.ndarray:
    aa = a - a.mean(1, keepdims=True)
    tt = template - template.mean()
    return (aa @ tt) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(tt) + 1e-12)


def features(x: np.ndarray, n_template: np.ndarray) -> dict[str, np.ndarray]:
    # Windows are 187 samples at 125 Hz with the annotated beat at index 90.
    baseline = np.median(x[:, :45], axis=1, keepdims=True)
    w = x - baseline
    amp = np.max(np.abs(w[:, 75:106]), axis=1) + 1e-8
    d = np.abs(np.diff(w, axis=1))
    qd = d[:, 72:108]
    peak = qd.max(axis=1)
    width = np.empty(len(x))
    for i, row in enumerate(qd):
        ix = np.flatnonzero(row >= .15 * peak[i])
        width[i] = ((ix[-1] - ix[0] + 1) / 125.0) if len(ix) else 0.0
    pseg = w[:, 50:80] / amp[:, None]  # approximately -320 to -80 ms
    nt = (n_template - np.median(n_template[:45]))
    nt = nt[50:80] / (np.max(np.abs(nt[75:106])) + 1e-8)
    return {
        "qrs_width_s": width,
        "p_corr_to_record_N": corr_rows(pseg, nt),
        "p_amp_over_qrs": np.max(np.abs(pseg - np.median(pseg, axis=1, keepdims=True)), axis=1),
        "early_upstroke_over_qrs": np.max(d[:, 82:96], axis=1) / amp,
    }


def rule(feat: dict[str, np.ndarray], y: np.ndarray, p_corr: float, up_frac: float, p_amp_q: float) -> np.ndarray:
    n, v = y == 0, y == 2
    width_n, width_v = np.median(feat["qrs_width_s"][n]), np.median(feat["qrs_width_s"][v])
    lo, hi = sorted((width_n, width_v))
    between_width = (feat["qrs_width_s"] >= lo) & (feat["qrs_width_s"] <= hi)
    p_amp_floor = np.quantile(feat["p_amp_over_qrs"][n], p_amp_q)
    p_present = (feat["p_corr_to_record_N"] >= p_corr) & (feat["p_amp_over_qrs"] >= p_amp_floor)
    un, uv = np.median(feat["early_upstroke_over_qrs"][n]), np.median(feat["early_upstroke_over_qrs"][v])
    # At least `up_frac` of the signed N->V displacement, so polarity/direction is record-specific.
    toward_v = ((feat["early_upstroke_over_qrs"] - un) * np.sign(uv - un)
                >= up_frac * abs(uv - un))
    return between_width & p_present & toward_v


def metrics(y_true: np.ndarray, y_rule: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_rule, labels=[0, 1])
    return {
        "n": int(len(y_true)), "prevalence_F": float(y_true.mean()),
        "n_rule_F": int(y_rule.sum()), "precision": float(precision_score(y_true, y_rule, zero_division=0)),
        "recall": float(recall_score(y_true, y_rule, zero_division=0)),
        "f1": float(f1_score(y_true, y_rule, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_rule)),
        "confusion_TN_FP_FN_TP": [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = np.load(ROOT / "data" / "processed" / "lopo" / "pool.npz")
    meta = json.loads((ROOT / "data" / "processed" / "lopo" / "records.json").read_text())["records"]
    x, y, rec, source = pool["X"][:, :, 0].astype(float), pool["y"].astype(int), pool["record"].astype(int), pool["source"].astype(int)
    f_records = sorted(set(rec[(source == 0) & (y == 3)].tolist()))
    rows, all_true, all_rule = [], [], []
    sensitivity = []
    for rid in f_records:
        take = (rec == rid) & np.isin(y, [0, 2, 3])
        xr, yr = x[take], y[take]
        if (yr == 0).sum() < 10 or (yr == 2).sum() < 5:
            rows.append({"record": meta[str(rid)]["name"], "status": "skipped", "reason": "<10 N or <5 V reference beats"})
            continue
        nt = np.median(xr[yr == 0], axis=0)
        feat = features(xr, nt)
        pred = rule(feat, yr, p_corr=.35, up_frac=.50, p_amp_q=.25)
        truth = yr == 3
        row = {"record": meta[str(rid)]["name"], "status": "evaluated", "n_N": int((yr == 0).sum()),
               "n_V": int((yr == 2).sum()), "n_F": int(truth.sum()), **metrics(truth, pred)}
        row["feature_medians"] = {k: {"N": float(np.median(v[yr == 0])), "V": float(np.median(v[yr == 2])), "F": float(np.median(v[yr == 3]))} for k, v in feat.items()}
        rows.append(row); all_true.append(truth); all_rule.append(pred)

        for pc in (.20, .35, .50):
            for uf in (.25, .50, .75):
                pr = rule(feat, yr, p_corr=pc, up_frac=uf, p_amp_q=.25)
                sensitivity.append((pc, uf, truth, pr))
    yt, yp = np.concatenate(all_true), np.concatenate(all_rule)
    sens_rows = []
    for pc in (.20, .35, .50):
        for uf in (.25, .50, .75):
            ys = np.concatenate([a for p, u, a, b in sensitivity if p == pc and u == uf])
            ps = np.concatenate([b for p, u, a, b in sensitivity if p == pc and u == uf])
            sens_rows.append({"p_corr_threshold": pc, "upstroke_fraction_N_to_V": uf, **metrics(ys, ps)})
    result = {
        "experiment_id": "E2-rule-proxy", "status": "completed",
        "scope": "MIT-BIH F-bearing records in LOPO pool; binary F vs N/V; primary MLII only",
        "warning": "Rule-vs-consensus agreement is NOT an inter-annotator or Bayes ceiling; rule misspecification can lower agreement.",
        "main_rule": {
            "qrs": "15%-peak derivative span lies between per-record median N and V widths",
            "p_wave": "correlation to record-N pre-QRS template >=0.35 and amplitude >=N 25th percentile",
            "upstroke": "at least 50% of signed displacement from median N toward median V early-upstroke score",
        },
        "pooled": metrics(yt, yp), "per_record": rows, "sensitivity": sens_rows,
        "counts": {"F_bearing_MIT_records": len(f_records), "evaluated_records": len(all_true), "skipped_records": len(f_records)-len(all_true)},
    }
    (OUT / "E2_rule_proxy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "pooled": result["pooled"], "sensitivity_f1": [round(x["f1"], 4) for x in sens_rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
