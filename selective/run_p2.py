"""P2 - post-hoc selective-decision analysis on frozen LOPO artifacts.

Consumes results/<dir>/fold{f}_seed{s}_{tag}_preds.npz (never retrains).

Per (fold, seed):
  1. Per-class temperature calibration, FIT ON inner-val (va) logits only,
     L2-regularised toward T=1 so the tiny F class barely moves.
  2. Mondrian split-conformal for S: nonconformity 1 - p_cal(S) on the
     conformal-cal split's true-S beats; threshold = conformal quantile at
     alpha_S. Realised S-coverage on outer-test is the reported evidence
     (LOPO breaks patient exchangeability -> nominal is a target, not a proof).
  3. F is FRONTIER-ONLY: sweep the confidence score over outer-test predicted-F
     beats and report the auto_precision_F / auto_coverage_F trade-off, per
     fold and pooled. No F threshold is selected, no guarantee claimed.

Run: python -u -m selective.run_p2   (analyses tag=vrce and tag=flat)
Outputs: results/p2/analysis_<tag>.json, results/p2/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibration_metrics import compute_classwise_ece  # noqa: E402
from selective.metrics import selective_metrics  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
S, F = 1, 3
ALPHAS = (0.10, 0.05)
OUT = Path("results/p2")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_class_temperature(logits: np.ndarray, y: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """5-vector T minimising NLL(softmax(z / T)) + lam * ||T - 1||^2.

    Coordinate descent over a small grid - robust with tiny per-class counts.
    """
    T = np.ones(5, dtype=np.float64)
    grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0])
    idx = np.arange(len(y))
    for _ in range(4):
        for c in range(5):
            best, best_obj = T[c], np.inf
            for t in grid:
                Tc = T.copy()
                Tc[c] = t
                p = softmax(logits / Tc)
                nll = -np.log(np.clip(p[idx, y], 1e-12, 1)).mean()
                obj = nll + lam * float(np.sum((Tc - 1.0) ** 2))
                if obj < best_obj:
                    best_obj, best = obj, float(t)
            T[c] = best
    return T


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Standard split-conformal quantile; +inf if alpha too small for this n."""
    n = len(scores)
    if n < 1:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def meta_of(art: dict) -> dict:
    try:
        return json.loads(str(art["meta_json"]))
    except Exception:  # noqa: BLE001
        return {}


def load_folds(result_dir: Path, tag: str) -> list[dict]:
    out = []
    for p in sorted(result_dir.glob(f"fold*_seed*_{tag}_preds.npz")):
        d = np.load(p, allow_pickle=True)
        art = {k: d[k] for k in d.files}
        art["_name"] = p.name
        out.append(art)
    return out


def analyse_tag(result_dir: Path, tag: str) -> dict:
    folds = load_folds(result_dir, tag)
    if not folds:
        return {"tag": tag, "error": f"no *_{tag}_preds.npz in {result_dir}"}

    per_fold = []
    pooled_f = []            # (score, true_is_F) over all folds, outer-test predicted-F
    n_true_F_total = 0
    for a in folds:
        va_z, va_y = a["va_logits"].astype(np.float64), a["va_y"].astype(int)
        te_z, te_y = a["te_logits"].astype(np.float64), a["te_y"].astype(int)
        ca_z, ca_y = a["conformal_cal_logits"].astype(np.float64), a["conformal_cal_y"].astype(int)

        T = fit_class_temperature(va_z, va_y)
        p_te = softmax(te_z / T)
        p_te_raw = softmax(te_z)
        p_ca = softmax(ca_z / T)
        te_pred = p_te.argmax(1)

        cw_before = compute_classwise_ece(p_te_raw, te_y)
        cw_after = compute_classwise_ece(p_te, te_y)

        # --- Mondrian-S conformal ---
        nonconf = 1.0 - p_ca[ca_y == S, S]
        s_res = {}
        for al in ALPHAS:
            q = conformal_quantile(nonconf, al)
            in_set_S = p_te[:, S] >= (1.0 - q)
            true_S = te_y == S
            realised = float(in_set_S[true_S].mean()) if true_S.any() else float("nan")
            # auto-decide S: predicted S AND only S survives its conformal set
            others_out = p_te[:, [0, 2, 3, 4]].max(1) < (1.0 - q)
            accepted = in_set_S & (te_pred == S) & others_out
            sm = selective_metrics(te_y, te_pred, accepted, classes=(S,))
            s_res[f"alpha_{al}"] = {
                "n_cal_S": int((ca_y == S).sum()),
                "nominal_coverage": round(1 - al, 3),
                "realised_S_coverage": round(realised, 4),
                "auto_coverage_S": round(sm["auto_coverage"][S], 4),
                "auto_precision_S": round(sm["auto_precision"][S], 4),
                "auto_recall_S": round(sm["auto_recall"][S], 4),
                "n_auto_S": int(accepted.sum()),
            }

        # --- F frontier (descriptive) ---
        conf_score = 1.0 - p_te.max(1)
        pf_mask = te_pred == F
        for sc, isf in zip(conf_score[pf_mask], te_y[pf_mask] == F):
            pooled_f.append((float(sc), bool(isf)))
        n_true_F_total += int((te_y == F).sum())

        per_fold.append({
            "name": a["_name"],
            "fold": meta_of(a).get("fold"),
            "seed": meta_of(a).get("seed"),
            "temperature": [float(round(x, 3)) for x in T],
            "classwise_ece_before": {CLASSES[i]: round(cw_before[i], 4) for i in range(5)},
            "classwise_ece_after": {CLASSES[i]: round(cw_after[i], 4) for i in range(5)},
            "conformal_S": s_res,
        })

    frontier = []
    if pooled_f:
        sc = np.array([r[0] for r in pooled_f])
        isf = np.array([r[1] for r in pooled_f])
        for thr in np.quantile(sc, np.linspace(0.05, 1.0, 20)):
            acc = sc <= thr
            n_acc = int(acc.sum())
            frontier.append({
                "threshold": round(float(thr), 4),
                "n_accepted_pred_F": n_acc,
                "auto_precision_F": round(float(isf[acc].mean()), 4) if n_acc else None,
                "auto_coverage_F": round(float(isf[acc].sum() / n_true_F_total), 4) if n_true_F_total else None,
            })

    return {"tag": tag, "result_dir": str(result_dir), "n_fold_seed": len(folds),
            "per_fold": per_fold, "f_frontier_pooled": frontier}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for tag, d in (("vrce", Path("results/mp_lopo")), ("flat", Path("results/lopo_cv"))):
        r = analyse_tag(d, tag)
        report[tag] = r
        (OUT / f"analysis_{tag}.json").write_text(json.dumps(r, indent=2))
        if "error" in r:
            print(f"[{tag}] {r['error']}")
            continue
        print(f"[{tag}] {r['n_fold_seed']} fold-seed artifacts")
        for pf in r["per_fold"]:
            cs = pf["conformal_S"].get("alpha_0.1", {})
            print(f"  fold={pf['fold']} seed={pf['seed']} T={pf['temperature']} "
                  f"| S realised@0.9={cs.get('realised_S_coverage')} "
                  f"auto_prec_S={cs.get('auto_precision_S')} auto_cov_S={cs.get('auto_coverage_S')}")
        fr = r["f_frontier_pooled"]
        if fr:
            mid = fr[len(fr) // 2]
            print(f"  F frontier mid: prec={mid['auto_precision_F']} cov={mid['auto_coverage_F']}")
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[OK] {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
