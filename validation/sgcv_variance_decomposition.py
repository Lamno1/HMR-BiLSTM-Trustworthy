"""Descriptive variance decomposition of SG-CV outer-test macro-F1 over core, fold and seed.

Reads the same 60 runs as sgcv_validation_blindness (6 cores x 5 outer folds x seeds 42/123),
writes them to a CSV, and splits the total sum of squares of outer-test macro-F1 (N/S/V/F)
in the balanced core x fold layout with two seeds per cell:

    SS_total = SS_core + SS_fold + SS_core:fold + SS_within
    SS_core  = F*S * sum_c (ybar_c.. - ybar)^2
    SS_fold  = C*S * sum_f (ybar_.f. - ybar)^2
    SS_core:fold = S * sum_cf (ybar_cf. - ybar_c.. - ybar_.f. + ybar)^2
    SS_within    = sum_cfs (y_cfs - ybar_cf.)^2          (seed, within core and fold)

Shares are SS / SS_total. The design is balanced, so the four terms are orthogonal and sum to
SS_total. The runs are not independent (both seeds and all six cores use the same subjects in a
fold), so the shares describe these 60 runs; they are not variance-component estimates and carry
no test. The split is cross-checked against a two-way OLS ANOVA (statsmodels).

    python -m validation.sgcv_variance_decomposition            # from the run JSONs
    python -m validation.sgcv_variance_decomposition --from-csv # from the committed CSV only
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from validation.sgcv_validation_blindness import ARCHS, ROOT, load

OUT = ROOT / "results" / "tables" / "sgcv_variance_decomposition.json"
CSV = ROOT / "results" / "tables" / "sgcv_arch_runs.csv"


def decompose(y: np.ndarray) -> dict:
    """y: core x fold x seed array of outer-test macro-F1."""
    n_c, n_f, n_s = y.shape
    g = y.mean()
    core = y.mean(axis=(1, 2)) - g
    fold = y.mean(axis=(0, 2)) - g
    cell = y.mean(axis=2)
    inter = cell - g - core[:, None] - fold[None, :]
    ss = {"core": n_f * n_s * float((core ** 2).sum()),
          "fold": n_c * n_s * float((fold ** 2).sum()),
          "core_x_fold": n_s * float((inter ** 2).sum()),
          "seed_within_cell": float(((y - cell[:, :, None]) ** 2).sum())}
    total = float(((y - g) ** 2).sum())
    assert abs(sum(ss.values()) - total) < 1e-9, "decomposition does not add up"
    return {"ss": ss, "ss_total": total, "share": {k: v / total for k, v in ss.items()}}


def main():
    # the flat ResMemECG run JSONs are not in the repository; the CSV holds all 60 runs
    if "--from-csv" not in sys.argv:
        runs = load()
        folds = sorted({f for _, f, _ in runs})
        seeds = sorted({s for _, _, s in runs})
        missing = [(a, f, s) for a in ARCHS for f in folds for s in seeds if (a, f, s) not in runs]
        assert not missing, f"unbalanced design, missing runs: {missing}"
        with open(CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["core", "fold", "seed", "inner_val_macro_f1", "outer_test_macro_f1"])
            for a in ARCHS:
                for f in folds:
                    for s in seeds:
                        w.writerow([a, f, s, *runs[(a, f, s)]])

    # recompute from the CSV so the reported numbers come from the written table
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    folds = sorted({int(r["fold"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    assert len(rows) == len(ARCHS) * len(folds) * len(seeds), "unbalanced design in the CSV"
    idx = {(r["core"], int(r["fold"]), int(r["seed"])): float(r["outer_test_macro_f1"]) for r in rows}
    y = np.array([[[idx[(a, f, s)] for s in seeds] for f in folds] for a in ARCHS])
    res = decompose(y)

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        df = pd.DataFrame(rows).astype({"fold": str, "outer_test_macro_f1": float})
        tab = anova_lm(smf.ols("outer_test_macro_f1 ~ C(core) * C(fold)", data=df).fit(), typ=2)
        check = {"core": tab.loc["C(core)", "sum_sq"], "fold": tab.loc["C(fold)", "sum_sq"],
                 "core_x_fold": tab.loc["C(core):C(fold)", "sum_sq"],
                 "seed_within_cell": tab.loc["Residual", "sum_sq"]}
        res["statsmodels_max_abs_diff"] = max(abs(check[k] - res["ss"][k]) for k in check)
    except ImportError:
        res["statsmodels_max_abs_diff"] = None

    res.update(n_runs=len(rows), cores=ARCHS, folds=folds, seeds=seeds,
               core_means=dict(zip(ARCHS, y.mean(axis=(1, 2)).round(4).tolist())),
               fold_means=dict(zip(map(str, folds), y.mean(axis=(0, 2)).round(4).tolist())))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res["share"].items():
        print(f"{k:17s} {v:6.1%}")
    print("statsmodels check, max |SS diff|:", res["statsmodels_max_abs_diff"])
    print(f"wrote {CSV.relative_to(ROOT)} and {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
