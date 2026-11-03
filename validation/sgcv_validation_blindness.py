"""Does the inner validation set rank recurrent cores by their outer-test macro-F1 under the
controlled subject-grouped CV? Reads results/sgcv_arch/<arch>/fold*_seed*.json and the flat
ResMemECG runs (arch `full`) in results/controlled_mit_subject_cv, and reports:
  * per arch: mean (SD) inner-val and outer-test macro-F1 over the available fold-seeds;
  * Spearman between inner-val and outer-test macro-F1 over all runs, and within each fold
    (ranking the cores trained on the same subjects; the within-fold means are the relevant
    selection question, because folds differ in difficulty);
  * how often the core with the best inner-val macro-F1 in a fold/seed is also best on test,
    and its test regret against the best core;
  * paired outer-test differences between cores (same fold, same seed).
Missing archs or fold-seeds are skipped, so it can run on partial results.

    python -m validation.sgcv_validation_blindness
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SGCV = ROOT / "results" / "sgcv_arch"
FLAT = ROOT / "results" / "controlled_mit_subject_cv"
OUT = ROOT / "results" / "tables" / "sgcv_validation_blindness.json"
ARCHS = ["full", "normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]


def load():
    runs = {}
    for a in ARCHS:
        files = FLAT.glob("fold*_seed*_flat.json") if a == "full" else (SGCV / a).glob("fold*_seed*.json")
        for p in files:
            d = json.load(open(p, encoding="utf-8"))
            runs[(a, int(d["fold"]), int(d["seed"]))] = (float(d["best_val_f1_macro"]),
                                                       float(d["test_metrics"]["f1_macro"]))
    return runs


def main():
    runs = load()
    archs = [a for a in ARCHS if any(k[0] == a for k in runs)]
    res = {"per_arch": {}}
    for a in archs:
        v = np.array([runs[k] for k in runs if k[0] == a])
        res["per_arch"][a] = {"n": len(v), "val_mean": v[:, 0].mean(), "val_sd": v[:, 0].std(ddof=1) if len(v) > 1 else None,
                              "test_mean": v[:, 1].mean(), "test_sd": v[:, 1].std(ddof=1) if len(v) > 1 else None}
        print(f"{a:13s} n={len(v):2d} val {v[:, 0].mean():.3f}  test {v[:, 1].mean():.3f} (SD {v[:, 1].std(ddof=1) if len(v) > 1 else 0:.3f})")

    allv = np.array(list(runs.values()))
    if len(allv) > 3:
        rho, p = spearmanr(allv[:, 0], allv[:, 1])
        res["spearman_all_runs"] = {"rho": float(rho), "p": float(p), "n": len(allv)}
        print(f"all runs: Spearman val->test {rho:+.3f} (p {p:.3f}, n {len(allv)})")

    # Within each (fold, seed): rank the cores trained on the same subjects.
    cells = sorted({(k[1], k[2]) for k in runs})
    within, hits, regret = [], [], []
    for f, s in cells:
        present = [a for a in archs if (a, f, s) in runs]
        if len(present) < 3:
            continue
        v = np.array([runs[(a, f, s)] for a in present])
        within.append(spearmanr(v[:, 0], v[:, 1])[0])
        pick = int(v[:, 0].argmax())
        hits.append(pick == int(v[:, 1].argmax()))
        regret.append(float(v[:, 1].max() - v[pick, 1]))
    if within:
        res["within_fold_seed"] = {"n_cells": len(within), "mean_spearman": float(np.nanmean(within)),
                                   "spearman": [float(x) for x in within],
                                   "val_pick_is_test_best": int(sum(hits)),
                                   "mean_test_regret_of_val_pick": float(np.mean(regret))}
        print(f"within fold-seed (n {len(within)}): mean Spearman {np.nanmean(within):+.3f}; "
              f"val pick = test best in {sum(hits)}/{len(hits)}; mean regret {np.mean(regret):.3f}")

    res["paired_test_diff"] = {}
    for a, b in combinations(archs, 2):
        d = np.array([runs[(a, f, s)][1] - runs[(b, f, s)][1] for f, s in cells
                      if (a, f, s) in runs and (b, f, s) in runs])
        dv = np.array([runs[(a, f, s)][0] - runs[(b, f, s)][0] for f, s in cells
                       if (a, f, s) in runs and (b, f, s) in runs])
        if len(d) < 2:
            continue
        res["paired_test_diff"][f"{a}-{b}"] = {"n": len(d), "test_mean": float(d.mean()),
                                               "test_sd": float(d.std(ddof=1)), "n_test_pos": int((d > 0).sum()),
                                               "val_mean": float(dv.mean()), "n_val_pos": int((dv > 0).sum())}
        print(f"{a}-{b}: test {d.mean():+.3f} (SD {d.std(ddof=1):.3f}, {int((d > 0).sum())}/{len(d)} > 0) | "
              f"val {dv.mean():+.3f} ({int((dv > 0).sum())}/{len(d)} > 0)")
    OUT.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
