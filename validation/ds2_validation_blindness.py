"""Can the DS1 validation records rank the matched-recipe configurations by DS2 macro-F1, and
does the record-214 bundle-branch-block failure explain the DS2 gaps between cores?
Reads results/matched_recipe/<arch>_seed<s>.json and the saved DS2 test logits (no retraining):
  * Spearman/Pearson between validation and test macro-F1 over all runs and over
    configuration means (with a configuration bootstrap interval), and the validation rank
    of every configuration;
  * the paired DS2 macro-F1 gaps between cores with and without the beats of record 214.

    python -m validation.ds2_validation_blindness
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from validation.matched_diagnostics import RUNS, SEEDS, SPLITS

OUT = Path(__file__).resolve().parents[1] / "results" / "tables" / "ds2_validation_blindness.json"
GAPS = [("normc", "bilstm"), ("full", "bilstm"), ("full", "normc")]


def macro_f1(y, pred):
    return f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)


def main():
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(RUNS.glob("*_seed*.json"))]
    v = np.array([[r["best_val_f1_macro"], r["test"]["f1_macro"]] for r in runs])
    res = {"n_runs": len(v), "spearman_runs": spearmanr(v[:, 0], v[:, 1])[0],
           "spearman_runs_p": spearmanr(v[:, 0], v[:, 1])[1], "pearson_runs": pearsonr(v[:, 0], v[:, 1])[0]}
    archs = sorted({r["arch"] for r in runs})
    means = np.array([v[[r["arch"] == a for r in runs]].mean(0) for a in archs])
    rng = np.random.default_rng(0)
    boot = [spearmanr(*means[rng.integers(0, len(archs), len(archs))].T)[0] for _ in range(2000)]
    order = np.argsort(-means[:, 0])
    res.update({"n_configs": len(archs), "spearman_configs": spearmanr(means[:, 0], means[:, 1])[0],
                "spearman_configs_boot95": np.nanpercentile(boot, [2.5, 97.5]).tolist(),
                "val_sd_configs": means[:, 0].std(ddof=1), "test_sd_configs": means[:, 1].std(ddof=1),
                "configs_by_val": [{"arch": archs[i], "val": means[i, 0], "test": means[i, 1],
                                    "val_rank": k + 1, "test_rank": int((means[:, 1] > means[i, 1]).sum()) + 1}
                                   for k, i in enumerate(order)]})
    print(f"runs n={len(v)} Spearman {res['spearman_runs']:+.3f} (p {res['spearman_runs_p']:.2f}); "
          f"configs n={len(archs)} Spearman {res['spearman_configs']:+.3f} "
          f"boot {np.round(res['spearman_configs_boot95'], 2)}")
    for row in res["configs_by_val"]:
        print(f"  val #{row['val_rank']:2d} test #{row['test_rank']:2d} {row['arch']:16s} "
              f"{row['val']:.3f} / {row['test']:.3f}")

    y = np.load(SPLITS / "inter_test.npz")["y"]
    rec = np.load(SPLITS / "inter_test_records.npy")
    keep = rec.astype(str) != "214"
    res["record_214"] = {"n_beats": int((~keep).sum())}
    for a, b in GAPS:
        d_all, d_no = [], []
        for s in SEEDS:
            pa = np.load(RUNS / f"{a}_seed{s}_test_logits.npz")["logits"].argmax(1)
            pb = np.load(RUNS / f"{b}_seed{s}_test_logits.npz")["logits"].argmax(1)
            d_all.append(macro_f1(y, pa) - macro_f1(y, pb))
            d_no.append(macro_f1(y[keep], pa[keep]) - macro_f1(y[keep], pb[keep]))
        res["record_214"][f"{a}-{b}"] = {"with": float(np.mean(d_all)), "without": float(np.mean(d_no))}
        print(f"{a}-{b}: gap {np.mean(d_all):+.3f} -> without record 214 {np.mean(d_no):+.3f}")
    OUT.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
