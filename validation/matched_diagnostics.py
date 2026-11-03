"""Diagnostics on the saved DS2 test logits of the matched-recipe runs (no retraining).

  1. confusion matrices of ResMemECG for every seed and summed over seeds
     -> latex/images/confusion_matrix_seeds.png;
  2. how the loss configuration changes the predictions of each recurrent core
     (per-class recall/precision, destination of true F and S beats, number of F predictions);
  3. record-cluster bootstrap of seed-averaged paired macro-F1 differences: the 22 DS2 records
     are resampled with replacement, so beats of one record always stay together, plus the
     leave-one-record-out change of each difference.

Needs data/processed/splits/inter_test_records.npy (written by validation.leakage_audit).

    python -m validation.matched_diagnostics
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import wfdb

from validation.preprocess_aami import AAMI_MAPPING, DS2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mitdb"
RUNS = ROOT / "results" / "matched_recipe"
SPLITS = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "results" / "tables" / "matched_diagnostics.json"
FIG = ROOT / "latex" / "images" / "confusion_matrix_seeds.png"
SEEDS = (42, 123, 456)
CLS = ["N", "S", "V", "F", "Q"]
LOSS_ARCHS = ["bilstm", "bilstm_ce", "bilstm_nocw", "normc", "full", "full_ce", "full_nocw"]
PAIRS = [("full", "normc"), ("full", "bilstm"), ("normc", "bilstm"), ("nohybrid", "normc"),
         ("bilstm_ce", "bilstm"), ("full_ce", "bilstm_ce"), ("full_nocw", "bilstm_nocw"),
         ("full", "reslstm"), ("normc", "reslstm")]
# Cell-implementation variants (run_cell_impl.bat); used only once all their seeds exist.
IMPL = ["normc_nodrop", "normc_nofb", "normc_torchinit", "normc_lstmlike"]
IMPL_DONE = [a for a in IMPL if all((RUNS / f"{a}_seed{s}_test_logits.npz").exists() for s in SEEDS)]
LOSS_ARCHS += IMPL_DONE
PAIRS += [(a, "normc") for a in IMPL_DONE]
if "normc_nodrop" in IMPL_DONE:
    PAIRS.append(("full", "normc_nodrop"))
if all((RUNS / f"full_nodrop_seed{s}_test_logits.npz").exists() for s in SEEDS):
    LOSS_ARCHS.append("full_nodrop")
    PAIRS += [("full_nodrop", "full"), ("full_nodrop", "normc_nodrop")]
B = 2000


def load(arch, seed, y_ref):
    d = np.load(RUNS / f"{arch}_seed{seed}_test_logits.npz")
    assert np.array_equal(d["y"], y_ref), f"{arch} s{seed}: label order differs from the split"
    return d["logits"].argmax(1)


def ds2_symbols():
    """Annotation symbol of every stored DS2 beat, same selection as validation/preprocess_aami."""
    out = []
    for r in DS2:
        rec = wfdb.rdrecord(str(RAW / r))
        ann = wfdb.rdann(str(RAW / r), "atr")
        n = len(signal.resample_poly(rec.p_signal[:, 0], 125, int(round(rec.fs))))
        for samp, sym in zip(ann.sample, ann.symbol):
            peak = int(samp * 125 / rec.fs)
            if sym in AAMI_MAPPING and peak - 90 >= 0 and peak + 97 <= n:
                out.append(sym)
    return np.array(out)


def cm(y, p):
    return np.bincount(y * 5 + p, minlength=25).reshape(5, 5)


def macro_f1(c):
    """Macro F1 over N/S/V/F from (..., 5, 5) count matrices (same as f1_score(labels=[0..3]))."""
    tp = np.diagonal(c, axis1=-2, axis2=-1)[..., :4].astype(float)
    den = c.sum(-2)[..., :4] + c.sum(-1)[..., :4]
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1.mean(axis=-1)


def main():
    y = np.load(SPLITS / "inter_test.npz")["y"]
    rec = np.load(SPLITS / "inter_test_records.npy")
    assert len(rec) == len(y)
    records = sorted(set(rec.tolist()))
    archs = sorted({a for pr in PAIRS for a in pr} | set(LOSS_ARCHS))
    pred = {(a, s): load(a, s, y) for a in archs for s in SEEDS}
    res = {}

    # 1. confusion matrices
    cms = {s: cm(y, pred[("full", s)]) for s in SEEDS}
    cms["sum"] = sum(cms[s] for s in SEEDS)
    res["confusion_full"] = {str(k): v.tolist() for k, v in cms.items()}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, k in zip(axes, [42, 123, 456, "sum"]):
        c = cms[k]
        rn = c / np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{c[i, j]}\n{rn[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if rn[i, j] > 0.6 else "black")
        ax.set_xticks(range(5), CLS); ax.set_yticks(range(5), CLS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Seed {k}" if k != "sum" else "Sum of three seeds")
    fig.tight_layout(); fig.savefig(FIG, dpi=200); plt.close(fig)

    # 2. loss configuration x recurrent core
    loss = {}
    for a in LOSS_ARCHS:
        rows = []
        for s in SEEDS:
            c = cm(y, pred[(a, s)])
            rows.append({
                "recall": (np.diag(c)[:4] / c.sum(1)[:4]).tolist(),
                "precision": (np.diag(c)[:4] / np.maximum(c.sum(0)[:4], 1)).tolist(),
                "true_F_to_N_V_F": (c[3, [0, 2, 3]] / c[3].sum()).tolist(),
                "true_S_to_N": float(c[1, 0] / c[1].sum()),
                "n_pred_F": int(c[:, 3].sum()),
                "n_pred_S": int(c[:, 1].sum()),
            })
        loss[a] = {"per_seed": rows,
                   "mean": {k: np.mean([r[k] for r in rows], axis=0).round(4).tolist()
                            for k in rows[0]}}
    res["loss_by_core"] = loss

    # 2b. original annotation symbol of the N-class beats predicted F (L = left bundle branch block)
    sym = ds2_symbols()
    assert np.array_equal(np.array([AAMI_MAPPING[s] for s in sym]), y)
    res["false_F_from_N_by_symbol"] = {
        "n_beats_by_symbol": {s: int((sym == s).sum()) for s in ("N", "L", "R", "j")},
        **{a: {s: [int(((pred[(a, sd)] == 3) & (y == 0) & (sym == s)).sum()) for sd in SEEDS]
               for s in ("N", "L", "R", "j")} for a in LOSS_ARCHS}}

    # 3. record-cluster bootstrap of seed-averaged paired macro-F1 differences
    rec_cm = {(a, s): np.stack([cm(y[rec == r], pred[(a, s)][rec == r]) for r in records])
              for a in archs for s in SEEDS}
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(records), np.full(len(records), 1 / len(records)), size=B)

    def seed_mean(a, w):  # w: (n, R) record weights -> (n,) seed-averaged macro F1
        return np.mean([macro_f1(np.einsum("nr,rij->nij", w, rec_cm[(a, s)]))
                        for s in SEEDS], axis=0)

    full_w = np.ones((1, len(records)))
    loo_w = 1 - np.eye(len(records))
    boot = {}
    for a, b in PAIRS:
        point = float(seed_mean(a, full_w)[0] - seed_mean(b, full_w)[0])
        d = seed_mean(a, W) - seed_mean(b, W)
        loo = seed_mean(a, loo_w) - seed_mean(b, loo_w)
        k = int(np.argmax(np.abs(loo - point)))
        boot[f"{a}-{b}"] = {
            "point": point, "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
            "p_gt0": float((d > 0).mean()),
            "loo_range": [float(loo.min()), float(loo.max())],
            "most_influential_record": records[k], "diff_without_it": float(loo[k]),
        }
    res["record_bootstrap"] = {"B": B, "n_records": len(records), "pairs": boot}

    OUT.write_text(json.dumps(res, indent=2))
    for k, v in boot.items():
        print(f"{k:24s} {v['point']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"P>0 {v['p_gt0']:.2f}  LOO [{v['loo_range'][0]:+.3f}, {v['loo_range'][1]:+.3f}]  "
              f"max-influence {v['most_influential_record']} -> {v['diff_without_it']:+.3f}")
    for a in LOSS_ARCHS:
        m = loss[a]["mean"]
        print(f"{a:12s} recall NSVF {np.round(m['recall'], 3)}  prec {np.round(m['precision'], 3)}  "
              f"F->N/V/F {np.round(m['true_F_to_N_V_F'], 3)}  S->N {m['true_S_to_N']:.3f}  "
              f"#predF {m['n_pred_F']:.0f}  #predS {m['n_pred_S']:.0f}")


if __name__ == "__main__":
    main()
