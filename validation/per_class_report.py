"""Per-class precision/recall/F1 + confusion matrices for the journal revision.

Inference only (no training). Sources:
  DS2  : ResMemECG seeds 42/123/456 (official recipe) and the Table-1 deep baselines
         (results/checkpoints/best_*.pt); LR/DT are refitted deterministically
         (random_state=42), exactly as in run_baselines.py.
  SG-CV: results/controlled_mit_subject_cv/fold*_seed*_flat_preds.npz.
Outputs: results/tables/per_class_ds2.{json,csv,tex}, per_class_sgcv.{json,csv,tex}.

    python -m validation.per_class_report
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report_results import load_hmr_bilstm  # noqa: E402
from run_baselines import (LSTMBaseline, ResNet1D, InceptionTime, TCN, ECGTransformer,  # noqa: E402
                           run_logistic_regression, run_decision_tree, flatten_sequences)

OUT = ROOT / "results" / "tables"
CLS = ["N", "S", "V", "F"]
SPLITS = ROOT / "data" / "processed" / "splits"


def prf(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                "support": int(s[i])} for i, c in enumerate(CLS)}


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(-1).cpu().numpy()
                           for i in range(0, len(X), bs)])


def ds2(device):
    te = np.load(SPLITS / "inter_test.npz")
    X, y = te["X"].astype(np.float32), te["y"].astype(np.int64)
    rows = {}
    ck = ROOT / "results" / "checkpoints"
    for seed, path in ((42, ck / "inter_best_rlstm.pt"), (123, ck / "ensemble/model_123.pt"),
                       (456, ck / "ensemble/model_456.pt")):
        model, _ = load_hmr_bilstm(str(path), device)
        rows[f"ResMemECG_s{seed}"] = predict(model, X, device)
    deep = {
        "LSTM": (lambda: LSTMBaseline(1, 96, False, 0.25, 5), "best_lstm.pt"),
        "BiLSTM": (lambda: LSTMBaseline(1, 96, True, 0.25, 5), "best_bilstm.pt"),
        "ResNet1D": (lambda: ResNet1D(1, 5, 64, 0.25), "best_resnet1d.pt"),
        "InceptionTime": (lambda: InceptionTime(1, 5, 24, 6, 0.25), "best_inceptiontime.pt"),
        "TCN": (lambda: TCN(1, 5, (32, 32, 64, 64), 7, 0.2), "best_tcn.pt"),
        "ECG-Transformer": (lambda: ECGTransformer(1, 96, 4, 3, 192, 0.25, 5), "best_ecgtransformer.pt"),
    }
    for name, (build, fname) in deep.items():
        m = build().to(device)
        m.load_state_dict(torch.load(ck / fname, map_location=device, weights_only=False))
        rows[name] = predict(m, X, device)
    tr = np.load(SPLITS / "inter_train.npz")
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced",
                            n_jobs=-1, solver="lbfgs").fit(flatten_sequences(tr["X"]), tr["y"])
    dt = DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, class_weight="balanced",
                                random_state=42).fit(flatten_sequences(tr["X"]), tr["y"])
    rows["Logistic regression"] = lr.predict(flatten_sequences(X))
    rows["Decision tree"] = dt.predict(flatten_sequences(X))

    res = {name: {"per_class": prf(y, p),
                  "macro_f1_nsvf": float(np.mean([prf(y, p)[c]["f1"] for c in CLS])),
                  "confusion_NSVFQ": confusion_matrix(y, p, labels=[0, 1, 2, 3, 4]).tolist()}
           for name, p in rows.items()}
    seeds = [res[f"ResMemECG_s{s}"] for s in (42, 123, 456)]
    agg = {c: {m: [float(np.mean([r["per_class"][c][m] for r in seeds])),
                   float(np.std([r["per_class"][c][m] for r in seeds], ddof=1))]
               for m in ("precision", "recall", "f1")} for c in CLS}
    res["ResMemECG_3seed_mean_sd"] = agg
    return res


def sgcv():
    root = ROOT / "results" / "controlled_mit_subject_cv"
    out = []
    for fold in range(5):
        per_seed = []
        for seed in (42, 123):
            d = np.load(root / f"fold{fold}_seed{seed}_flat_preds.npz")
            per_seed.append(prf(d["y"].astype(int), d["logits"].argmax(1)))
        row = {"fold": fold}
        for c in CLS:
            row[c] = {m: float(np.mean([s[c][m] for s in per_seed])) for m in ("precision", "recall", "f1")}
            row[c]["support"] = per_seed[0][c]["support"]
        out.append(row)
    mean = {c: {m: float(np.mean([r[c][m] for r in out])) for m in ("precision", "recall", "f1")} for c in CLS}
    return {"per_fold_seed_mean": out, "fold_mean": mean}


def write(ds, sg):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_class_ds2.json").write_text(json.dumps(ds, indent=2))
    (OUT / "per_class_sgcv.json").write_text(json.dumps(sg, indent=2))
    order = ["Logistic regression", "Decision tree", "LSTM", "BiLSTM", "ResNet1D",
             "InceptionTime", "TCN", "ECG-Transformer"]
    with open(OUT / "per_class_ds2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{c}_{m}" for c in CLS for m in ("P", "R", "F1")])
        for name in order + [f"ResMemECG_s{s}" for s in (42, 123, 456)]:
            pc = ds[name]["per_class"]
            w.writerow([name] + [f"{pc[c][m]:.3f}" for c in CLS for m in ("precision", "recall", "f1")])
    lines = []
    for name in order:
        pc = ds[name]["per_class"]
        lines.append(name + " & " + " & ".join(f"{pc[c][m]:.3f}" for c in CLS
                                                for m in ("precision", "recall", "f1")) + r" \\")
    agg = ds["ResMemECG_3seed_mean_sd"]
    lines.append(r"ResMemECG (3 seeds) & " + " & ".join(
        f"{agg[c][m][0]:.3f}$\\pm${agg[c][m][1]:.3f}" for c in CLS for m in ("precision", "recall", "f1")) + r" \\")
    (OUT / "per_class_ds2.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for r in sg["per_fold_seed_mean"]:
        lines.append(f"{r['fold']} & " + " & ".join(f"{r[c][m]:.3f}" for c in ("S", "F")
                                                     for m in ("precision", "recall", "f1")) + r" \\")
    m = sg["fold_mean"]
    lines.append("Mean & " + " & ".join(f"{m[c][k]:.3f}" for c in ("S", "F")
                                        for k in ("precision", "recall", "f1")) + r" \\")
    (OUT / "per_class_sgcv.tex").write_text("\n".join(lines) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, sg = ds2(device), sgcv()
    write(ds, sg)
    for name, r in ds.items():
        if "per_class" in r:
            print(f"{name:20s} macro={r['macro_f1_nsvf']:.3f} " + " ".join(
                f"{c}:P{r['per_class'][c]['precision']:.2f}/R{r['per_class'][c]['recall']:.2f}/F{r['per_class'][c]['f1']:.2f}"
                for c in CLS))
    print("SG-CV fold mean:", {c: {k: round(v, 3) for k, v in sg['fold_mean'][c].items()} for c in CLS})


if __name__ == "__main__":
    main()
