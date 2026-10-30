"""Summarise the multi-seed matched-recipe ablations into results/tables/matched_ablation.{json,tex}.

Each ablation is compared with its reference model trained with the same seeds:
ResMemECG ablations with `full`, nn.LSTM recipe ablations with `bilstm`. Missing runs are
skipped, so the script can be run while run_matched_ablations.bat is still going.

    python -m validation.matched_ablation_table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "matched_recipe"
OUT = ROOT / "results" / "tables"
SEEDS = (42, 123, 456)
# (arch, reference, label)
ROWS = [
    ("full", None, "ResMemECG (reference)"),
    ("nohybrid", "full", "No hybrid fallback ($\\beta_t=1$)"),
    ("nointer", "full", "No interaction term ($m_t=0$)"),
    ("meanpool", "full", "Mean pooling instead of attention"),
    ("normc", "full", "No RMC path"),
    ("full_noadv", "full", "No FGSM training"),
    ("full_ce", "full", "Cross-entropy instead of focal loss"),
    ("full_nocw", "full", "Focal loss without class weights"),
    ("full_nosmooth", "full", "No smoothness penalty"),
    ("bilstm", None, "BiLSTM (\\texttt{nn.LSTM}, reference)"),
    ("bilstm_noadv", "bilstm", "BiLSTM, no FGSM training"),
    ("bilstm_ce", "bilstm", "BiLSTM, cross-entropy"),
    ("bilstm_nocw", "bilstm", "BiLSTM, no class weights"),
]
KEYS = ("macro_f1", "f1_S", "f1_V", "f1_F", "auc_F")


def load(arch):
    out = {}
    for s in SEEDS:
        p = RUNS / f"{arch}_seed{s}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text())["test"]
        pc = t["per_class"]
        out[s] = {"macro_f1": t["f1_macro"], "f1_S": pc["S"]["f1"], "f1_V": pc["V"]["f1"],
                  "f1_F": pc["F"]["f1"], "auc_F": pc["F"]["auc"]}
    return out


def main():
    runs = {a: load(a) for a, _, _ in ROWS}
    res, lines = {}, []
    for arch, ref, label in ROWS:
        r = runs[arch]
        if len(r) < len(SEEDS):
            print(f"{arch:14s} {len(r)}/{len(SEEDS)} seeds done -- skipped")
            continue
        m = {k: [float(np.mean([r[s][k] for s in SEEDS])),
                 float(np.std([r[s][k] for s in SEEDS], ddof=1))] for k in KEYS}
        entry = {"per_seed": r, "mean_sd": m}
        diff_txt = "--"
        if ref and len(runs[ref]) == len(SEEDS):
            d = [r[s]["macro_f1"] - runs[ref][s]["macro_f1"] for s in SEEDS]
            entry["paired_macro_diff_vs_" + ref] = d
            diff_txt = "/".join(f"{x:+.3f}" for x in d)
        res[arch] = entry
        lines.append(f"{label} & {m['macro_f1'][0]:.3f} ({m['macro_f1'][1]:.3f}) & {diff_txt} & "
                     + " & ".join(f"{m[k][0]:.3f}" for k in KEYS[1:]) + " \\\\")
        if arch == "full_nosmooth":
            lines.append("\\midrule")
        print(f"{arch:14s} macro {m['macro_f1'][0]:.3f}±{m['macro_f1'][1]:.3f}  diff {diff_txt}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matched_ablation.json").write_text(json.dumps(res, indent=2))
    (OUT / "matched_ablation.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
