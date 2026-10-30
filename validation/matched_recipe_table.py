"""Summarise the matched-recipe controls into results/tables/matched_recipe.{json,tex}.

Reads results/matched_recipe/{full,normc,bilstm}_seed{42,123,456}.json and, when present,
results/matched_recipe/analysis.json (FGSM / PGD-20 / beta from matched_recipe_analysis.py).

    python -m validation.matched_recipe_table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "matched_recipe"
OUT = ROOT / "results" / "tables"
ARCHS = {"bilstm": "BiLSTM (\\texttt{nn.LSTM})", "normc": "No-RMC cell", "full": "ResMemECG"}
SEEDS = (42, 123, 456)


def row(arch, seed, analysis):
    r = json.loads((RUNS / f"{arch}_seed{seed}.json").read_text())
    t, pc = r["test"], r["test"]["per_class"]
    out = {"macro_f1": t["f1_macro"], "acc": t["accuracy"], "auc_macro": t["auc_macro_nsvf"],
           "best_epoch": r["best_epoch"], "n_params": r["n_params"]}
    for c in "NSVF":
        out[f"f1_{c}"], out[f"rec_{c}"], out[f"prec_{c}"] = pc[c]["f1"], pc[c]["recall"], pc[c]["precision"]
        out[f"auc_{c}"] = pc[c]["auc"]
    a = analysis.get(f"{arch}_seed{seed}")
    if a:
        out["fgsm_f1"] = a["fgsm"]["adv_macro_f1_nsvf"]
        out["pgd_sub_clean_f1"] = a["pgd20_subset"]["clean_macro_f1_nsvf"]
        out["pgd_sub_f1"] = a["pgd20_subset"]["adv_macro_f1_nsvf"]
        if "beta_mean" in a:
            out["beta"] = a["beta_mean"]
    return out


def main():
    ap = RUNS / "analysis.json"
    analysis = json.loads(ap.read_text()) if ap.exists() else {}
    res = {a: {s: row(a, s, analysis) for s in SEEDS if (RUNS / f"{a}_seed{s}.json").exists()}
           for a in ARCHS}
    summary = {}
    for a, seeds in res.items():
        if len(seeds) < 2:
            continue
        keys = seeds[SEEDS[0]].keys()
        summary[a] = {k: [float(np.mean([seeds[s][k] for s in seeds])),
                          float(np.std([seeds[s][k] for s in seeds], ddof=1))]
                      for k in keys if all(k in seeds[s] for s in seeds)}
    paired = {}
    for a, b in (("full", "normc"), ("full", "bilstm"), ("normc", "bilstm")):
        if all(s in res.get(a, {}) and s in res.get(b, {}) for s in SEEDS):
            paired[f"{a}-{b}"] = {k: [res[a][s][k] - res[b][s][k] for s in SEEDS]
                                  for k in ("macro_f1", "f1_S", "f1_V", "f1_F", "auc_S", "auc_F")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matched_recipe.json").write_text(json.dumps(
        {"per_seed": res, "mean_sd": summary, "paired_diff_by_seed": paired}, indent=2))

    lines = []
    for a, label in ARCHS.items():
        for s in SEEDS:
            if s not in res.get(a, {}):
                continue
            r = res[a][s]
            lines.append(f"{label} & {s} & {r['macro_f1']:.3f} & {r['f1_N']:.3f} & {r['f1_S']:.3f} & "
                         f"{r['f1_V']:.3f} & {r['f1_F']:.3f} & {r['auc_S']:.3f} & {r['auc_F']:.3f} \\\\")
        if a in summary:
            m = summary[a]
            lines.append(f"\\quad mean (SD) & & " + " & ".join(
                f"{m[k][0]:.3f} ({m[k][1]:.3f})" for k in
                ("macro_f1", "f1_N", "f1_S", "f1_V", "f1_F", "auc_S", "auc_F")) + " \\\\")
        lines.append("\\midrule")
    (OUT / "matched_recipe.tex").write_text("\n".join(lines[:-1]) + "\n")

    for a in summary:
        m = summary[a]
        print(f"{a:7s} macro {m['macro_f1'][0]:.3f}±{m['macro_f1'][1]:.3f}  "
              + "  ".join(f"{c} {m['f1_' + c][0]:.3f}" for c in "NSVF")
              + f"  AUC-S {m['auc_S'][0]:.3f} AUC-F {m['auc_F'][0]:.3f}")
    for k, v in paired.items():
        print(k, {kk: [round(x, 3) for x in vv] for kk, vv in v.items()})


if __name__ == "__main__":
    main()
