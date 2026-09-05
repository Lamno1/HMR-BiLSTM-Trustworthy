"""Render the deterministic E5 JSON artifacts into the Research Lead note."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results"/"forensic"
a=json.loads((OUT/"E5_subtype.json").read_text()); b=json.loads((OUT/"E5_fgeometry.json").read_text())

lines=["# E5 — Subtype and patient-reference forensic","","## Reproducibility controls","",
f"- Raw annotation join: {a['mapping_summary']['joined']:,}/{a['mapping_summary']['pool_beats']:,}; unjoined={a['mapping_summary']['unjoined']}.",
f"- E1 S-AUC reproduction max absolute difference against rounded reference: {a['sanity']['max_abs_vs_e1_rounded']:.6f}.",
"- Raw `e` is mapped to AAMI-N by the actual preprocessing; it is not treated as S.",
"- Geometry uses no F labels to construct each record's N/V centroids.","",
"## E5-A — S subtype stratification","",
"### MIT-BIH composition (seed-invariant)","","| Fold | S | A | a | J | S-symbol | unjoined |","|---:|---:|---:|---:|---:|---:|---:|"]
for r in a["composition_seed42"]:
    s=r["symbols"]; lines.append(f"| {r['fold']} | {r['n_S']} | {s.get('A',0)} | {s.get('a',0)} | {s.get('J',0)} | {s.get('S',0)} | {r['unjoined_S']} |")
lines += ["","### Cross-fold subtype AUC spreads (support ≥20 per cell)","","| Model | Source | Subtype | folds | AUC by fold | spread |","|---|---|---|---:|---|---:|"]
for r in a["cross_fold_spreads"]:
    vals=", ".join(f"{k}:{v:.3f}" for k,v in r["fold_auc"].items())
    lines.append(f"| {r['model']} | {r['scope']} | {r['subtype']} | {r['n_folds']} | {vals} | {r['auc_spread']:.3f} |")

lines += ["","## E5-B — F patient-conditional geometry","",
"Headline tertile tables pool outer-test F predictions across five folds and two seeds; repeated seed predictions are descriptive, not independent beats.",""]
for r in b["cross_tabs"]:
    if r["variable"] not in ("intermediacy","d_N"): continue
    lines += [f"### {r['space']} / {r['distance']} / {r['variable']}","","| Tertile (low→high) | n | predicted F | predicted N |","|---:|---:|---:|---:|"]
    for c in r["cells"]: lines.append(f"| {c['tertile']} | {c['n']} | {c['pred_F']:.3f} | {c['pred_N']:.3f} |")
    lines.append("")
lines += ["### Explicit dominant-patient control (embedding/cosine intermediacy)",""]
for recname,tabs in b["record_cross_tabs"].items():
    r=next(x for x in tabs if x["space"]=="embedding" and x["distance"]=="cosine" and x["variable"]=="intermediacy")
    lines += [f"- **{recname}:** " + "; ".join(f"T{c['tertile']} n={c['n']}, pred-F={c['pred_F']:.3f}, pred-N={c['pred_N']:.3f}" for c in r["cells"])]
lines.append("")
lines += ["### Isolation correlations", "", "| Space | Target | n record-seed | Spearman rho | descriptive p |","|---|---|---:|---:|---:|"]
for r in b["iso_correlations"]: lines.append(f"| {r['space']} | {r['target']} | {r['n_record_seed']} | {r['spearman_rho']:.3f} | {r['pvalue_descriptive']:.3f} |")

lines += ["","## Facts and interpretation","",
"### Observations","",
"- E5-A exactly joins every pool beat, so subtype findings are not driven by silent alignment loss.",
"- MIT-BIH A (atrial premature) is the only subtype with support ≥20 across all five folds; rare `a` and `J` appear in only two eligible folds, and raw `S` has insufficient support.",
"- E5-B shows a strong within-representation gradient: high intermediacy tertiles are much more often predicted F, while low d_N tertiles are overwhelmingly predicted N. This holds in logits and embeddings and with cosine/Euclidean distances.",
"- The patient-isolation correlation is weak, non-significant descriptively, and changes sign between logits and embeddings.","",
"### Anomalies and contradictions","",
"- The protocol names `e` as an S subtype, but repository preprocessing maps `e` to AAMI-N; using it as S would corrupt the control.",
"- The protocol's H-F4 success criterion says recall correlates positively with isolation, while its parenthetical mechanism says isolated F patients fail, which predicts a negative correlation.",
"- Embedding spaces from independently trained fold models are not cross-fold aligned. Isolation is therefore computed only among F records inside the same fold/model; logits provide the semantically aligned control.",
"- Geometry separation can be partly tautological in logit space because argmax-F is defined from those same logits. Replication in the pre-head shared embedding reduces but does not remove selection effects.","",
"### Hypothesis support/refutation","",
"- **H-S3 rejected by its registered criterion:** MIT-BIH `A` is the only subtype eligible across all five folds and remains highly variable (spread 0.314). Stable `a` appears in only folds 0/3 and is easy, while hard `J` is only 52/2,490 S beats in fold 4; subtype mix does not explain low aggregate folds 0/4.",
"- **H-F4 geometric-intermediacy clause:** supported—correct F predictions concentrate at high intermediacy and F→N at low d_N.",
"- **H-F4 patient-isolation clause:** not supported robustly—the correlation is weak and representation-dependent. Therefore the full registered H-F4 success criterion is not met.",
"- **H3/missing context:** remain viable alternatives; geometry explains model behavior but does not prove why consensus labels occupy those regions.","",
"### What E5 does not prove","",
"- It does not show that subtype-specific S performance will generalize to new patients when several subtype cells have only two eligible folds.",
"- It does not establish causality between subtype mix and aggregate fold AUC without a composition-adjusted/patient-weighted analysis.",
"- It does not prove a patient-adaptive F model will work; the required isolation-performance relationship was not robust.",
"- It does not estimate an F human/Bayes ceiling or distinguish label ambiguity from missing temporal/clinical context.",""]
(OUT/"E5_notes.md").write_text("\n".join(lines),encoding="utf-8")
print(OUT/"E5_notes.md")
