# E5 — Subtype and patient-reference forensic

## Reproducibility controls

- Raw annotation join: 185,252/185,252; unjoined=0.
- E1 S-AUC reproduction max absolute difference against rounded reference: 0.000335.
- Raw `e` is mapped to AAMI-N by the actual preprocessing; it is not treated as S.
- Geometry uses no F labels to construct each record's N/V centroids.

## E5-A — S subtype stratification

### MIT-BIH composition (seed-invariant)

| Fold | S | A | a | J | S-symbol | unjoined |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 388 | 343 | 43 | 0 | 2 | 0 |
| 1 | 391 | 386 | 3 | 1 | 1 | 0 |
| 2 | 576 | 567 | 0 | 0 | 9 | 0 |
| 3 | 860 | 731 | 97 | 30 | 2 | 0 |
| 4 | 2490 | 2427 | 7 | 52 | 4 | 0 |

### Cross-fold subtype AUC spreads (support ≥20 per cell)

| Model | Source | Subtype | folds | AUC by fold | spread |
|---|---|---|---:|---|---:|
| r2 | mitbih | A | 5 | 0:0.664, 1:0.978, 2:0.956, 3:0.977, 4:0.785 | 0.314 |
| r2 | mitbih | a | 2 | 0:0.966, 3:0.958 | 0.009 |
| r2 | mitbih | J | 2 | 3:0.773, 4:0.711 | 0.062 |
| r2 | incart | A | 5 | 0:0.923, 1:0.989, 2:0.939, 3:0.993, 4:0.991 | 0.070 |
| flat | mitbih | A | 5 | 0:0.367, 1:0.882, 2:0.823, 3:0.917, 4:0.437 | 0.550 |
| flat | mitbih | a | 2 | 0:0.922, 3:0.783 | 0.139 |
| flat | mitbih | J | 2 | 3:0.545, 4:0.781 | 0.236 |
| flat | incart | A | 5 | 0:0.751, 1:0.980, 2:0.838, 3:0.986, 4:0.982 | 0.235 |

## E5-B — F patient-conditional geometry

Headline tertile tables pool outer-test F predictions across five folds and two seeds; repeated seed predictions are descriptive, not independent beats.

### embedding / euclidean / intermediacy

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.003 | 0.620 |
| 2 | 655 | 0.139 | 0.450 |
| 3 | 656 | 0.555 | 0.194 |

### embedding / euclidean / d_N

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.046 | 0.936 |
| 2 | 655 | 0.345 | 0.316 |
| 3 | 656 | 0.306 | 0.012 |

### embedding / cosine / intermediacy

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.002 | 0.597 |
| 2 | 655 | 0.131 | 0.467 |
| 3 | 656 | 0.564 | 0.200 |

### embedding / cosine / d_N

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.024 | 0.968 |
| 2 | 655 | 0.307 | 0.292 |
| 3 | 656 | 0.366 | 0.005 |

### logits / euclidean / intermediacy

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.000 | 0.551 |
| 2 | 655 | 0.218 | 0.374 |
| 3 | 656 | 0.479 | 0.338 |

### logits / euclidean / d_N

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.032 | 0.963 |
| 2 | 655 | 0.275 | 0.290 |
| 3 | 656 | 0.390 | 0.011 |

### logits / cosine / intermediacy

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.000 | 0.649 |
| 2 | 655 | 0.061 | 0.447 |
| 3 | 656 | 0.636 | 0.168 |

### logits / cosine / d_N

| Tertile (low→high) | n | predicted F | predicted N |
|---:|---:|---:|---:|
| 1 | 655 | 0.005 | 0.995 |
| 2 | 655 | 0.443 | 0.267 |
| 3 | 656 | 0.250 | 0.002 |

### Explicit dominant-patient control (embedding/cosine intermediacy)

- **mitdb/208:** T1 n=248, pred-F=0.000, pred-N=0.524; T2 n=248, pred-F=0.319, pred-N=0.363; T3 n=248, pred-F=0.540, pred-N=0.262
- **mitdb/213:** T1 n=241, pred-F=0.000, pred-N=0.519; T2 n=241, pred-F=0.000, pred-N=0.635; T3 n=242, pred-F=0.397, pred-N=0.248

### Isolation correlations

| Space | Target | n record-seed | Spearman rho | descriptive p |
|---|---|---:|---:|---:|
| logits | F_recall | 68 | -0.124 | 0.315 |
| logits | F_precision | 54 | -0.228 | 0.098 |
| embedding | F_recall | 68 | 0.157 | 0.200 |
| embedding | F_precision | 54 | 0.227 | 0.098 |

## Facts and interpretation

### Observations

- E5-A exactly joins every pool beat, so subtype findings are not driven by silent alignment loss.
- MIT-BIH A (atrial premature) is the only subtype with support ≥20 across all five folds; rare `a` and `J` appear in only two eligible folds, and raw `S` has insufficient support.
- E5-B shows a strong within-representation gradient: high intermediacy tertiles are much more often predicted F, while low d_N tertiles are overwhelmingly predicted N. This holds in logits and embeddings and with cosine/Euclidean distances.
- The patient-isolation correlation is weak, non-significant descriptively, and changes sign between logits and embeddings.

### Anomalies and contradictions

- The protocol names `e` as an S subtype, but repository preprocessing maps `e` to AAMI-N; using it as S would corrupt the control.
- The protocol's H-F4 success criterion says recall correlates positively with isolation, while its parenthetical mechanism says isolated F patients fail, which predicts a negative correlation.
- Embedding spaces from independently trained fold models are not cross-fold aligned. Isolation is therefore computed only among F records inside the same fold/model; logits provide the semantically aligned control.
- Geometry separation can be partly tautological in logit space because argmax-F is defined from those same logits. Replication in the pre-head shared embedding reduces but does not remove selection effects.

### Hypothesis support/refutation

- **H-S3 rejected by its registered criterion:** MIT-BIH `A` is the only subtype eligible across all five folds and remains highly variable (spread 0.314). Stable `a` appears in only folds 0/3 and is easy, while hard `J` is only 52/2,490 S beats in fold 4; subtype mix does not explain low aggregate folds 0/4.
- **H-F4 geometric-intermediacy clause:** supported—correct F predictions concentrate at high intermediacy and F→N at low d_N.
- **H-F4 patient-isolation clause:** not supported robustly—the correlation is weak and representation-dependent. Therefore the full registered H-F4 success criterion is not met.
- **H3/missing context:** remain viable alternatives; geometry explains model behavior but does not prove why consensus labels occupy those regions.

### What E5 does not prove

- It does not show that subtype-specific S performance will generalize to new patients when several subtype cells have only two eligible folds.
- It does not establish causality between subtype mix and aggregate fold AUC without a composition-adjusted/patient-weighted analysis.
- It does not prove a patient-adaptive F model will work; the required isolation-performance relationship was not robust.
- It does not estimate an F human/Bayes ceiling or distinguish label ambiguity from missing temporal/clinical context.
