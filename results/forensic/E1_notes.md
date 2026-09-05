# E1 — LOPO error forensic

## Direct answers

1. **S ranking vs calibration:** R2 S-AUC by fold = 0.733, 0.980, 0.951, 0.950, 0.816. The pre-registered calibration-lever criterion (AUC >= 0.85 on >=4/5 folds) is NOT MET.

2. **F errors:** among R2 true-F→N errors, 48.7% are in the bottom three P(F) deciles and 4.5% in the top three. This tests whether errors are low-ranked ambiguity or high-ranked argmax losses.

3. **Fold 4:** mitdb/232 present = True; largest S-record share = 55.5%. Top composition = mitbih/mitdb/232: 1382, incart/incart/I33: 589, incart/incart/I20: 179, incart/incart/I22: 124, mitbih/mitdb/223: 73.

4. **Can P2 alone salvage the current S story?** No, not from current evidence. The registered 4/5 high-AUC criterion fails and existing useful selective coverage is fold-4-only. Folds 1–3 do show high S ranking, so a targeted calibration analysis remains informative, but a global 'calibration-only' explanation is contradicted by folds 0 and 4.

## Fold-level MIT-BIH comparison (two-seed means)

| Fold | R2 S AUC | Flat S AUC | R2 S AP | R2 S F1 | R2 S OVR-ECE | R2 F AUC | R2 F AP | R2 F F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.733 | 0.499 | 0.063 | 0.144 | 0.026 | 0.850 | 0.120 | 0.164 |
| 1 | 0.980 | 0.873 | 0.422 | 0.506 | 0.023 | 0.808 | 0.100 | 0.155 |
| 2 | 0.951 | 0.795 | 0.517 | 0.437 | 0.012 | 0.785 | 0.001 | 0.000 |
| 3 | 0.950 | 0.823 | 0.286 | 0.322 | 0.041 | 0.671 | 0.055 | 0.010 |
| 4 | 0.816 | 0.485 | 0.540 | 0.475 | 0.053 | 0.862 | 0.006 | 0.022 |

## Observations and anomalies

- S is heterogeneous: ranking is excellent on folds 1–3, poor on fold 0, and intermediate on fold 4. This is neither a uniform ranking failure nor a uniform calibration failure.
- Fold 4's apparent S success is composition-heavy: record mitdb/232 alone contributes 1,382/2,490 S beats (55.5%). Beat-weighted metrics therefore overrepresent one patient.
- R2 F→N errors are strongly low-rank: bottom three P(F) deciles contain 48.7%, versus 4.5% in the top three. This contradicts the pre-registered decision-rule signature.
- F AUC can remain moderate/high while AP and F1 collapse because prevalence is tiny and false positives dominate. AUC alone previously overstated usable F separation.

## Candidate explanations

1. **S subgroup/domain heterogeneity (supported):** fold-level AUC and AP vary materially; record composition is concentrated.
2. **S calibration-only (partially refuted):** supported on folds 1–3, refuted as a global explanation by fold 0 AUC=0.733 and fold 4 AUC=0.816 plus fold-4-only selective utility.
3. **F argmax/threshold-only (refuted for most errors):** very few F→N errors occupy the highest P(F) deciles.
4. **F label/representation ambiguity or patient-specific morphology (supported but not distinguished):** errors are low-ranked; E1 cannot tell whether labels are ambiguous or signal/representation is insufficient.

## What remains unknown

- Whether specific S annotation subtypes or rhythms explain fold 0/4 ranking loss; AAMI aggregation hides subtype identity in saved artifacts.
- Whether fold-4 selective utility persists under patient-weighted rather than beat-weighted operating points.
- Whether F low-rank errors are human-ambiguous labels or recoverable with unavailable temporal/clinical information.

## Interpretation safeguards

- The table and registered S-AUC gate use MIT-BIH-only outer-test beats; JSON also contains all/INCART scopes.
- Fold means average two seeds; folds, not beats or duplicate seed predictions, are the primary generalization units.
- AUC/AP diagnose ranking; ECE diagnoses calibration. Neither proves label identifiability.
- The F decile pool repeats each outer-test beat once per seed and is descriptive, not an i.i.d. sample.
- RR columns are ratios: pre/local, post/pre, and local/record-median.
