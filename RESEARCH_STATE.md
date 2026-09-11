# RESEARCH_STATE.md

Canonical current research state. Updated after every experiment that changes a
verdict. Source of raw numbers: `results/`, `docs/RARE_CLASS_PROGRESS.md`.

Last updated: **2026-09-09** — C1 controlled MIT-only subject-grouped CV was
authorised after the original closure and is now complete. All 10 runs and the
independent artifact audit passed. Production work only; no additional modeling.

## C1 ADDENDUM (supersedes the old generalisation headline)

- Protocol: MIT-BIH only; 44 records/43 subject groups; records 201+202 bound;
  five folds; seeds 42/123; official 45-epoch recipe; train-N=37,208 per fold.
- Fold-first means: F1-F **0.01085**, F1-S **0.2316**, AUC-F **0.5413**, AUC-S
  **0.7054**. F1-S fold range: **0.039-0.696**.
- The earlier 0.053 flat result belongs to mixed-source record-grouped CV and
  remains only the matched baseline for R2/P2. It was not guaranteed subject-
  disjoint and must not be called LOPO.
- DS1/DS2 0.241 versus C1 0.0109 is evidence of cross-protocol and subject-
  composition sensitivity, not a causal estimate of the split effect.
- Sources: `docs/CONTROLLED_MIT_SUBJECT_CV_PROTOCOL.md` and
  `results/controlled_mit_subject_cv/{REPORT.md,integrity_report.json}`.

## FINAL VERDICT (2026-09-05, after E1/E2/E5/P2-final)

- **F auto-classification: not viable** on single-lead single-beat inter-patient
  MIT-BIH. Errors low-rank (E1), calibration-irrecoverable (E1/P2-final),
  governed by patient-relative N/V geometry with ~1/3 of F geometrically
  N-coincident (E5-B). No numerical ceiling (E2). -> F is REFERRED, not
  classified. Framed operationally, never via a ceiling number.
- **S selective automation: NO-GO.** P2-final pre-registered criterion
  (overall risk <= 0.02 AND auto-S coverage >= 0.15 AND auto-S precision >= 0.90)
  met by 0/3 ranking-reliable folds (fold 2: precision 0.96 but coverage 0.026;
  fold 3: coverage 0.38 but precision 0.30; fold 1: no safe-zone coverage).
  Flat also 0/3. Per-class temperature calibration is NOT the lever (refuted).
- **S ranking: R2 is a real but modest improvement over flat** — S-AUC up on all
  5 folds, subtype-confirmed (MIT-A cross-fold spread halved 0.55 -> 0.31),
  seed-consistent. Present as "rhythm branch recovers prematurity signal", NOT
  "solved S".
- **H-S3 (subtype) REJECTED** (E5-A: MIT-S is ~90%+ subtype A; A-AUC spread 0.31,
  worse than aggregate). Fold-0/4 S failure is patient-level, not subtype.
- **H-F4 mechanism supported, actionable clause rejected** (E5-B: geometry yes;
  patient-isolation <-> performance null, sign-unstable). No basis for E6.
- **Methodological headline:** standard DS1/DS2 overstates rare-class ECG
  performance ~3x (F1-F 0.24 -> 0.07; the single F1-S number hides a per-patient
  0.14-0.53 range). This is the paper's contribution.

---
(historical state below; superseded by the FINAL VERDICT above)

---

## PROBLEM DEFINITION

Inter-patient AAMI 5-class heartbeat classification (N / S / V / F / Q) on
single-lead ECG (MIT-BIH primary; INCART pooled for F/S diversity). Backbone
`hmr_bilstm.py::RLSTMClassifier` (CNN -> 2-layer BiLSTM w/ RMC cell -> attention
pool -> classifier).

## CURRENT OBJECTIVE

Was: "solve F and S" (stable, high F1). **Under revision** - see ACTIVE_PLAN.md.
N and V are done (F1 ~0.96 / ~0.87, tiny variance). The entire open problem is
F and S.

## CURRENT BEST RESULT

- Official DS1/DS2 (best seed): F1-F ~0.24, F1-S ~0.40. **Now known to be a
  single-patient measurement artifact** - test-F is 93% patient 213, test-S 75%
  patient 232.
- Honest LOPO (5 folds x 2 seeds, best-epoch, mitbih block):
  - F1-F: mean **0.07 +- 0.09**, folds 2/3/4 ~= 0.
  - F1-S: mean **~0.38**, fold-heterogeneous 0.14-0.53, seed-stable (spread 0.006-0.038).
  - macro: 0.52 +- 0.03.

## BEST VALIDATED MODEL / PIPELINE

- **Classifier:** HMR-MultiPath R2 = `hmr_multipath.py::HMRMultiPath(use_v1=True,
  use_rhythm=True, use_grl=False)`. MLII trunk + V1->F-residual head + rhythm-MLP
  ->S-residual head. Trained via `train_multipath_lopo.py`, best-epoch selected on
  inner-val {S,V,F} macro-F1.
- **Strongest baseline:** flat `RLSTMClassifier` via `train_lopo_cv.py --arch flat`.
- **Evaluation protocol:** record-level 5-fold grouped CV (`docs/LOPO_PROTOCOL.md`)
  + a sentinel screening fold. This is the PRIMARY F/S evaluation; DS1/DS2 is
  supplementary literature-table only.

## DATASET CHARACTERISTICS

- LOPO pool `data/processed/lopo/pool.npz`: 185,252 beats, 80 records
  (44 MIT-BIH MLII + 36 INCART lead-II w/ 0.5 Hz HP), fields
  X, X2(V1), y, rr(3), record, source, lead2.
- **F:** ~983 beats, 34 records (17 MIT + 17 INCART); MIT-BIH F ~= 2 patients,
  208 dominant. Effective #F-patients ~= 1.
- **S:** ~4,705 beats, 62 records. Healthier but heterogeneous (PAC/APC/junctional
  likely mixed).
- Per-fold F-beat counts: 399/388/82/72/42. S-beat: 388/391/576/860/2490.

## MAIN BOTTLENECK (two-headed, both NOT architecture)

- **F: DATA + LABEL + TASK.** ~1-2 patients of F; label not identifiable from a
  single beat (fusion = N (+) V continuum); task (beat-level F classification)
  likely mis-specified.
- **S: CALIBRATION + FOLD-DEPENDENT GENERALIZATION.** R2 ranks S better (wins the
  P1 gate) but calibrated confidence supports auto-decision on only 1/5 folds
  (P2, fold 4 only).

## KNOWN LIMITATIONS

- Single lead MLII in the official pipeline; V1 only in the 2-channel/LOPO pools.
- `run_d1.py` has no per-epoch checkpoint (rerun prerequisite).
- P2 (`selective/run_p2.py`) is a first cut - missing continuous risk-coverage
  curve, set-size distribution, fold/patient-level CI, operating-points table.
- Conformal validity assumes beat-level exchangeability which LOPO deliberately
  stresses; realized-vs-nominal coverage is the actual evidence, not a guarantee.

## STRONG HYPOTHESES

- **H3** (F label non-identifiability / low ceiling): `AUC_F 0.96` vs `F1_F 0.24`,
  threshold-tuning caps F1-F at 0.296, annotator's own uncertainty language.
- **H1** (F data insufficiency): 25 architectures fail identically; LOPO F1-F ~0.
- **H4-for-S** (S is a calibration problem, not ranking): R2 wins the full-coverage
  S gate but P2 can't commit S on 4/5 folds.

## WEAK / UNRESOLVED HYPOTHESES

- **H2** (beat-vs-sequence unit misspecification): plausible for S (prematurity is
  a neighbor relation) but hand-crafted RR features gave only modest gains.
- **H5** (patient-signature shortcut): likely a *symptom* of H1/H3 - GRL
  de-confounding destroyed F, meaning confound and F signal are inseparable here.

## DEAD ENDS (do NOT re-try)

Any aux loss on the flat 5-way head (confusion-margin, mixture-decomp,
prototype-repulsion, S-boost, OvR); hierarchical head (3 calibration variants);
generative F augmentation (patient-208 wall); naive multi-beat concat;
cross-DB pooling of INCART as-is; threshold tuning alone on F; DSTA-C as base;
GRL record adversary (scrubs F); balanced-softmax as a stack; SWA for R2;
bigger models.

## RULED OUT AS THE BOTTLENECK

Architecture (25+ experiments + LOPO + sentinel); loss/aux-loss tricks; model
size; adversarial de-confounding; cross-DB pooling; generative aug; the DS1/DS2
F1-F 0.24 as evidence of generalization (LOPO disproves it).

## OPEN QUESTIONS / UNKNOWNS

1. Is the per-fold F1-S gap a RANKING failure or a CALIBRATION failure? (E1)
2. Are true-F->N errors on low-P(F)-rank beats (label ambiguity) or high-rank
   beats losing the argmax (decision rule)? (E1)
3. Is fold-4's S success a patient-232 artifact or a distributional coincidence? (E1)
4. What is the inter-annotator / rule-vs-label agreement ceiling for MIT-BIH F? (E2)
5. Does F1-F rise monotonically with #training F-records? (E3, deferred)
6. Does a learned beat/rhythm-SEQUENCE model lift F1-S consistently across all
   5 folds? (E4, deferred - only if E1 says S ranking also fails)
7. Real cause of the D1 3.2 h/epoch (isolated profile says ~8.1 min) - UNPROVEN.

## VERDICTS (keep separate)

1. **P1 classifier gate: R2 WINS S.** dS_fold_mean +0.1445, 4/5 folds not down,
   R2 S-seed-spread 0.033 <= flat 0.036; large consistent S gains on folds 1
   (+0.235/+0.240) and 4 (+0.364/+0.469) across both seeds. F: no full-coverage
   claim (dF_fold_mean +0.017, fold-0 flips sign by seed).
2. **Overall Q1 STOP/GO: NOT passed.** P2 useful S selective auto-decision exists
   on fold 4 only; "effect on >1 fold" fails.
