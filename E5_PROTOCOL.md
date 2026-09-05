# E5 PROTOCOL — Subtype + patient-reference forensic

Status: **APPROVED to execute** (Research Lead, 2026-09-05).
Constraints: **CPU-only, no training, no GPU, no new model. Analysis on data
and artifacts already on disk.** Run E5-A then E5-B, then STOP and wait for
review. Do not touch E3/E4/D1/D2.

Context: E1 rejected "F errors are argmax losses" and "S is a global
calibration problem". E2 produced no numerical F ceiling. Two NEW hypotheses
now dominate and are untested:

- **H-S3:** AAMI-`S` aggregates PAC / aberrant-PAC / junctional / SVT / atrial
  flutter. The fold-to-fold S-AUC variance (0.73 / 0.98 / 0.95 / 0.95 / 0.82)
  is subtype-composition variance, not model instability. Saved artifacts
  collapsed subtype identity so this was never visible.
- **H-F4:** `F` is a patient-conditional boundary, not a class. "F for 208" and
  "F for 213" occupy different signal-space regions (E2: same rule, kappa
  -0.088 vs +0.169). A patient-invariant F classifier is structurally
  impossible; F is inherently a few-shot / patient-adaptive problem.

E5 is the only cheap experiment that discriminates H-S3 and H-F4. Every other
option (E3 scaling, E4 sequence model, E6 patient-adaptive F) is a
GPU-training commitment that E5 tells us whether to make.

---

## E5-A — S subtype stratification (tests H-S3)

**Research question:** Does the per-fold S ranking-quality variance collapse
when S beats are conditioned on their original (pre-AAMI) sub-type?

**Hypothesis discriminated:** H-S3 (subtype-composition) vs "S is patient- or
model-unstable".

**Inputs (no training):**
- Raw beat annotations: `data/raw/mitdb/*.atr` (and the INCART equivalents used
  to build the pool) via `wfdb.rdann`. Sub-symbols of interest for AAMI-S:
  `A` (atrial premature), `a` (aberrant atrial premature), `J` (junctional
  premature), `S` (supraventricular premature, other), `e` (atrial escape).
- Model predictions: `results/mp_lopo/fold*_seed*_vrce_preds.npz` and
  `results/lopo_cv/fold*_seed*_flat_preds.npz` (fields: te_logits, te_y,
  te_record, te_source; te_y is AAMI-mapped).
- The pool build mapping (`validation/preprocess_lopo.py`) to align each pooled
  beat back to its (record, sample-index) so the raw sub-symbol can be joined.

**Method:**
1. Build a per-beat join key (record + beat R-peak sample). For every S beat
   (te_y == 1) in every fold's outer test, recover its raw sub-symbol from the
   `.atr` file. Beats that cannot be joined are reported as an "unjoined"
   count, not silently dropped.
2. Per fold, per sub-symbol with support >= 20 in that fold: compute S-vs-rest
   AUC, AP, and argmax F1, using `softmax(te_logits)[:, 1]` as the S score,
   over the fold's full outer-test set (so the negative class is the true
   population). R2 and flat side by side. Two-seed means.
3. Cross-fold view: for each sub-symbol, list its per-fold AUC. Compute the
   spread (max - min) of AUC (a) across folds within a subtype, and (b) the
   original across-fold spread on aggregate AAMI-S (baseline = 0.98 - 0.73 =
   0.247 for R2).
4. Report the subtype composition of each fold's S test set (counts + share).
5. INCART S beats: report separately (INCART uses its own annotation set;
   note which sub-symbols map and which do not).

**Controls:**
- Same score, same negative population, same two-seed averaging as E1.
- Report support per (fold, subtype) cell; suppress cells with support < 20.
- Aggregate-AAMI-S numbers must reproduce E1's fold S-AUC exactly (sanity check).

**Artifacts:** `results/forensic/E5_subtype.json` (per-fold per-subtype
AUC/AP/F1/support + composition tables) and a section in
`results/forensic/E5_notes.md`.

**Success criterion (H-S3 supported):** within at least one high-support
sub-symbol, cross-fold AUC spread <= 0.10 (i.e. the subtype is uniformly
easy or uniformly hard), AND the folds with low aggregate S-AUC (0, 4) are
explained by carrying a high share of a hard subtype.

**Failure criterion (H-S3 rejected):** AUC remains fold-variable (spread
> 0.15) *within* every high-support sub-symbol -> the instability is patient-
or model-level, not subtype.

---

## E5-B — F patient-conditional geometry (tests H-F4 vs H3 vs missing-context)

**Research question:** Are the F beats the model gets right the ones that are
genuine geometric intermediates between that patient's N and V, and the F
beats it gets wrong the ones that sit on top of that patient's N? And does a
patient's F-classifiability track how far that patient's F-region is from
other patients' F-regions?

**Hypotheses discriminated:** H-F4 (patient-conditional boundary) vs H3 (label
ambiguity / irreducible N-coincidence) vs "recoverable with missing context".

**Inputs (no training):**
- `results/mp_lopo/fold*_seed*_vrce_preds.npz`. The saved artifacts contain
  `te_logits` but NOT embeddings. Use one of:
  (a) the pre-softmax logit vector as a 5-d "representation" (cheap, always
      available), OR
  (b) if the per-fold `fold*_seed*_vrce_model.pt` files exist, a small
      forward-only extractor that loads them and returns `HMRMultiPath.embed()`
      output on the saved outer-test indices. CPU forward only, no gradient,
      no training -> permitted under E5's constraints. Prefer (b) if available;
      else (a).
- Raw waveforms + record IDs from `data/processed/lopo/pool.npz` (X, X2, y,
  record) to build per-record N and V templates.

**Method:**
1. For each record with F beats in any outer-test fold: compute that record's
   N-centroid and V-centroid in the chosen representation space, from that
   record's own N and V beats. Centroids are not a trained parameter; state
   whether train-split-only or all-of-record beats were used and why.
2. For each true-F beat in outer test: compute
   `d_N = dist(F_beat, record_N_centroid)`, `d_V = dist(F_beat, record_V_centroid)`,
   and `intermediacy = 1 - |d_N - d_V| / (d_N + d_V)` (1 = equidistant
   intermediate, 0 = sits on one pole). Cosine and Euclidean both.
3. Cross-tab: model-correct (pred == F) vs model-wrong (pred == N) against
   intermediacy tertiles and against `d_N` tertiles. Report fractions.
4. Patient-specificity index: for each record, F-centroid; then
   `iso(record) = mean dist(record_F_centroid, other records' F_centroids)`.
   Correlate `iso(record)` with that record's F-recall and F-precision.
5. Repeat the key cross-tabs for records 208 and 213 explicitly.

**Controls:**
- State the representation used ((a) logits or (b) embeddings) and rerun the
  headline cross-tab in both if feasible.
- Centroids computed without outer-test F labels.
- Report support per cell.

**Artifacts:** `results/forensic/E5_fgeometry.json` and a section in
`results/forensic/E5_notes.md`.

**Success criterion (H-F4 supported):** model-correct F beats concentrate in
the high-intermediacy tertile AND model-wrong (F->N) beats concentrate in the
low-intermediacy / N-coincident tertile, AND per-record F-recall correlates
positively with `iso(record)` (patients whose F looks unlike other patients'
F are the ones the model fails).

**Failure / alternative:** F-correct and F-wrong beats do not separate by
intermediacy -> H-F4 not supported; either labels are noise irrespective of
geometry (H3) or the discriminating info is outside this representation
(missing context).

---

## E5 DECISION TREE (Research Lead updates the 3 knowledge-base docs after)

- **E5-A supports H-S3** (a subtype is fold-stable) -> S is subtype-solvable.
  New direction: subtype-aware S evaluation + P2 restricted to the solvable
  subset. S story becomes publishable properly. NOT a dead end.
- **E5-A rejects H-S3** -> S instability is patient/model-level; S selective
  utility genuinely limited to fold-1-3 calibration + fold-4 (one patient).
- **E5-B supports H-F4** -> F = patient-adaptive problem. Justifies exactly ONE
  GPU experiment: **E6 patient-adaptive F** (few-shot calibration on the target
  patient's first-N-minutes reference beats). Paper F contribution = "why a
  patient-invariant F approach cannot work + a patient-adaptive alternative".
- **E5-B rejects H-F4** (F-wrong beats are N-coincident regardless of geometry)
  -> F is irreducibly N-like for a large fraction; F -> refer-only is the final
  answer, no E6.
- **Both parts inconclusive** -> the limiting factor is label/data quality, not
  method. Recommend an independent relabeling study + more F patients over any
  further modeling.

## After E5

Codex reports observations / anomalies / contradictions / candidate
explanations (not just "done"). Research Lead runs the review, updates
RESEARCH_STATE.md + EXPERIMENT_LOG.md + ACTIVE_PLAN.md, and picks the next step
by information gain (pivot / E6 / subtype-S direction / data acquisition).
