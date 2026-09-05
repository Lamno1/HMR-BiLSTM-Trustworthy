# ACTIVE_PLAN.md

What we are doing RIGHT NOW and why. One active plan at a time.
See RESEARCH_STATE.md for context, EXPERIMENT_LOG.md for history.

Last updated: **2026-09-05**.

---

## SITUATION

- P1 (LOPO R2 vs flat) + P2 first-cut done. **Verdict split:**
  1. P1 classifier gate: **R2 wins S** (locked criteria met).
  2. Overall Q1 STOP/GO: **NOT passed** - P2 useful S selective auto-decision is
     fold-4-only.
- A month of architecture/loss experiments (~25) all failed. **RESEARCH LOOP
  confirmed at the MODEL layer.** Evidence points to DATA / LABEL / TASK for F,
  and CALIBRATION / FOLD-GENERALIZATION for S.

## RESEARCH LEAD DECISION

**Freeze all training.** Before any further modeling, run two cheap, CPU-only,
high-information diagnostics that discriminate the competing hypotheses and
decide whether the current thesis is salvageable or must pivot.

Do NOT: rerun flat/R2 LOPO; run D2; rerun D1; train any model; run E3/E4.

## ACTIVE EXPERIMENTS

### E1 - LOPO error forensic  [P0, CPU, ~1 day, 0 GPU]

**Research question:** For S - is the per-fold F1 gap a RANKING failure or a
CALIBRATION failure? For F - are true-F->N errors on genuinely ambiguous (low
P(F)-rank) beats, or on well-ranked beats losing the argmax?

**Hypotheses discriminated:** H4 (calibration/decision-rule, fixable) vs
{H2, H3} (unit / label - not fixable by P2).

**Inputs (no training):** `results/mp_lopo/fold*_seed*_vrce_preds.npz`,
`results/lopo_cv/fold*_seed*_flat_preds.npz` (fields: te_logits, te_y,
te_record, te_source, te_rr_raw; also va_/tuning_/conformal_cal_).

**Analysis:**
1. Per fold, per class (S, F, and N/V for reference): AUC (OvR), AP, F1, ECE,
   classwise-ECE, on te. R2 and flat side by side.
2. S: for each fold, F1-S vs AUC-S. If AUC-S is high (>=0.85) on >=4/5 folds
   while F1-S varies 0.14-0.53 -> calibration is the lever.
3. F confusion forensic: take all true-F beats; bin by P(F) quantile (deciles).
   For each decile: fraction predicted N / V / F. Report RR and source record of
   the true-F->N errors. F->N errors in LOW P(F) deciles -> label ambiguity
   (H3). In HIGH P(F) deciles losing to N's mass -> decision rule (H4).
4. Fold-4 vs folds 0-3: is patient mitdb/232 in fold-4 test? Compare fold-4
   test-S RR distribution and record composition to folds 0-3.

**Artifacts:** `results/forensic/E1_report.json` + `results/forensic/E1_notes.md`
(written interpretation answering questions 1-4).

**Success criteria (S = calibration):** S AUC >= 0.85 on >=4/5 folds.
**Failure criteria:** S AUC also degrades (<0.80) on folds 0-3.
**Success criteria (F = decision rule):** F->N errors concentrate in the top 3
P(F) deciles.
**Failure (F = label ambiguity):** F->N errors spread across / concentrate in
the bottom deciles.

**Expected information gain:** decides whether the paper is salvageable via P2
(calibration) or needs a modeling pivot or a task reformulation. For F, decides
whether E3/E4 are worth considering.

---

### E2 - F label ceiling  [P0, ~half day, literature + light compute, 0 GPU]

**Research question:** What is the achievable ceiling for beat-level F
classification on single-lead MIT-BIH - how much do independent labelers (or a
transparent rule) agree with the MIT-BIH F annotations?

**Hypothesis discriminated:** H3 (F label non-identifiability). If the ceiling
is ~0.4-0.5, F1-F 0.9 is mathematically impossible and the "classify F" thesis
is dead as posed.

**Steps, in order (stop as soon as one gives a clear answer):**
1. **Literature search:** published inter-annotator / inter-database agreement
   on MIT-BIH beat types, specifically the F class. Report number(s) + citation.
2. **Rule-vs-label proxy:** a transparent fusion detector - a beat is "rule-F"
   if QRS width is between the per-record N-median and V-median AND a preceding
   P-wave is detectable AND an early/steep ventricular upstroke is present.
   Compute agreement (F1, Cohen's kappa) between rule-F and MIT-BIH F on the
   MIT-BIH F-bearing records. Agreement = a ceiling estimate.
3. If a cardiologist is reachable: 100 blind-relabeled beats (F/N/V mix).

**Artifacts:** `results/forensic/E2_ceiling.md` (citations + rule-vs-label
agreement table + a stated ceiling estimate).

**Expected information gain:** the single most consequential number in the
project. Confirms or kills H3.

---

## STOP / GO DECISION TREE (after E1 + E2)

- **E2 ceiling ~= 0.4 AND E1 S-AUC degrades on folds 0-3**
  -> **EXECUTE THE PIVOT** (below). No E3/E4 for F.
- **E1 S-AUC good (>=0.85) on >=4/5 folds**
  -> S is a calibration problem. Expand P2 to the full mandatory evidence set
     (continuous risk-coverage curve, set-size + singleton/ambiguous rate,
     fold/patient-level CI, fixed operating-points table). Paper = "MultiPath +
     honest LOPO + selective S". E2 still decides F's framing.
- **E1 S-AUC fails everywhere AND E2 ceiling NOT confirmed low**
  -> run E4 (learned beat/rhythm-sequence model) before any pivot. This is the
     one untested modeling assumption (learned sequence representation of rhythm
     context, not hand-crafted RR, not waveform concat).

## PROPOSED PIVOT (Research Lead, confidence ~60%)

From CLASSIFICATION to a HONEST TRIAGE SYSTEM:
1. Drop F as a target class - fold into a `V (union) F` ventricular-origin
   superclass, OR treat F as designated-abstain by design.
2. N / S / (V+F) / Q classification + a calibrated selective layer quantifying
   per-beat AND per-patient uncertainty.
3. Present the quantified F ceiling (from E2) as a scientific contribution:
   "LOPO + inter-annotator analysis shows beat-level fusion classification on
   single-lead ECG has an irreducible ceiling of ~X; fusion beats should be
   referred, not classified."

**What would change my mind / kill the pivot:** E4 shows F1-S (or F1-F) -> 0.5+
CONSISTENTLY across all 5 LOPO folds with a sequence model. Then H2 wins.

## DEFERRED (do NOT start without a Research Lead go)

- E3 (F-record scaling curve) - only if E1 says F is not purely label-bound.
- E4 (sequence model) - only per the decision tree above.
- D1 rerun - needs per-epoch checkpoint + timing instrumentation first
  (see RESEARCH_STATE.md limitations); supplementary, non-blocking.
- D2 (+V1-only ablation) - only if the pivot is rejected and R2 is the paper
  model with V1-vs-rhythm attribution still open.
