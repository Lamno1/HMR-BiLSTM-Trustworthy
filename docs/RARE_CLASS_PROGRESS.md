# Rare-class (F & S) R&D progress log

> **Purpose of this file.** A machine-readable lab notebook so any AI agent (Codex,
> a fresh Claude session, etc.) can pick up this project and understand: the
> problem, everything already tried, what was concluded, and the current step.
> Update the **CURRENT STEP** block and the **EXPERIMENT LOG** table after every run.
> Source of truth for individual run numbers = the JSON files under `results/`.

Last updated: **2026-08-31** (two-layer classifier + selective-decision plan locked; P0 instrumentation in progress, no full LOPO run launched).

---

## 0. TL;DR for a new agent

- Paper: "Trustworthy ECG Classification" (HMR-BiLSTM), inter-patient AAMI 5-class
  (N/S/V/F/Q) on MIT-BIH. Backbone = `hmr_bilstm.py::RLSTMClassifier`
  (CNN → 2-layer bidirectional HMR-BiLSTM → attention pool → classifier).
- N and V are **solid** (F1 ≈ 0.96 / 0.87, tiny seed variance). Done, no work needed.
- **The whole problem is F and S**: F1-F ≈ 0.24 mean, swings 0.04↔0.45 across seeds;
  F1-S ≈ 0.33 mean, swings 0.19↔0.43. This instability is the blocker.
- Root cause is **measured, not guessed** (see §2). It is a *measurement + decision-rule
  + confound* problem, **not a feature problem** — `AUC_F = 0.959` already, but
  `F1_F = 0.241`. The model can rank F beats; it can't commit the argmax.
- ~25 architecture/loss experiments over ~2 weeks all lost to the plain baseline.
  Full dead-end list in §5.
- Current classifier strategy: **HMR-MultiPath R2** (`--v1 --rhythm`), selected by
  sentinel screening. Its 5-fold, 2-seed LOPO validation is still required.
- Realistic target: **F1-F 0.35–0.50, F1-S 0.45–0.62, STABLE** (mean up AND seed
  variance down). Not 0.6+. Fusion beats have a real Bayes-error ceiling.
- The paper has two stacked layers: (1) classifier probabilities for every beat;
  (2) a selective decision layer that calibrates them and either auto-decides or
  refers. Full-coverage F1 is the classifier floor, not the trust headline.
- Trust target: report class-conditional auto-coverage and auto-precision for S/F.
  A precision target only counts with non-trivial coverage; it does not claim that
  all F beats can reach F1 0.8–0.9.

---

## 1. CURRENT STEP  (session handoff 2026-08-31 — P0 instrumentation)

**All MultiPath sentinel rungs screened. Final stack = `--v1 --rhythm` (R2).**

**P0 DONE + committed** (`5f22ab4`, `0c89833`, `709ee30`): seeded checkpoints,
`*_preds.npz` + `_model.pt` dump (float32), record-disjoint model-train /
inner-val(=tuning) / conformal-cal(S-only) split, `selective/metrics.py`,
`selective/run_p2.py`, `run_plan.py` driver, `validation/check_lopo_health.py`
+ `inspect_p0_artifacts.py`. Smoke: both trainers PASS 16/16.

**To run the whole chain unattended:** `python -u run_plan.py` (own terminal).
Stages: p1a_r2_lopo -> p1b_flat_lopo -> p2_selective -> [d1_ds12_r2 STUB] ->
d2_v1_ablation. Resumable (`results/plan_state.json`); stops on a failed run,
an abnormal health check, or the d1 stub. `validation/run_d1.py` still to be
written (best done after P1 numbers are in).

**P0 run note (2026-08-31):** a one-epoch R2 smoke launch reached fold-0 training
on CUDA but was terminated externally before the first epoch/checkpoint; it
produced no JSON/NPZ/model artifact and is **not evidence**. Re-run the smoke in
an interactive/persistent terminal before P1.

**P0 CPU split preflight (2026-08-31):** S conformal-cal has 8 F-free/S-bearing
records in every fold, with 174 / 680 / 1566 / 1741 / 400 S beats (folds 0–4).
Inner-val F beats are 25 / 71 / 18 / 25 / 400; their largest-record shares are
56% / 79% / 61% / 72% / 93%. Thus no fold supplies patient-diverse F calibration:
folds 0, 2, and 3 are absolutely small, while fold 4's 400 beats are 93% one
record. Do not treat duplicate predictions from two seeds as additional
calibration samples. They may form an ensemble prediction, but they do not
increase patient/beat evidence. P2-F therefore reports outer-test
risk--coverage frontiers per fold and pooled as descriptive results, not a
per-fold calibrated/guaranteed threshold. Do not use a pooled-inner-val F
operating point in the abstract or conclusion: its beats repeat records across
outer folds and it is not patient-independent. An illustrative
clinician-selectable point may appear only on the frontier figure, explicitly
without a guarantee claim.

**Full sentinel ladder** (test = held-out MIT-BIH 213-F + 232-S; best-epoch model,
SWA in parens; `results/{lopo_cv,mp_lopo}/sentinel_*.json`):

| rung | mitbih F | mitbih S | mitbih N | mitbih macro | verdict |
|---|---|---|---|---|---|
| flat | 0.443 (.411) | 0.032 (.163) | 0.79 | 0.503 (.559) | reference |
| +V1 residual (R1b) | 0.461 (.440) | 0.178 (.169) | 0.80 | 0.563 (.566) | ✅ kept |
| +rhythm (R2) | **0.564** (.338) | **0.274** (.407) | **0.82** | **0.626** (.600) | ✅ **KEPT — final stack** |
| +GRL (R3) | 0.089 (.234) | 0.361 (.388) | 0.82 | 0.497 (.562) | ❌ **rejected** — GRL scrubs F morphology from e_morph (F1-F craters, V −0.13) |
| +balanced (R3b) | 0.285 (.300) | 0.559 (.560) | **0.62** (.80) | 0.582 (.630) | ❌ **rejected as a stack** — trades F+N for S; best-ep N 0.82→0.62, macro < R2 |

Official DS1/DS2: F1-F ~0.24, F1-S ~0.40 (best seed). **R2 on the sentinel
best-epoch model: 213-F 0.24→0.56, 232-S ~0.03→0.27, mitbih macro 0.50→0.63,
N held.** That is the headline candidate.

Notes:
- SWA (0.8E window) still hurts R2's F (best-ep .564 → SWA .338) — the rhythm
  specialist shifts S↑/F↓ late in training; **best-epoch + {S,V,F} selection is
  the model to report**, SWA is not helping this arch. The full 5-fold decides
  whether best-epoch's *fold-to-fold variance* is acceptable (the actual goal).
- De-confounding (tier 2 / A3): handled by INCART patient diversity in train +
  LOPO holding patients out. Forcing it with a record adversary destroys F. Closed.
- balanced-softmax: if the paper's framing later shifts to "maximize S, defer F",
  R3b is on record (232-S 0.56) — but not the default.

### P1 IN PROGRESS (started 2026-08-31 23:47 UTC via `run_plan.py`)

R2 partial (9/10 runs, best-epoch, mitbih block): F1-F 0.07 +- 0.09
(folds 2/3/4 ~= 0), F1-S 0.36 +- 0.13 (fold-heterogeneous 0.14-0.53),
N 0.94, V 0.71, macro 0.52 +- 0.03. Flat baseline not started.

**Confidence discipline (do not overclaim in the paper):**
- F1-F is low on held-out MIT-BIH patients -> the DS1/DS2 0.24 does NOT
  represent patient generalization. This is NOT "Bayes-error ceiling" - that
  needs independent data/labels. Say "not recovered by architecture on this
  data" / "patient-generalization gap".
- S: in the 4 seed-paired folds the seed spread is 0.006-0.038 while the
  fold spread is large. Correct phrasing: "observed variance is currently
  fold-dominated, not seed-dominated." NOT "seed-lottery solved" - needs the
  full 5 folds x 2 seeds.
- Macro stability != F/S stability (N,V inflate macro). Paper leads with
  F1-F, F1-S and per-fold numbers, macro second.
- Never pool the 10 runs as i.i.d. samples.

**P1 primary analysis (after flat completes):**
1. average the two seeds within each fold; report the 5 fold-level values.
2. paired R2 - flat delta per (fold, seed) AND per fold-mean.
3. per fold: F/S support, TP/FP/FN (fold-0 F=0.262 may be a handful of
   correct calls on patient 213, not a durable capability).

**P1 gate (corrected):**
- R2 "wins S" only if mean delta F1-S > 0 AND most folds not down AND R2
  seed spread not worse than flat.
- F: no full-coverage improvement claim unless the paired delta vs flat is
  clearly positive; otherwise F goes entirely to referral/frontier.
- P2 reports realized coverage / risk / set-size / spread per fold; S
  conformal is not called a "guarantee" under patient shift.
- D2 (+V1-only, 33 GPU-h) is NOT auto-run (removed from run_plan.py): only
  if R2 beats flat on S and V1-vs-rhythm attribution is still open.

### THESIS PIVOT (2026-09-03, from R2 partial results)

F1-F ~0.07 on held-out MIT-BIH patients is NOT a classification result. It
cannot support a Q1 claim of "effective F classification" or clinical
auto-decision for F.

| Not defensible | Defensible if P1/P2 confirm |
|---|---|
| "HMR-MultiPath solves F beats" | "LOPO evaluation exposes an F failure-to-generalize that the standard split hides" |
| "conformal guarantees patient safety" | "per-fold calibration shows realized coverage under patient shift, with stated limits" |
| "high accuracy on ECG arrhythmia" | "selective referral identifies the S cases that can be automated and the F cases that must go to an expert" |

- **F -> refer-only / failure analysis.** Report support, TP/FP/FN,
  precision/recall, patient/fold-level spread - not F1 or pooled beats.
- **S -> research signal, not yet trustworthy for auto-decision.** Must show:
  risk-coverage curve, achieved coverage, selective risk, prediction-set
  size, uncertainty/CI at patient/fold level (NOT beat-level i.i.d.), and a
  concrete useful operating point (coverage where S auto-handling has
  substantially lower error + how many cases referred).

### STOP / GO after p1_gate + P2

Continue to a Q1 write-up ONLY if ALL of:
1. flat baseline complete AND paired R2 - flat shows R2 clearly wins S
   (mean dF1-S > 0, >=3/5 folds not down, seed spread not worse).
2. P2 selective system reaches a useful S operating point: coverage well
   above ~0 with selective risk substantially below the full-coverage error.
3. DS1/DS2 is supplementary only; not used to assert generalization if LOPO
   says otherwise.

If flat ~= R2 on S, OR selective utility coverage ~= 0 (risk reduced only by
referring almost everything): STOP before writing. Redirect effort to
data/label quality and F representation, not more ablations.

**P2 evidence gaps to add before write-up** (current run_p2.py is a first
cut): S risk-coverage *curve* (sweep, not 2 alpha points); prediction-set
size distribution; fold-level bootstrap CI (over folds, not beats); the
"useful operating point" readout. Do this in a fresh session when flat +
p1_gate are done.

**Two-layer / four-block plan (locked):**

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | Per-fold logits/model artifacts, correct seed-resume behavior, flat two-seed runner, selective metrics, smoke test | Must pass before GPU-scale P1 |
| P1 | R2 and flat full 5-fold LOPO, two independent seeds | Compare mean **and** seed/fold spread |
| P2 | Nested calibration/referral analysis; Mondrian sets; RCPS only where calibration count supports it | S is primary guarantee candidate; F is descriptive unless its bound is meaningful |
| P3 | DS1/DS2 R2 3-seed, ablations, paper revision | Uses P1/P2 evidence only |

**P0 implementation notes:**
- Save best-epoch logits (and SWA logits for ablation), labels, record/source,
  raw RR, normalisation metadata, and a model-state artifact for each `(fold, seed, tag)`.
- To protect the classifier's tiny F-record pool, **inner-val is also the tuning
  split** for calibration/referral-score fitting. A separate conformal-cal split
  is carved randomly from S-bearing, F-free records (default 15% of records).
  Therefore S can receive a split-conformal *nominal guarantee under the usual
  exchangeability assumption* if its count supports it; LOPO's patient shift
  makes realized outer-fold coverage versus nominal coverage the actual evidence.
  F is per-fold descriptive in this paper iteration, not guaranteed.
- Checkpoint filenames **must include seed**. Previously, a seed-123 run could
  silently resume seed-42's checkpoint and falsify the variance result.
- `train_lopo_cv.py` needs `--seeds` matching MultiPath so flat and R2 have the
  same independent-run protocol.
- `selective/metrics.py` must keep denominators explicit: auto-coverage is
  `P(accept | Y=c)`, while auto-precision is `P(Y=c | accept, prediction=c)`.
- MC-dropout is post-hoc inference from saved models, optional and never a reason
  to re-run P1.

**P1 commands after P0 smoke succeeds — ONE GPU job at a time (LOPO_PROTOCOL.md):**
1. **Full 5-fold LOPO, 2 seeds, on R2** (`--v1 --rhythm`) — the real validation.
   One process; it loops folds×seeds internally. Do NOT split with `--folds` for
   parallelism — the 4 GB GPU runs a single trainer at a time.
   ```
   python -u train_multipath_lopo.py --mode kfold --v1 --rhythm --k 5 --epochs 25 --seeds 42 123 --out-dir results/mp_lopo
   ```
   Then, only after (1) finishes:
   ```
   python -u train_lopo_cv.py --arch flat --k 5 --epochs 25 --seeds 42 123 --out-dir results/lopo_cv
   ```
   AMP on; resumable per (fold, seed); ~3 days GPU wall total for both runs.
   Writes `fold{f}_seed{s}_{tag}.json` + `fold{f}_seed{s}_{tag}_preds.npz` +
   `_model.pt`. **Success = per-fold F1-F/F1-S mean ↑ vs the flat baseline AND
   std small** (that is the whole point — not a single number).
2. **P2 protocol, after P1 only:** inner-val is the tuning split (calibration +
   referral-score fitting); the conformal-cal split (S-bearing, F-free records) is
   distinct and never touched until the decision function is frozen; never tune
   anything on outer test. Use per-fold Mondrian sets
   as the first decision rule and show the alpha/coverage curve. Do **not** call a
   simple pool of residuals from five outer models “grouped CV+” or claim a
   patient-level guarantee from it. RCPS is supplementary and only for groups
   with enough independent calibration examples; report per-fold spread and the
   patient-exchangeability limitation in all cases.
3. **P2-F is frontier-only.** Every fold's inner-val F is 56–93 % one record
   (25/71/18/25/400 beats), so no patient-diverse F threshold calibration is
   possible. Report outer-test F risk–coverage frontiers per fold and pooled as
   descriptive results. No pooled-inner-val F operating point in the abstract or
   conclusion. An illustrative clinician-selectable point may appear only on the
   frontier figure, without a guarantee claim.
4. DS1/DS2 3-seed of R2 for the literature table — needs a MLII+V1 loader on the
   official split (§6; INCART not involved there, V1 = `p_signal[:,1]`).
5. Ablation table (flat / +V1 / +rhythm) + paper write-up (plan § D-E).

**Everything BUILT & committed**: `hmr_multipath.py`, `train_multipath_lopo.py`,
`train_lopo_cv.py` (+AMP), `validation/preprocess_lopo.py` v2, `docs/LOPO_PROTOCOL.md`.
Pool: `data/processed/lopo/pool.npz` (X, X2, y, rr, record, source).

### Hierarchical head: CLOSED (2026-08-30)
All three calibration variants failed on DS2 (vs official F .255 / S .397):
`hier` F .173 / S .261 · `hier_la` F .019 · `hier_ba` F .097 / S .218. The
decision head alone is not a big enough lever. Per the locked plan the binary
abnormal gate does NOT go into HMR-MultiPath; rung-4 calibration is
balanced-softmax/logit-adjust on the flat 5-way head, decided on its own merits.
`hmr_hd.py` / `train_hd_ladder.py` kept as evidence.

### The LOPO / grouped-K-fold harness (built 2026-08-29)
Per the external review (2026-08-29): the biggest gap is *evaluation*, not
architecture — the fixed DS1/DS2 F1-F/F1-S can never become "stable" because
test-F is one patient. Build the record-level CV harness BEFORE combining
architecture changes.
- `validation/preprocess_lopo.py` → `data/processed/lopo/pool.npz` (185k beats,
  80 records: 44 MIT-BIH MLII + 36 INCART F/S-bearing lead-II w/ 0.5 Hz HP).
  F from 34 records (17 MIT + 17 INCART), S from 62. **DONE, built.**
- `train_lopo_cv.py` — grouped K-fold (K=5), records as fold unit, snake-drafted
  by F/S beat count so 208/213/232 land in different folds. Per-fold: inner-val
  (record-disjoint, ~15%) for model selection on {S,V,F}-macro; per-source
  z-score fit on train-fold only; **SWA over tail epochs** (the B2 fix); eval
  split by database (MIT vs INCART). Resumable per fold. **BUILT, helper unit
  tests pass, needs a 1-fold GPU smoke before the full run.**
  `python train_lopo_cv.py --arch flat --k 5 --epochs 25`
- Fold F-beat counts: 399/388/82/72/42 (folds 0,1 carry 208,213; 2-4 carry the
  diffuse F tail + INCART). S-beat: 388/391/576/860/2490 (fold 4 carries 232).
  Heterogeneous by construction — the 5 fold values ARE the patient distribution.
- **Run order**: hier_ba finishes → LOPO `flat` baseline (~20-27h, one-time
  reference distribution) → then LOPO is the eval for steps 3/4/5.

---

## 2. THE PROBLEM, decomposed into tiers (all evidence measured this project)

| Tier | Problem | Fixable by modeling? | Evidence |
|---|---|---|---|
| **1. Measurement** | Test-F is 93% one patient (213); test-S is 75% one patient (232); train-F is 97% one patient (208). F1-F/F1-S ≈ single-patient→single-patient generalization → irreducible seed variance. | **NO** | `_scratch_fclass_patient_analysis.py`; effective #patients ≈ 1.1 |
| **1b. Val is blind** — AND fixing it did not help | Val split has 30 F / 3 patients (none is 208 or 213). Making val rich (75 F / 9 patients via INCART) still gave best_epoch 5 and F1-F swing .19–.45. The variance is about the **test** patient, which no val contains. | **NO** (train/val side) | Stage A v2 result, 2026-08-29 |
| **2. Patient-identity confound** | 208-N vs 213-N separable at **AUC 1.000** (normal beats!). N-vs-F transfer only 0.904. Part of the model's AUC_F 0.96 is "looks like patient 208", which doesn't transfer → F→N 45%. | **YES** | `_scratch_patient_signature.py` |
| **3. Decision rule** | Flat 5-way softmax at 0.8% F base rate: `AUC_F 0.959` but `F1_F 0.241`. True-F argmax: →N 44.6%, →V 26.5%, →F 27.8%. Threshold tuning caps F1-F at 0.296. | **YES** | official ckpt eval, `evaluate_entropy_thresholding.py` context |
| **3b. Model selection** | Early-stop on val macro-F1 is dominated by stable N/V; picks an F-lucky epoch ~randomly → the 0.037↔0.432 F1-F swing, seed-independent. | **YES** (partly) | multi-seed logs; seed456 had normal val (.556), collapsed on test |
| **4. Discarded information** | Pipeline uses only lead MLII (`p_signal[:,0]`). Lead V1 lifts N-vs-F transfer 0.904→0.957. RR features help but were throttled ~140× when fed through the LSTM gate. | **YES** | `_scratch_fclass_patient_analysis.py`, `project_dsta_architecture_research` |
| **5. Bayes-error ceiling** | Fusion beats are by definition normal+PVC mixtures. `213.hea`: they "vary from almost normal to almost identical to that of the PVCs." Some F→N / F→V errors are *correct given the signal*. | **NO** | MIT-BIH annotation notes |
| **6. Cross-DB domain shift** | INCART S morphology ≠ MIT-BIH S (lead II vs MLII, no baseline filter, different population). Pooling INCART into train **collapsed DS2 S** .33→.17. | **YES (partial)** | Stage A v2 result |

**Synthesis.** A viable breakthrough = **tiers 2 + 3 + 4 combined**, evaluated via
**leave-one-patient-out CV** (the only fix for tier 1). Tiers 1 and 5 cap the
achievable number and must be stated plainly in the paper.

---

## 3. Target architecture: HMR-MultiPath (LOCKED 2026-08-29 via external review)

ONE model. DSTA-C and CAMP-Net are frozen as evidence/fallback — they add
gate/state/loss complexity on the same MLII + shared representation (wrong
inductive bias); they do not address the measured causes. Do NOT build/run
DSTA-HD or CAMP-HD in parallel.

```
MLII ── HMR-BiLSTM trunk ── e_morph ─────────────┐
                                                  ├─ shared N/V/Q logits (from e_morph)
V1 ─── lightweight CNN ───── e_v1 ───────────────┤
                                                  ├─ F-specialist: [e_morph, e_v1] → z_F
RR/context ─ rhythm MLP ──── e_rhythm ───────────┤
                                                  └─ S-specialist: [e_morph, e_rhythm] → z_S

e_morph ── class-conditional GRL record adversary  (adversary sees [e_morph, onehot(y_true)])

[z_N, z_S, z_V, z_F, z_Q] ── softmax ── calibrated/balanced 5-way loss   (NO hierarchical gate)
```

Design principles (locked):
- HMR stays the morphology trunk — only thing that ever held S and F together.
- V1 → **only** the F branch. Not both leads through a shared CNN (that halved F1-S).
- RR/context → **only** the S branch. Not through the recurrent gate (β throttled 140×).
- F and S get **separate heads** — no shared classifier forced to learn two
  opposing clinical definitions.
- GRL = **conditional record adversary**: input `[e_morph, onehot(y_true)]`,
  predicts record_id, reversed gradient → encoder drops record info *conditioned
  on class*, keeping the class signal. N-only adversary = diagnostic/ablation
  backup only. λ ramp 0→0.2. Kill rule: F1-N or F1-V drops clearly, OR seed/fold
  variance does not improve.
- Final head is class-specific logits from different feature subsets → concat →
  softmax. **Not** a hierarchical binary gate. Differentiable.
- hier_ba (§4) only tells us whether a balance mechanism works *when a binary
  abnormal gate exists*. MultiPath has no such gate → do NOT carry `pos_weight`
  over. Step-4 calibration is decided **independently** for the flat 5-way head
  (balanced-softmax / logit-adjust on the 5 final logits is a valid, not-yet-run
  test regardless of hier_ba's outcome).

### Build order (one module per rung, ablate via sentinel then LOPO)
| rung | add | attacks | gate to promote |
|---|---|---|---|
| 0 | HMR trunk + flat 5-way head (= `train_lopo_cv.py --arch flat`) | — (reference) | establishes baseline distribution |
| 1 | V1 CNN branch + F-specialist head | B4, B3 | sentinel F1(213-F) ↑, F1-S/N/V not down |
| 2 | rhythm MLP + S-specialist head | B3, tier-4 | sentinel F1(232-S) ↑, others not down |
| 3 | conditional GRL record adversary | tier 2 (A3) | fold/seed variance ↓, N/V held |
| 4 | calibrated/balanced final 5-way loss | B1 | mean ↑, variance not worse |

**Evaluation**: sentinel screening fold first (`--mode sentinel`), then full
5-fold LOPO 2-seed only for rungs that pass. Protocol: `docs/LOPO_PROTOCOL.md`.
Final kept model also gets DS1/DS2 3-seed for the comparison table.

---

## 4. EXPERIMENT LOG (this strategy, 2026-08 onward)

Status: ✅ kept / ❌ rejected / 🔄 running / ⏸ built-not-run

| Date | Experiment | Setup | Result (DS2) | Verdict |
|---|---|---|---|---|
| 08-27 | **Multi-seed base audit** | compile all existing HMR vs DSTA-C runs, full recipe | HMR mean macro .584±.064 (N .962±.003, V .863±.010, **S .329±.122, F .182±.126**); DSTA-C mean .601±.032 but same-seed non-reproducible (.636→.573) | ✅ **HMR-BiLSTM is the base, DSTA-C is not** (DSTA can't hold S+F together, CP-decomp flat direction) |
| 08-27 | Root-cause diagnostics (CPU, no training) | linear probes on waveforms + official ckpt analysis | AUC_F 0.96 vs F1-F 0.24; 208-N/213-N AUC 1.000; V1 lifts transfer 0.90→0.96; val = 30 F / 3 patients | ✅ reframed problem (§2) |
| 08-28 | **Cross-DB Stage A, run 1** | MIT-BIH DS1 + INCART pooled, plain arch | DS2 macro .375, V collapsed | ❌ **FAILED** — INCART baseline-wander bug + recipe drift |
| 08-29 | **Cross-DB Stage A, v2** (fixed) | + 0.5 Hz Butterworth HP on INCART, per-source z-score, DS2 re-extracted (=byte-identical to paper), official recipe imported. 3 seeds. | macro .582±.031 (=off .584); **S .174±.002 — variance killed but mean COLLAPSED .33→.17**; **F .311±.106 — mean up +.13 but spread .259 not reduced**; best_ep still 5/14/5 | ❌ **Pooling rejected.** INCART S = different domain. Val-blindness hypothesis (tier 1b) largely disproven. F-mean lever noted for a later Stage-B oversample. |
| 08-29 | **HD ladder step 1 (`hier`)** | flat softmax → soft hierarchical head; loss kept = focal γ1.5 + class weights. 2 seeds (42,123). | macro .560 vs off .621; **S .261 vs .397; F .173 vs .255** — worse across the board | ❌ **Rejected.** Hierarchical head + big class weights + focal = double imbalance correction. |
| 08-29 | **HD ladder step 2 (`hier_la`)** | hier head + logit-adjust on [S,V,F,Q] sub-head only; focal OFF, weights OFF. seed 42 (killed 123 mid-run). | seed42: macro .496, S .129, **F .019** | ❌ **Rejected, worse.** Abn head had NO imbalance correction → abnormal underpredicted → F collapses. |
| 08-30 | **HD ladder step 2b (`hier_ba`)** | `HDLossBalanced`: prior-correct BOTH levels — BCE(abn, pos_weight≈8.4) + logit-adj sub CE. 2 seeds. | macro .539 vs off .621; S .218 vs .397; **F .097 vs .255** (F stable spread .007 but collapsed) | ❌ **Rejected.** 3/3 calibration variants fail. **Hierarchical head is DEAD** — decision head alone is not a big enough lever. Do NOT put the binary abn gate in HMR-MultiPath. |
| 08-30 | **LOPO sentinel `flat` baseline** (SWA) | plain RLSTMClassifier, sentinel fold, 25 ep, {S,V,F} selection, train-N cap 35k, 208+INCART-F in train | **all**: F .477 S .507 N .927 V .954 · **mitbih (213-F/232-S)**: F **.411** S **.163** · **incart**: F .622 S .939 | ✅ **reference set.** {SWA + {S,V,F}-selection + INCART-F in train + N-cap} ≈ **doubled F1-213-F vs DS1/DS2 (.24→.41)**. Residual hard problem = MIT-BIH 232-S (.16, INCART S .94 doesn't transfer) and pushing 213-F further. **[BENCHMARK] 197 min/fold → full 5-fold ~16.5 h/seed.** |
| 08-30 | **MultiPath rung 1 sentinel** (`--v1`, *replace* wiring) | F-specialist head **replaces** shared F logit | SWA mitbih F **.312** (flat .411), all F .367 (flat .477); S also down | ❌ **Rejected.** Narrow F-only path `[e_morph+e_v1]→1` underfits vs the shared head co-trained on abundant N/V gradient. `sentinel_vce_replaceWiring_REJECTED.json`. |
| 08-31 | **MultiPath rung 1b sentinel** (`--v1`, *residual*) | `z_F = shared_F + f_corr` (zero-init) | **best-ep**: mitbih F **.461** (flat .443) S **.178** (.032) macro .563 (.503); all F .502 (.474) macro .721 (.694). **SWA**: mitbih F .440 (.411), but all F −.009 / incart F −.081 | ✅ **PASSES sentinel** (best-ep improves 213-F + 232-S, N/V not down). V1-residual kept. SWA mixed — full 5-fold will decide. |
| 08-31 | **MultiPath rung 2 sentinel** (`--v1 --rhythm`) | + rhythm MLP → S-specialist (residual on z_S) | **best-ep**: mitbih F **.564** (flat .443, R1b .461) · S **.274** (.032) · macro **.626** (.503) ; all macro **.750** (.694). **SWA**: S huge (mitbih .16→.41) but F tanked (.44→.34) | ✅ **PASSES sentinel (best-epoch)** — big clean win on 213-F + 232-S. Rhythm S-specialist KEPT. SWA window was too wide (0.6E) → narrowed to 0.8E for rung 3+. |
| 08-31 | **MultiPath rung 3 sentinel** (`--v1 --rhythm --grl`) | + conditional GRL record adversary, λ 0→0.2 | best-ep mitbih F **.089** (R2 .564), V .718 (.847) | ❌ **Rejected.** GRL scrubs F-relevant morphology out of e_morph. De-confounding left to INCART diversity + LOPO holdout. |
| 08-31 | **MultiPath rung 3b sentinel** (`--v1 --rhythm --loss balanced`) | balanced-softmax on the 5-way head | best-ep mitbih S **.559** (R2 .274) but F **.285** (.564), N **.615** (.817) | ❌ **Rejected as a stack** — trades F+N for S. On record if framing shifts to "maximize S". |
| 08-31 | **AMP + parallel training** | mixed precision both trainers; 2 jobs concurrent (16 cores, RAM ok) | rung 3 + 3b ran in parallel, ~1.5× throughput | ✅ adopted |
| 08-29 | **LOPO harness + sentinel fold** | `preprocess_lopo.py` + `train_lopo_cv.py` (`--mode kfold|sentinel`, per-fold norm, inner-val selection on {S,V,F}, SWA, by-DB). Protocol locked in `docs/LOPO_PROTOCOL.md`. | — | ✅ **built** (helper tests pass); needs 1-fold GPU smoke + wall-time benchmark |
| — | HMR-MultiPath rung 0 (`flat` sentinel + LOPO) | plain trunk, reference distribution | — | ⏸ next after hier_ba |
| — | HMR-MultiPath rung 1 (V1 + F-specialist) | attacks B4/B3 | — | ⏸ not built |
| — | HMR-MultiPath rung 2 (rhythm MLP + S-specialist) | attacks B3 | — | ⏸ not built |
| — | HMR-MultiPath rung 3 (conditional GRL) | attacks A3 | — | ⏸ not built |
| — | HMR-MultiPath rung 4 (calibrated final head) | attacks B1 | — | ⏸ not built |

**Reference — official baseline (`data/processed/splits/`, full recipe), per seed:**
`42: macro .6279 N .9643 S .4335 V .8728 F .2411` ·
`123: macro .6137 N .9653 S .3600 V .8609 F .2688` ·
`456: macro .5109 N .9580 S .1940 V .8543 F .0371`

---

## 5. DEAD ENDS — do NOT re-try on the 1-channel MIT-BIH feature set

- Any aux loss on the flat 5-way softmax: confusion-margin, mixture-decomposition,
  prototype-repulsion, additive S-boost, One-vs-Rest. All tried, all lost.
- Generative F augmentation (VAE / GAN / diffusion) validated against existing
  train-F: hits the "patient-208 wall" (train-F is one patient, so synthetic F
  just interpolates 208).
- Naive multi-beat concatenation.
- Hierarchical cascade with a **hard** router (F misroutes 26%). (Soft/differentiable
  cascade is different and is being tried — HD ladder.)
- Both raw leads through the shared CNN: halves F1-S, 3-seed consistent.
- RR fed through the LSTM gate via z_t: β throttles it ~140×.
- Cross-DB **pooling** (INCART into train as-is): collapses DS2 S (Stage A v2).
- Threshold tuning alone on the flat head: F1-F caps at 0.296, macro drops.
- DSTA-C as the base: structurally can't hold S and F simultaneously.
- Hierarchical head + focal + large class weights (HD step 1): double-corrects.

---

## 6. KEY FILES & HOW TO RUN / READ

**Models**
- `hmr_bilstm.py` — `RLSTMClassifier` (the base), `RLSTMLoss` (focal + smoothness).
  Official checkpoint `results/checkpoints/inter_best_rlstm.pt` is `num_layers=2`;
  load `ck['model_state']`.
- `hmr_hd.py` — `RLSTMClassifierHD` (soft hierarchical head), `HDLoss`
  (focal OR logit-adjust). `python hmr_hd.py` = self-test.

**Training (all import the official recipe from `train_inter_patient.py` — cosine
LR, FGSM adv-training, epochs=45, patience=15)**
- `train_inter_patient.py` — the official single-run trainer. `ENSEMBLE_SEED` env var.
- `train_hd_ladder.py --variant {hier,hier_la} --seeds 42 123` — HD ladder steps 1-2.
  Per-epoch checkpoint/resume (`results/hd_ladder/checkpoints/`). `--summarize-only`
  prints the vs-official diff table.
- `train_crossdb_multiseed.py` — cross-DB Stage A (rejected, kept for reference).
- `validation/preprocess_aami_rr_crossdb.py` — v2 cross-DB preprocessing (INCART
  high-pass + per-source z-score). Output `data/processed/splits_rr_crossdb/`.

**Data**
- `data/processed/splits/` — official plain inter-patient split (window −90/+97,
  global train z-score). Used by `train_inter_patient` and `train_hd_ladder`.
- `data/processed/splits_rr/` — + 3 RR-ratio features per beat.
- `data/raw/mitdb/`, `data/raw/incartdb/` — raw WFDB records.

**Results**
- `results/crossdb_multiseed/` — Stage A (`seed_*.json`, `summary.json`,
  `_run1_failed_archive/`).
- `results/hd_ladder/` — HD ladder (`{variant}_seed_{s}.json`, `summary_{variant}.json`).
- `results/checkpoints/inter_best_rlstm.pt` — official model.
- `outputs/v1.0_20260819_053738/uncertainty/mc_selective_classification.json` —
  MC-Dropout selective classification (83% of F referred @ 90% coverage — the UQ
  referral story for the paper).

**Machine constraints**
- RTX 3050 Ti Laptop **4 GB GPU**, **13.9 GB RAM** (2×8GB, 2/2 slots, max 32).
  512 GB NVMe SSD (C/D/E = partitions of one disk).
- ~10–13 min/epoch on the LOPO pool (25 ep = ~3.3 h/sentinel-fold).
- **AMP (mixed precision) is on** in both LOPO trainers (`--amp 1` default) — frees
  VRAM headroom. Small sentinel jobs were run two-in-parallel earlier, but **P1
  runs ONE trainer at a time** (LOPO_PROTOCOL.md; full-fold jobs are heavier than
  sentinel). Keep ≥ 3 GB RAM free; enlarge the pagefile (48 GB on D:) as the OOM
  safety net.
- Machine reboots intermittently → all trainers checkpoint per epoch and resume
  (scaler state included, absent-key tolerant).
- Git Bash: use `ps -W` (not `tasklist`) to check if a Windows PID is alive.

---

## 7. REALISTIC TARGET (agreed with user)

- F1-F: **0.35–0.50 stable** (baseline 0.24; current swing 0.04–0.45).
- F1-S: **0.45–0.62 stable** (baseline 0.33; current swing 0.19–0.43).
- **Success = mean up AND seed variance down.** A higher single-seed mean with the
  same wild spread is NOT success (that is what ~2 weeks of experiments produced).
- Fusion beats have a real physiological ceiling — no method reaches 0.9. The
  residual uncertain F is handled by the MC-Dropout referral path, not the classifier.
