# Rare-class (F & S) R&D progress log

> **Purpose of this file.** A machine-readable lab notebook so any AI agent (Codex,
> a fresh Claude session, etc.) can pick up this project and understand: the
> problem, everything already tried, what was concluded, and the current step.
> Update the **CURRENT STEP** block and the **EXPERIMENT LOG** table after every run.
> Source of truth for individual run numbers = the JSON files under `results/`.

Last updated: **2026-08-29** (HD ladder step 2 running).

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
- Current strategy: **HMR-BiLSTM-HD** — a 4-part architecture change aimed at the
  measured causes (§3), run as an ablation ladder (§4).
- Realistic target: **F1-F 0.35–0.50, F1-S 0.45–0.62, STABLE** (mean up AND seed
  variance down). Not 0.6+. Fusion beats have a real Bayes-error ceiling.
- Fallback (user directive): if the HD ladder fails on HMR-BiLSTM, port the same
  ideas to an updated DSTA (`dsta_*.py`, `camp_net*.py`).

---

## 1. CURRENT STEP

**HD ladder step 2 (`hier_la`) — RUNNING** (launched 2026-08-29 ~15:00, `results/logs/hd_hier_la.log`).

- Script: `train_hd_ladder.py --variant hier_la --seeds 42 123`
- Change vs baseline: flat 5-way softmax → **soft hierarchical head**
  (`hmr_hd.py`) + **balanced-softmax / logit-adjustment** on the [S,V,F,Q]
  sub-head (focal OFF, class-weights OFF).
- Data: `data/processed/splits/` (the official plain inter-patient split — single
  variable = the head + loss).
- Question: does a *principled* prior correction pair with the hierarchical head
  better than the ad-hoc focal+weights (which failed, see step 1)?
- Read when done: `python train_hd_ladder.py --variant hier_la --seeds 42 123 --summarize-only`
- 2 seeds only (user: 2 seeds enough to conclude a step).

**Next if step 2 works** (mean held/up + variance not worse): build step 3
(dual-lead V1 branch). **Next if step 2 fails**: build step 4 (GRL de-confound)
standalone on the flat model, or pivot to DSTA.

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

## 3. Target architecture: HMR-BiLSTM-HD

Front end (CNN + BiRLSTM + attention pool) **unchanged**. Four additions, each a
ladder rung:

1. **Soft hierarchical head** (attacks tier 3). Replace flat softmax:
   `p_abn = σ(abn_head(e))`, `q = softmax(sub_head(e))` over [S,V,F,Q];
   `P(N) = 1 − p_abn`, `P(c) = p_abn · q[c]`. Differentiable, end-to-end, **no hard
   router** (the hard cascade misrouted 26% of F — dead end). Implemented in
   `hmr_hd.py::RLSTMClassifierHD`. Output = 5-way log-probs; `softmax(log_probs) ==
   probs`, so `train_inter_patient.evaluate()` works unchanged.
2. **Balanced-softmax / logit adjustment** (Menon et al. 2020) on the sub-head
   (attacks tier 3 + A6). `HDLoss(logit_adjust=τ·log prior)`. Replaces ad-hoc
   focal + class weights.
3. **Dual-lead V1 branch** (attacks tier 4). MLII → proven backbone unchanged;
   V1 → separate lightweight CNN; fuse at pooled embedding. **Not** both leads
   through the shared CNN (that halved F1-S, 3-seed confirmed). NOT built yet.
4. **Patient de-confounding** (attacks tier 2). Gradient-reversal patient/record
   classifier on the embedding, λ ramp 0→0.3 over 10 epochs. ± supervised
   contrastive. NOT built yet.

Plus a **rhythm MLP** on [pre_RR, post_RR, local_RR, post/pre] concatenated at the
embedding (attacks tier 4 / B3), and optional **multi-beat context** input.

**Evaluation target**: LOPO-CV over all F/S-bearing patients (MIT-BIH 208/213 +
INCART records + SVDB for S) → F1-F/F1-S mean ± std over held-out patients.
Still report DS1/DS2 3-seed for literature comparison. NOT built yet.

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
| 08-29 | **HD ladder step 2 (`hier_la`)** | soft hierarchical head + balanced-softmax/logit-adjust; focal OFF, weights OFF. 2 seeds (42,123). | — | 🔄 **RUNNING** |
| — | HD ladder step 3 (dual-lead V1) | + separate V1 CNN branch | — | ⏸ not built |
| — | HD ladder step 4 (GRL de-confound) | + gradient-reversal patient classifier | — | ⏸ not built |
| — | LOPO-CV evaluation harness | pooled MIT-BIH+INCART(+SVDB) patient-out CV | — | ⏸ not built |

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
- 4 GB GPU, 14 GB RAM. **One training job at a time.** ~11–15 min/epoch.
- Machine reboots intermittently → all trainers must checkpoint per epoch and resume.
- Git Bash: use `ps -W` (not `tasklist`) to check if a Windows PID is alive.

---

## 7. REALISTIC TARGET (agreed with user)

- F1-F: **0.35–0.50 stable** (baseline 0.24; current swing 0.04–0.45).
- F1-S: **0.45–0.62 stable** (baseline 0.33; current swing 0.19–0.43).
- **Success = mean up AND seed variance down.** A higher single-seed mean with the
  same wild spread is NOT success (that is what ~2 weeks of experiments produced).
- Fusion beats have a real physiological ceiling — no method reaches 0.9. The
  residual uncertain F is handled by the MC-Dropout referral path, not the classifier.
