# Final Publication Readiness Report — HMR-BiLSTM

**Project:** HMR-BiLSTM — Hybrid Memory-Residual Bidirectional LSTM for ECG Arrhythmia Classification  
**Source documents:** `SCIENTIFIC_AUDIT_REPORT.md` (154 issues), `CRITICAL_FIXES_SUMMARY.md`, `REMAINING_ISSUES.md`  
**Date:** 2026-06-23  
**Scope:** Classification and prioritisation of all 154 audit issues; final publication verdict.

---

## Classification Key

| Code | Category | Definition |
|---|---|---|
| **CB** | Code Bug | Deterministic coding error fixable by editing source; no new training run required |
| **EP** | Experimental Protocol | Experimental design flaw; requires redesigning and re-running experiments (may include retraining) |
| **RI** | Reporting Inconsistency | Results are mislabelled, mis-averaged, or inconsistently presented; underlying data may or may not be valid |
| **FP** | False Positive | Issue does not exist in the actual file; auditor artefact |
| **RT** | Requires Retraining | Fixing the code changes model weights, training data, or checkpoint selection; published numbers cannot be maintained without a new training run |
| **FX** | Fixed | Already resolved in the critical fix pass |

**Effort scale:** Trivial (<30 min) · Low (30 min–4 h) · Medium (1–3 days) · High (1–2 weeks) · Very High (>2 weeks)  
**Publication risk:** None · Low · Medium · High · Critical

---

## Aggregate Summary

| Category | CRITICAL | HIGH | MEDIUM | LOW | Total |
|---|---|---|---|---|---|
| Fixed (FX) | 6 | 0 | 2 | 0 | **8** |
| False Positive (FP) | 8 | 0 | 0 | 0 | **8** |
| Code Bug (CB) | 4 | 22 | 35 | 13 | **74** |
| Experimental Protocol (EP) | 5 | 8 | 6 | 0 | **19** |
| Reporting Inconsistency (RI) | 4 | 10 | 11 | 2 | **27** |
| Requires Retraining (RT) | 1 | 15 | 3 | 0 | **19** |
| **Total** | **28** | **55** | **57** | **15** | **154** |

| Fixability (NOT FIXED only) | Count |
|---|---|
| Can fix without retraining (CB + RI where re-run ≠ retrain) | **81** |
| Cannot fix without retraining (RT + EP requiring new training) | **57** |
| **Total remaining** | **138** |

---

## CRITICAL Issues — Full Classification (28 issues)

| # | Description | Category | Fix w/o Retrain? | Effort | Pub Risk |
|---|---|---|---|---|---|
| 1 | Duplicate `for t in range(T)` in hmr_bilstm.py — import fails | FP | N/A | — | None |
| 2 | Line 108 appears twice in hmr_bilstm_ablation.py — file corrupt | FP | N/A | — | None |
| 3 | `RLSTMLoss.__init__` extra indent — ImportError in ablation | FP | N/A | — | None |
| 4 | `x.to(device)` outside loop in temperature_scaling.py | FP | N/A | — | None |
| 5 | Intra-patient split: beats from same patient in train and test | EP | NO | Very High | **Critical** |
| 6 | Baselines on inter-patient split; HMR-BiLSTM on intra-patient | EP | NO | Very High | **Critical** |
| 7 | PGD-F1 (0.84) > Clean-F1 (0.56) in table5 — physically impossible | RI | NO¹ | High | **Critical** |
| 8 | Two evaluation regimes (F1=0.56 vs F1=0.88) silently mixed | RI | NO¹ | Very High | **Critical** |
| 9 | Tolerance-aware Jaccard double-counts matched elements | FX | — | — | — |
| 10 | `accuracy_in_bin` outside `if` body — evaluate_calibration.py | FP | N/A | — | None |
| 11 | PGD attack passes `r_fwd=None` — incomplete loss in pgd_convergence | CB | YES | Low | High |
| 12 | AutoAttack/CW use n=200 pre-filtered subset; clean F1=1.0 baseline | EP | YES² | Medium | **Critical** |
| 13 | 8 packages missing from requirements.txt | FX | — | — | — |
| 14 | Inter-patient splits never created by orchestration scripts | CB | YES | Low | High |
| 15 | Trailing semicolons in mc_dropout.py — possible SyntaxError | FP | N/A | — | None |
| 16 | Duplicate line numbers in mc_dropout.py, deep_ensemble.py | FP | N/A | — | None |
| 17 | Single-checkpoint Deep Ensemble; weight perturbation unimplemented; MI=0 | RT | NO | High | **Critical** |
| 18 | FGSM missing data-range clamp | FX | — | — | — |
| 19 | `cw_c=1e-4` effectively disables C&W attack | FX | — | — | — |
| 20 | Noise robustness evaluated on intra-patient test.npz | FX | — | — | — |
| 21 | 4-class vs 5-class macro F1 switches silently across all tables | RI | YES | Medium | High |
| 22 | SHAP CLASS_NAMES omits Q; global importance averages 4 classes | FX | — | — | — |
| 23 | Intra-patient splits saved un-normalized; inter-patient normalized | RT | NO | High | High |
| 24 | Duplicate `return` in temporal_smoothness_loss | FP | N/A | — | None |
| 25 | Demo section line collision in hmr_bilstm.py — smoke-test broken | CB | YES | Trivial | Low |
| 26 | PGD vs AutoAttack epsilon spaces not verified equivalent | CB | YES | Low | High |
| 27 | Global scalar normalization instead of per-feature in preprocess.py | RT | NO | Very High | Medium |
| 28 | AutoAttack/CW F1 drops not comparable to FGSM/PGD (different baselines) | RI | NO¹ | High | **Critical** |

¹ These are downstream consequences of #5/#6/#12 — fixable only after re-running evaluations on a unified split.  
² Requires re-running attack scripts on the full test set, not a training change.

---

## HIGH Issues — Full Classification (55 issues)

| # | Description | Category | Fix w/o Retrain? | Effort | Pub Risk |
|---|---|---|---|---|---|
| 29 | `self.dropout` in RLSTMCell defined but never called — effective dropout=0 | RT | NO | High | High |
| 30 | Substring name match for bias init — fragile on architecture extension | CB | YES | Trivial | Low |
| 31 | `requires_grad=True` leaf disconnected from graph in smoothness loss | CB | YES | Trivial | Low |
| 32 | `c_keep + c_add` does not decompose `c_lstm` — math claim incorrect | RI | YES | Low | High |
| 33 | `F.layer_norm` on near-zero `c_add` — degenerate attention scoring | RT | NO | High | Medium |
| 34 | LayerNorm gamma can saturate `r_t` gate | RT | NO | High | Medium |
| 35 | `h_T`/`c_T` computed but silently unused — dead code | CB | YES | Trivial | Low |
| 36 | `model.zero_grad()` before `loss.backward()` during FGSM training | RT | NO | Medium | Medium |
| 37 | NaN batch skipped silently; no dropped-batch counter | CB | YES | Trivial | Low |
| 38 | `train.py` uses 5-class F1; `train_inter_patient.py` uses 4-class for early stopping | RT | NO | Medium | High |
| 39 | AUC: un-renormalized `probs[:,:4]` passed to `roc_auc_score` | CB | YES | Low | High |
| 40 | Focal loss deviates from Lin et al. (2017) — alpha after focal multiplier | EP | NO | High | Medium |
| 41 | Baselines: 12 epochs, no LR schedule vs HMR-BiLSTM: 45 epochs, cosine LR, FGSM | EP | NO | Very High | **Critical** |
| 42 | `int(fs)` truncation for resampling; float `fs` for annotation index | RT | NO | High | Medium |
| 43 | Boundary guard uses `<` instead of `<=` — drops valid beats | RT | NO | High | Medium |
| 44 | 10-bin ECE in evaluate_calibration.py vs 15-bin in calibration_metrics.py | CB | YES | Trivial | Medium |
| 45 | Temperature scaling never applied before ECE computation | CB | YES | Low | High |
| 46 | Fixed equal-width bins unsuitable for class-imbalanced MIT-BIH | CB | YES | Low | Medium |
| 47 | `TemperatureScaling.fit()` has no `.to(device)` guard | CB | YES | Trivial | Low |
| 48 | Pre- and post-calibration ECE mixed in reporting without labelling | RI | YES | Low | High |
| 49 | `attn_weights` aligned to T/4 downsampled axis, not original ECG timesteps | CB | YES | Low | Medium |
| 50 | `enable_mc_dropout` misses `Dropout2d`/`Dropout3d`/`AlphaDropout` | CB | YES | Trivial | Low |
| 51 | Baseline wander frequency formula uses hardcoded 187 — not true Hz | CB | YES | Low | Medium |
| 52 | MI clipped without diagnostic; asymmetric eps biases MI upward | CB | YES | Low | Medium |
| 53 | Synthetic OOD generated from ID test set — AUROC label misleading | EP | YES | Medium | High |
| 54 | MC Dropout MI and Ensemble MI share function with undocumented semantic difference | RI | YES | Low | Medium |
| 55 | 4-class F1 (`labels=[0,1,2,3]`) in trustworthiness inconsistent with 5-class elsewhere | RI | YES | Low | High |
| 56 | `abs(eps) - 0.02 < 0.001` selects wrong epsilon row in T8 scorecard | CB | YES | Trivial | High |
| 57 | `n_steps=50` insufficient for MaxPool+ReLU+attention IG convergence | EP | YES | Medium | Medium |
| 58 | Zero baseline non-neutral after BatchNorm running stats | EP | YES | Medium | Medium |
| 59 | `mean_abs_attribution_plot` computed as signed mean — IG plots wrong | CB | YES | Trivial | High |
| 60 | Cosine gradient similarity mislabelled as TracIn; single checkpoint | RI | YES | Low | High |
| 61 | No `model.eval()` assertion in data attribution gradient loop | CB | YES | Trivial | Low |
| 62 | Unstratified SHAP background dominated by N-class beats | EP | YES | Low | Medium |
| 63 | Ensemble OOD-AUROC (0.6228) < MC Dropout (0.6341) — implausible | RI | NO | High | High |
| 64 | `auc_ovr=NaN` in all validation entries for ensemble seed models | RT | NO | High | High |
| 65 | No-Adv model outperforms full model on inter-patient split by +0.18 F1 | EP | NO | Very High | **Critical** |
| 66 | Final hyperparameters not supported by documented tuning sweep | RI | YES | Low | High |
| 67 | Batch-level fallback bounds for PGD clamping — not globally consistent | CB | YES | Low | Medium |
| 68 | Duplicate for-loop in auto_attack.py:131–134 — dead overhead | CB | YES | Trivial | Low |
| 69 | `d_min`/`d_max` from subset, not full test set — epsilon budget inflated | CB | YES | Low | Medium |
| 70 | Fragile 4D→3D shape routing in `AAWrapper` — silent permutation risk | CB | YES | Medium | Medium |
| 71 | `step_accs[0]` always 1.0 by construction — misleading baseline | CB | YES | Low | Medium |
| 72 | Intra/inter models compared on different-scale inputs | EP | NO | Very High | High |
| 73 | Intra-patient `class_weights.npy` used for inter-patient baseline training | RT | NO | High | Medium |
| 74 | `y_pred` computed on train data then immediately overwritten — dead code | CB | YES | Trivial | Low |
| 75 | MCE increases after temperature scaling (0.238→0.369) — minority classes worsen | RI | YES | Medium | Medium |
| 76 | ECE reported as 0.0391, 0.2309, and 0.0397 in different tables | RI | YES | Medium | High |
| 77 | Private `torchattacks._autoattack` access — gradient masking detection broken | CB | YES | Medium | High |
| 78 | `test_shape.py` tests Mock model, not `RLSTMClassifier`; calls `.cuda()` unconditionally | CB | YES | Low | Low |
| 79 | 5 diagnostic scripts hardcode run-ID that only exists on author's machine | CB | YES | Low | Low |
| 80 | `verify_gradients.py` tests wrong model flags — does not replicate training | CB | YES | Low | Low |
| 81 | Stale full-model checkpoint used if retrained after ablation | CB | YES | Low | Medium |
| 82 | FGSM uses `RLSTMLoss` for LSTM/BiLSTM trained with `CrossEntropyLoss` | CB | YES | Low | High |
| 83 | 4-class vs 5-class macro F1/AUC mixed in `final_results.csv` | RI | YES | Low | High |

---

## MEDIUM Issues — Full Classification (56 issues; 2 already fixed)

| # | Description | Category | Fix w/o Retrain? | Effort | Pub Risk |
|---|---|---|---|---|---|
| 84 | Alpha (B,1) vs beta (B,H) asymmetry undocumented | RI | YES | Low | Low |
| 85 | Detached dict values in `RLSTMLoss` return — silent trap | CB | YES | Trivial | Low |
| 86 | `no_rmc` variant gets zero smoothness loss — unfair ablation | EP | NO | High | High |
| 87 | Commented-out `use_interaction` attribute — dead code | CB | YES | Trivial | Low |
| 88 | Redundant substring match for orthogonal init | CB | YES | Trivial | Low |
| 89 | Cosine LR off-by-one — never reaches `min_lr` at final epoch | RT | NO | Medium | Low |
| 90 | Seed-42 model excluded from ensemble (only 2 of 3 seeds used) | RT | NO | High | Medium |
| 91 | Shared DataLoader across ablation variants; shuffle state not reset | EP | NO | High | Medium |
| 92 | Ablation intra uses 5-class F1; ablation inter uses 4-class | RI | YES | Low | Medium |
| 93 | Baselines save raw `state_dict`; HMR-BiLSTM saves wrapper dict | CB | YES | Low | Low |
| 94 | `best_f1=0.0` init may prevent checkpoint write at epoch 1 | RT | NO | Low | Medium |
| 95 | 5-class AUC silently returns 0.0 if Q-class absent | CB | YES | Low | Medium |
| 96 | Unconditional `torch.load` without checking file exists | CB | YES | Trivial | Low |
| 97 | `steps` loop variable overwritten by list comprehension | CB | YES | Trivial | Medium |
| 98 | Per-class rounding: total samples may not equal `subset_size` | CB | YES | Trivial | Low |
| 99 | List of dicts saved as numpy object array — requires `allow_pickle` | CB | YES | Low | Low |
| 100 | Random shuffle instead of stratified split for intra-patient set | EP | NO | High | Medium |
| 101 | Class weight cap 10.0 severely under-weights minority classes (true ~59) | RT | NO | High | Medium |
| 102 | Algebraically weak leakage check; `leakage_detected` is dead code | CB | YES | Low | Low |
| 103 | Threshold check does not work for per-feature normalization | CB | YES | Low | Low |
| 104 | Conditional ECE weights by `n_c` not `N` — incomparable to global ECE | CB | YES | Low | Medium |
| 105 | Clipping entire prob matrix breaks simplex constraint | CB | YES | Low | Medium |
| 106 | No LBFGS convergence diagnostic in temperature scaling | CB | YES | Trivial | Low |
| 107 | Degenerate confidences produce silent `ece=0.0` | CB | YES | Trivial | Low |
| 108 | `results/tables/` directory never created — `FileNotFoundError` | CB | YES | Trivial | Low |
| 109 | Dead `model.eval()` at line 315; no guard between MC Dropout passes | CB | YES | Low | Medium |
| 110 | Bar width=0.08 in calibration bar chart; plot ignores MC statistics | CB | YES | Trivial | Low |
| 111 | `std_max = probs.max(axis=2).std()` ignores class identity | CB | YES | Low | Medium |
| 112 | JSON key `ood_detection_auroc` persists despite rename to `corruption_detection` | CB | YES | Trivial | Low |
| 113 | Gaussian noise sigma not referenced to signal power; SNR not reported | RI | YES | Low | Low |
| 114 | `np.roll` axis undocumented; fragile if data format changes | CB | YES | Trivial | Low |
| 115 | `shap_background_samples` silently overridden to 100 without warning | CB | YES | Trivial | Low |
| 116 | `ModelWrapper` returns logits not probabilities — logit-SHAP not calibrated | CB | YES | Low | Medium |
| 117 | Global SHAP importance uses different sample set than Jaccard samples | EP | YES | Low | Low |
| 118 | IG convergence delta collected but never used to filter unreliable attributions | CB | YES | Low | Low |
| 119 | Confidence threshold on model trained on noisy data — memorized noise invisible | EP | YES | Medium | Medium |
| 120 | Run ID hardcoded in plot_disagreements.py — silently uses stale results | CB | YES | Trivial | Low |
| 121 | Full explanation set as one batch — OOM risk on low-memory CPU | CB | YES | Low | Low |
| 122 | CLASS_NAMES has 4 entries; class-4 displays as `None` | FX | — | — | — |
| 123 | Mixed old/new NumPy RNG causes hidden global state mutation | CB | YES | Low | Low |
| 124 | F1 drop units inconsistent: absolute vs percentage across tables | RI | YES | Low | Medium |
| 125 | Fusion (F) class recall silently omitted from FGSM robustness table | RI | YES | Low | Medium |
| 126 | table5_consolidated.csv cannot be regenerated; LSTM/BiLSTM PGD provenance unknown | RI | YES | High | **Critical** |
| 127 | `evaluate_autoattack.py` missing from Python orchestrator | CB | YES | Trivial | Medium |
| 128 | T8 reads `fgsm_comparison_results.json` that is deleted by compare script | CB | YES | Low | Medium |
| 129 | `evaluate_calibration` hardcodes 10 bins vs config's 15 | CB | YES | Trivial | Medium |
| 130 | `evaluate_calibration.py` never writes to `outputs/<run_id>/calibration/results.json` | CB | YES | Low | Medium |
| 131 | `dict\|None` type hint — crashes on Python 3.9 | CB | YES | Trivial | Low |
| 132 | `plot_case_visualization.py` uses non-inter-patient model — visualizations mismatch metrics | CB | YES | Trivial | Low |
| 133 | `evaluate_robustness_all.py` uses intra-patient splits | FX | — | — | — |
| 134 | `test_speed.py` permanently mutates global `CONFIG` dict | CB | YES | Trivial | Low |
| 135 | Double-append to `skipped` list in diag_verify_coupling_v2.py | CB | YES | Trivial | Low |
| 136 | No-Adv outperforms full model by +0.18 F1 on inter-patient split | RI | NO | Very High | **Critical** |
| 137 | Two-config hyperparameter grid; neither config matches final model | RI | YES | Low | High |
| 138 | Full model clean F1 differs by 0.012 between ablation_robustness.csv and ablation_table | RI | YES | Low | Low |
| 139 | PGD convergence: steps 50 and 100 bit-identical to step 10 — likely copy-paste | RI | YES | Medium | High |

---

## LOW Issues — Full Classification (15 issues)

| # | Description | Category | Fix w/o Retrain? | Effort | Pub Risk |
|---|---|---|---|---|---|
| 140 | Second unreachable `return` in `temporal_smoothness_loss` | CB | YES | Trivial | Low |
| 141 | Mutable `last_*` state on RLSTMCell — not thread-safe, incompatible with `torch.compile` | CB | YES | Low | Low |
| 142 | `W_beta` always allocated even when unused — parameter count misleading | CB | YES | Low | Low |
| 143 | Cosine LR off-by-one in train.py:84–86 (minor final-epoch difference) | RT | NO | Low | Low |
| 144 | `beta` documented as scalar but is vector (B, H) — paper may be incorrect | RI | YES | Low | Low |
| 145 | Mixed tab/space indentation in preprocess.py:94 | CB | YES | Trivial | Low |
| 146 | `std` epsilon added after `std()` not inside `sqrt` | CB | YES | Trivial | Low |
| 147 | `data_leakage_prevented` flag does not check test set | CB | YES | Trivial | Low |
| 148 | `print_dist` divides by `len(y)` without zero-check | CB | YES | Trivial | Low |
| 149 | Duplicate `return out` in `compute_conditional_ece` — dead code | CB | YES | Trivial | Low |
| 150 | Empty-bin phantom rows in reliability diagram CSV | CB | YES | Trivial | Low |
| 151 | `dim=-1` vs `dim=1` in softmax across calibration modules — cosmetic inconsistency | CB | YES | Trivial | Low |
| 152 | `import copy` unused in deep_ensemble.py — unimplemented feature relic | CB | YES | Trivial | Low |
| 153 | `label_accuracy` and `accuracy` are duplicate keys in FGSM result dict | CB | YES | Trivial | Low |
| 154 | `gen_guide.py` references `class_weights.json`; actual file is `class_weights.npy` | CB | YES | Trivial | Low |

---

## Consolidated Statistics

### By Category (all 154 issues)

| Category | Count | % |
|---|---|---|
| Fixed (FX) | 8 | 5.2 % |
| False Positive (FP) | 8 | 5.2 % |
| Code Bug (CB) | 74 | 48.1 % |
| Experimental Protocol (EP) | 19 | 12.3 % |
| Reporting Inconsistency (RI) | 27 | 17.5 % |
| Requires Retraining (RT) | 18 | 11.7 % |

### Fixability of the 138 NOT FIXED issues

| Can fix without retraining? | Count | Notes |
|---|---|---|
| YES | 81 | CB + RI not requiring new training + EP where only evaluation re-runs needed |
| NO | 57 | RT (18) + EP requiring training redesign (16) + RI downstream of bad data (23) |

### NOT FIXED issues by severity and fixability

| Severity | Total NOT FIXED | Fix w/o Retrain | Need Retrain/Redesign |
|---|---|---|---|
| CRITICAL | 14 | 4 | 10 |
| HIGH | 55 | 36 | 19 |
| MEDIUM | 54 | 42 | 12 |
| LOW | 15 | 14 | 1 |
| **Total** | **138** | **96** | **42** |

### CRITICAL issues that can be fixed without retraining

These 4 issues are code-only fixes that immediately reduce CRITICAL risk:

| # | Fix Required | Est. Effort |
|---|---|---|
| 11 | Add `r_fwd`/`r_bwd` to PGD attack loss in `pgd_convergence.py:64` | Low |
| 14 | Add `preprocess_aami.py` + `train_inter_patient.py` to both orchestration scripts | Low |
| 25 | Verify and repair demo section in `hmr_bilstm.py:436` | Trivial |
| 26 | Add epsilon-space assertion in `auto_attack.py` | Low |

---

## Blocking Items for Publication

The following issues must be resolved before any submission. They are grouped by the
underlying root cause to avoid redundant re-runs.

### BLOCK A — Unified evaluation split (blocks: #5, #6, #7, #8, #21, #28, #83)

**Root cause:** The primary model (`train.py`) was evaluated on the intra-patient split
while all baselines and trustworthiness steps use the inter-patient split. Every
cross-model comparison table is confounded.

**Required action:**
1. Designate `train_inter_patient.py` + `inter_best_rlstm.pt` as the sole primary model.
2. Retire `train.py` / `best_rlstm.pt` results from all paper tables.
3. Retrain all baselines (LR, DT, LSTM, BiLSTM) on the inter-patient split with the
   same epoch budget and LR schedule as HMR-BiLSTM (also resolves #41).
4. Regenerate every table using a single macro-F1 averaging convention (5-class
   including Q, or 4-class per AAMI EC57 — pick one and document it).

**Effort:** Very High | **Risk if unresolved:** Critical (paper rejected)

---

### BLOCK B — Adversarial evaluation protocol (blocks: #12, #17, #28, #41, #65)

**Root cause:** AutoAttack and CW use a trivially-biased n=200 pre-correct subset
(clean F1=1.0); the Deep Ensemble is a single-model run (MI=0); No-Adv outperforms
Full by +0.18 F1 (unexplained); baselines are under-trained compared to HMR-BiLSTM.

**Required action:**
1. Re-run AutoAttack and CW on the full inter-patient test set (or stratified ≥1,000
   samples/class). Report F1 drop relative to the same-sample clean baseline.
2. Train at least 2 additional independent ensemble seeds (123, 456 already exist; need
   seed 42 and ideally a 3rd) and verify ensemble MI > 0 before reporting.
3. Investigate why No-Adv > Full model; likely a split mismatch bug — resolve after
   Block A is complete.
4. Standardize baseline training to the same epoch budget and scheduler as HMR-BiLSTM.

**Effort:** Very High | **Risk if unresolved:** Critical

---

### BLOCK C — Table provenance (blocks: #7, #8, #126, #136, #139)

**Root cause:** `table5_consolidated.csv` and `ablation_results_inter.json` contain
values that cannot be reproduced from the current codebase, including physically
impossible comparisons (adversarial F1 > clean F1) and likely copy-paste results in
`pgd_convergence_results.json`.

**Required action:**
1. Delete all pre-computed result CSV and JSON files from `results/tables/` and
   `outputs/`.
2. After completing Blocks A and B, regenerate all tables from scratch using a single
   pipeline run with the unified evaluation configuration.
3. Verify that PGD convergence results show monotonically decreasing accuracy with
   increasing steps.

**Effort:** High | **Risk if unresolved:** Critical (data fabrication appearance)

---

### BLOCK D — Model training correctness (blocks: #29, #36, #38, #40, #42, #43)

**Root cause:** Several training-time bugs affect the model actually trained: dropout
is never applied inside RLSTMCell, `model.zero_grad()` is in the wrong position during
adversarial training, early stopping uses different F1 class counts between training
scripts, and the focal loss deviates undisclosed from Lin et al. (2017).

**Required action:**
1. Fix `evaluate_fgsm.py`-style training: move `model.zero_grad()` after
   `loss.backward()` in `train.py`.
2. Either apply `self.dropout` in `RLSTMCell.forward` or explicitly state in the paper
   that within-cell dropout=0 was used.
3. Unify early stopping F1 class count across `train.py` and `train_inter_patient.py`.
4. Describe the focal loss as "modified focal loss" in the paper (alpha applied after
   focal term) rather than citing Lin et al. (2017) as the direct source.
5. After fixing items 1–4, retrain the primary model.

**Effort:** High | **Risk if unresolved:** High

---

### BLOCK E — Explainability validity (blocks: #59, #60, #32, #57, #58)

**Root cause:** IG visualisation plots show signed mean attribution (cancelling positive
and negative contributions); the method labelled "TracIn" is gradient cosine similarity;
the mathematical claim that `c_keep + c_add` decomposes `c_lstm` is incorrect.

**Required action:**
1. Fix `mean_abs_attribution_plot` to use `torch.abs()` before mean — one-line change,
   no retraining. Regenerate all IG figures.
2. Rename "TracIn" to "gradient cosine similarity (single-checkpoint approximation)"
   in code, paper, and all result JSONs.
3. Correct the mathematical description of the RMC mechanism in the paper (it is a
   competing memory path, not a partition of the LSTM update).
4. Increase IG steps to at least 200 and switch to a neutral baseline (e.g., mean of
   training set over a moving window, post-BN adaptation).

**Effort:** Medium | **Risk if unresolved:** High

---

### BLOCK F — Calibration reporting (blocks: #44, #45, #48, #76)

**Root cause:** ECE is never computed on calibrated logits (temperature scaling not
applied before ECE in `evaluate_calibration.py`); bin count inconsistency (10 vs 15)
between scripts; pre/post-calibration ECE presented without labelling.

**Required action:**
1. Apply temperature scaling before computing ECE — single call to `temp_model(logits)`.
2. Standardize to 15-bin ECE across all scripts.
3. Label all ECE values as "pre-calibration" or "post-calibration" in tables.
4. Investigate why MCE worsens after temperature scaling (#75) — if temperature scaling
   genuinely hurts minority class calibration, state this in the paper as a limitation.

**Effort:** Medium | **Risk if unresolved:** High

---

## Issues Fixable in One Session (Code-Only, No Experiments)

These 41 HIGH or MEDIUM code bugs can be patched in source code without running any
experiments. They are safe to fix in any order after the blocking items above.

| Category | Issue IDs |
|---|---|
| One-line fixes | #31, #35, #37, #44, #47, #50, #56, #59, #61, #68, #74, #85, #87, #88, #96, #97, #98, #106, #107, #108, #110, #112, #114, #115, #127, #128, #129, #131, #132, #134, #135 |
| Short fixes (<1 day) | #11, #26, #39, #45, #46, #49, #51, #52, #67, #69, #71, #79, #80, #81, #82, #93, #95, #99, #104, #105, #109, #111, #116, #118, #120, #121, #123, #130 |

---

## Final Publication Readiness Verdict

### Overall Status: **NOT READY FOR PUBLICATION**

The codebase has been partially remediated (8 critical fixes applied, 8 false positives
eliminated), but 138 issues remain, including 14 that are CRITICAL severity. The most
fundamental problems — data leakage, mixed evaluation regimes, and physically impossible
table values — require experiment redesign and cannot be addressed by code edits alone.

### Readiness by Subsystem

| Subsystem | Status | Blocking Issues |
|---|---|---|
| Core model (hmr_bilstm.py) | ⚠ PARTIAL | #29, #32, #33, #34 (retrain after fixes) |
| Data pipeline | ✗ BLOCKED | #5, #6, #23, #27 — split mismatch, normalization |
| Training pipeline | ✗ BLOCKED | #29, #36, #38, #40, #41, #42, #43 — multiple training bugs |
| Robustness evaluation | ⚠ PARTIAL | #12, #17, #28, #41, #65 — attack protocol problems |
| Calibration | ⚠ PARTIAL | #45, #48, #76 — fixable without retraining |
| Uncertainty quantification | ✗ BLOCKED | #17 — ensemble is a single-model run |
| Explainability | ⚠ PARTIAL | #59, #60 — fixable; #57, #58 need IG re-run |
| Results tables | ✗ BLOCKED | #7, #8, #126, #136 — impossible values, unknown provenance |
| Reproducibility | ⚠ PARTIAL | #14, #79 — fixable; orchestration scripts incomplete |

### Estimated work remaining to reach publication readiness

| Work Package | Effort | Issues Resolved |
|---|---|---|
| Code-only fixes (no experiments) | 2–3 days | ~41 issues |
| Unified data pipeline + preprocessing | 1 week | #5, #6, #23, #27, #42, #43, #100 |
| Baseline retraining (fair conditions) | 1–2 weeks | #41, #73, #86 |
| Primary model retraining (after training fixes) | 3–5 days | #29, #33, #36, #38, #40 |
| Ensemble retraining (3 independent seeds) | 3–5 days | #17, #64, #90 |
| Adversarial evaluation re-run (full test set) | 2–3 days | #12, #26, #28, #65, #67, #69 |
| IG/SHAP re-run (after code fixes) | 1–2 days | #57, #58, #59, #62 |
| Calibration re-run | 1 day | #45, #48, #76 |
| Table regeneration (unified pipeline) | 2 days | #7, #8, #21, #55, #76, #83, #126, #139 |
| Paper revision (method descriptions) | 3–5 days | #32, #40, #60, #66, #84, #144 |
| **Total estimated** | **~8–12 weeks** | **~120 of 138 open issues** |

### Minimum viable submission path

If resources allow only the highest-priority work before submission:

1. **Week 1:** Apply all code-only fixes (blocks E, F, and the 41 one-session fixes).
2. **Week 2–3:** Unify data pipeline onto inter-patient split; retire intra-patient results from all tables.
3. **Week 4–5:** Retrain baselines with fair settings on the unified split.
4. **Week 6:** Re-run all evaluation scripts (robustness, calibration, explainability) on the unified inter-patient test set.
5. **Week 7:** Retrain ensemble (3 seeds); verify MI > 0.
6. **Week 8:** Regenerate all tables; verify no physically impossible values remain.
7. **Week 9–10:** Address model training bugs (#29, #36, #38); retrain primary model; re-run full evaluation once more.
8. **Week 11–12:** Paper revision and final review.

Issues that may remain deferred until revision (lower risk): #27 (normalization axis), #33/#34 (architecture stability), #57/#58 (IG baseline), #91 (ablation DataLoader seeding), #42/#43 (preprocessing off-by-ones).

---

*Generated: 2026-06-23*  
*Total issues: 154 — 8 Fixed · 8 False Positive · 74 Code Bugs · 19 Experimental Protocol · 27 Reporting Inconsistency · 18 Requires Retraining*  
*Remaining unresolved: 138 — 81 fixable without retraining · 57 requiring experiment redesign or retraining*
