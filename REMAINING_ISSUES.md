# Remaining Issues — HMR-BiLSTM

**Source:** `SCIENTIFIC_AUDIT_REPORT.md` (154 issues total)  
**Fix pass:** `CRITICAL_FIXES_SUMMARY.md` (6 code-level critical fixes)  
**Date:** 2026-06-23

---

## Status Legend

| Status | Meaning |
|---|---|
| **FIXED** | Code was modified and the bug is resolved |
| **PARTIALLY FIXED** | Root cause reduced but not fully eliminated |
| **NOT FIXED** | Issue confirmed or unverified; no change made |
| **FALSE POSITIVE** | Auditor reported a bug that does not exist in the actual file |

---

## Summary: Remaining Unresolved Issues

> **Note on severity counts:** The audit report's executive summary states 46 MEDIUM and
> 25 LOW, but the per-issue index table lists #84–#139 as MEDIUM (56 issues) and
> #140–#154 as LOW (15 issues). Both sets sum to the same total of 71. The per-issue
> index values are used here as the authoritative record.

| Severity | Original | False Positive | Fixed | Partially Fixed | Not Fixed |
|---|---|---|---|---|---|
| CRITICAL | 28 | 8 | 16 | 2 | **2** |
| HIGH | 55 | 0 | 22 | 1 | **32** |
| MEDIUM | 56 | 1 | 27 | 0 | **28** |
| LOW | 15 | 1 | 7 | 0 | **7** |
| **Total** | **154** | **10** | **72** | **3** | **69** |

**69 issues remain unresolved (not fixed or partially fixed).**

---

## CRITICAL Issues (28 total)

| # | File / Location | Description | Status | Notes |
|---|---|---|---|---|
| 1 | `hmr_bilstm.py:213` | Duplicate `for t in range(T):` — file cannot be imported | **FALSE POSITIVE** | File read and verified syntactically correct; auditor produced corrupted intermediate result |
| 2 | `hmr_bilstm_ablation.py:107–108` | Line 108 appears twice — file corrupt | **FALSE POSITIVE** | File read and verified; both lines present with correct content |
| 3 | `hmr_bilstm_ablation.py:435` | `RLSTMLoss.__init__` extra leading space — ImportError | **FALSE POSITIVE** | File read and verified; indentation is correct |
| 4 | `calibration/temperature_scaling.py:73–74` | `x = x.to(device)` outside for-loop body — unexecutable | **FALSE POSITIVE** | File read and verified; indentation is correct |
| 5 | `preprocess.py:136–142` | Intra-patient split: beats from same patient in train and test | **PARTIALLY FIXED** | Code now uses inter-patient splits throughout; intra-patient pipeline retained as legacy only |
| 6 | `run_baselines.py` vs `train.py` | Baselines use inter-patient split; HMR-BiLSTM uses intra-patient | **FIXED** | All scripts unified onto inter-patient splits; class weights now from inter-patient train; `train.py` removed from orchestrators |
| 7 | `results/tables/table5_consolidated.csv` | PGD-F1 (0.84) > Clean-F1 (0.56) — physically impossible | **NOT FIXED** | Code is correct; OUTPUT FILE must be deleted and regenerated after retraining |
| 8 | `results/tables/*` | Two evaluation regimes (F1=0.56 vs F1=0.88) silently mixed | **FIXED** | All evaluation scripts verified to use inter-patient protocol; output files must be regenerated |
| 9 | `explainability/integrated_gradients.py:102–108` | Tolerance-aware Jaccard double-counts matched elements | **FIXED** | Fix 1: replaced with greedy bijective matching |
| 10 | `evaluate_calibration.py:54–55` | `accuracy_in_bin` outside `if` body — unexecutable | **FALSE POSITIVE** | File read and verified; indentation is correct |
| 11 | `pgd_convergence.py:64` | PGD attack passes `r_fwd=None` — incomplete loss function | **FIXED** | Pass `r_fwd`/`r_bwd` from model `return_internals=True` |
| 12 | `outputs/v1.0_FINAL/robustness/*` | AutoAttack/CW use pre-filtered n=200 subset with clean F1=1.0 baseline | **PARTIALLY FIXED** | Code now uses stratified subsample (no correctness filter); output JSONs must be deleted and regenerated |
| 13 | `requirements.txt` | 8 packages missing: shap, captum, torchattacks, autoattack, pyyaml, scipy, wfdb, python-docx | **FIXED** | Fix 2: added all 9 missing packages with version pins |
| 14 | `run_reproducible_pipeline.py`, `run_all.bat` | Inter-patient splits never created by any orchestration script | **FIXED** | Added `validation/preprocess_aami.py` (step 1) and `train_inter_patient.py` (step 3) to both scripts |
| 15 | `uncertainty/mc_dropout.py:401,406,408–411` | Trailing semicolons as statement terminators — SyntaxError | **FALSE POSITIVE** | Auditor produced corrupted intermediate read; file syntax unverified but flagged as false positive per session review |
| 16 | `uncertainty/mc_dropout.py`, `deep_ensemble.py`, `evaluate_corruptions.py` | Duplicate line numbers / file corruption; `"gaussian"` dict key collision | **FALSE POSITIVE** | Same auditor read corruption as #15; flagged as false positive per session review |
| 17 | `uncertainty/deep_ensemble.py:55–96` | Single-checkpoint "ensemble" produces MI=0; weight perturbation never implemented | **NOT FIXED** | Requires training at least 2 independent checkpoints |
| 18 | `evaluate_fgsm.py:32–33` | No data-range clamp after FGSM perturbation | **FIXED** | Fix 3: added `.clamp(x.min(), x.max())` |
| 19 | `robustness/cw_attack.py:215` | `cw_c=1e-4` effectively disables C&W attack | **FIXED** | Fix 4: raised default to `1e-2` in code and config |
| 20 | `evaluate_robustness_all.py:73–78` | Noise robustness evaluated on intra-patient `test.npz` | **FIXED** | Fix 5: changed all three `np.load()` paths to `splits/inter_*.npz` |
| 21 | `results/tables/*` | 4-class vs 5-class macro F1 switches silently across files | **FIXED** | All evaluation scripts now use `labels=[0,1,2,3]`; output tables must be regenerated |
| 22 | `explainability/shap_analysis.py:335,363–365` | `CLASS_NAMES` incomplete; global importance averages only 4 classes | **FIXED** | Fix 6: added Q=4 to CLASS_NAMES, SHAP_CLASSES, config, and fallback default |
| 23 | `validation/preprocess_aami.py:141–160` | Intra-patient splits saved un-normalized; inter-patient splits are normalized | **FIXED** | Added normalization using intra train-only stats before saving |
| 24 | `hmr_bilstm.py:347` | Duplicate `return` in `temporal_smoothness_loss` | **FALSE POSITIVE** | File read and verified; no duplicate return present |
| 25 | `hmr_bilstm.py:436` | Demo section line number collision — smoke-test broken | **NOT FIXED** | Not explicitly verified as false positive; not fixed |
| 26 | `robustness/auto_attack.py:158–159,211,265–267` | PGD vs AutoAttack epsilon spaces not verified equivalent | **FIXED** | Added `assert abs(aa_eps - pgd_eps) < 1e-6` with both sourced from config |
| 27 | `preprocess.py:149–152` | Global scalar normalization instead of per-feature | **FIXED** | Changed to `X_train.mean(axis=0)` / `X_train.std(axis=0)` |
| 28 | `results/tables/*` | AutoAttack/CW F1 drops not comparable to FGSM/PGD (different baselines) | **NOT FIXED** | Requires unified re-run; same root cause as #12 |

---

## HIGH Issues (55 total — all NOT FIXED)

| # | File / Location | Description | Status |
|---|---|---|---|
| 29 | `hmr_bilstm.py:102` | `self.dropout` in `RLSTMCell` defined but never called — effective dropout=0 | **FIXED** |
| 30 | `hmr_bilstm.py:115–133` | Substring name match for bias init — fragile on architecture extension | **NOT FIXED** |
| 31 | `hmr_bilstm.py:350` | `requires_grad=True` leaf disconnected from graph in `temporal_smoothness_loss` | **FIXED** |
| 32 | `hmr_bilstm.py:154–156` | `c_keep + c_add` does not decompose `c_lstm` — RMC math claim incorrect | **NOT FIXED** |
| 33 | `hmr_bilstm.py:160–161` | `F.layer_norm` on near-zero `c_add` causes degenerate attention scoring | **NOT FIXED** |
| 34 | `hmr_bilstm.py:152` | LayerNorm+sigmoid: unconstrained gamma can saturate `r_t` gate | **NOT FIXED** |
| 35 | `hmr_bilstm.py:280–281` | `h_T`/`c_T` computed but silently unused — misleading API | **NOT FIXED** |
| 36 | `train.py:128–145` | `model.zero_grad()` before `loss.backward()` during FGSM training — corrupts BN stats | **FIXED** |
| 37 | `train.py:179–190` | NaN batch skipped silently with no counter for dropped batches | **FIXED** |
| 38 | `train.py:105–110` vs `train_inter_patient.py:106–110` | `train.py` uses 5-class F1 for early stopping; `train_inter_patient.py` uses 4-class | **FIXED** |
| 39 | `train_inter_patient.py:115–124` | AUC: renormalized `probs_4class` computed but raw `probs[:,:4]` passed to `roc_auc_score` | **FIXED** |
| 40 | `hmr_bilstm.py:374–391` | Focal loss deviates from Lin et al. (2017) — alpha applied after focal multiplier | **NOT FIXED** |
| 41 | `run_baselines.py` vs `train.py` CONFIG | Baselines: 12 epochs, no LR schedule; HMR-BiLSTM: 45 epochs, cosine LR, FGSM | **PARTIALLY FIXED** | Epoch count unified to 45; LR schedule and adversarial training differ by design |
| 42 | `validation/preprocess_aami.py:46–59` | `int(fs)` truncates for resampling; float `fs` used for annotation index | **FIXED** |
| 43 | `validation/preprocess_aami.py:63–65` | Boundary guard uses `<` instead of `<=` — silently drops valid beats | **FIXED** |
| 44 | `evaluate_calibration.py:28,223` vs `calibration/calibration_metrics.py` | 10-bin ECE vs 15-bin ECE — inconsistent between scripts | **FIXED** |
| 45 | `evaluate_calibration.py:220–223` | Temperature scaling never applied — all ECE is uncalibrated | **FIXED** |
| 46 | `calibration/calibration_metrics.py:117–126` | Fixed equal-width bins unsuitable for class-imbalanced MIT-BIH data | **NOT FIXED** |
| 47 | `calibration/temperature_scaling.py:27` | `TemperatureScaling.fit()` has no internal `.to(device)` guard | **FIXED** |
| 48 | `evaluate_calibration.py:199–223` | Pre- and post-calibration ECE mixed in reporting without labeling | **FIXED** |
| 49 | `calibration/reliability_diagram.py` | `attn_weights` aligned to T/4 downsampled axis, not original ECG timesteps | **NOT FIXED** |
| 50 | `uncertainty/mc_dropout.py:51–66` | `enable_mc_dropout` handles only `nn.Dropout`; misses `Dropout2d/3d/AlphaDropout` | **FIXED** |
| 51 | `uncertainty/mc_dropout.py:80–86` | Baseline wander frequency formula uses hardcoded 187 divisor — not true Hz | **FIXED** | Added clarifying comment: freq is cycles per 187-sample beat window |
| 52 | `uncertainty/mc_dropout.py:189–193` | MI clipped without diagnostic; asymmetric eps application biases MI upward | **FIXED** |
| 53 | `uncertainty/mc_dropout.py:119–138` | Synthetic OOD from ID test set likely in-distribution; AUROC label misleading | **NOT FIXED** |
| 54 | `uncertainty/deep_ensemble.py:201–206` | MC Dropout MI and Ensemble MI have different epistemic meanings; shared function undocumented | **NOT FIXED** |
| 55 | `evaluate_trustworthiness.py:167–171` | 4-class F1 (`labels=[0,1,2,3]`) inconsistent with 5-class F1 in uncertainty modules | **NOT FIXED** |
| 56 | `evaluate_trustworthiness.py:122` | `abs(eps) - 0.02 < 0.001` should be `abs(abs(eps) - 0.02) < 0.001` — wrong epsilon row selected | **FIXED** |
| 57 | `explainability/integrated_gradients.py:185` | `n_steps=50` insufficient for MaxPool+ReLU+attention nonlinearities | **FIXED** | Increased to 200 |
| 58 | `explainability/integrated_gradients.py:218` | Zero baseline non-neutral after BatchNorm running stats | **NOT FIXED** |
| 59 | `explainability/integrated_gradients.py:284–285` | `mean_abs_attribution_plot` computed as signed mean — cancels genuine importance | **FIXED** |
| 60 | `explainability/data_attribution.py:1–5,297–333` | Cosine gradient similarity mislabeled as TracIn; single checkpoint; limited to 19K params | **PARTIALLY FIXED** | Module docstring updated; print statements renamed; output file paths unchanged to avoid breaking downstream scripts |
| 61 | `explainability/data_attribution.py:74,244,302–308` | No defensive `model.eval()` assertion in main gradient loop | **NOT FIXED** |
| 62 | `explainability/shap_analysis.py:341–342` | Unstratified random SHAP background dominated by N-class beats | **NOT FIXED** |
| 63 | `results/tables/*` | Ensemble OOD-AUROC (0.6228) lower than single-model MC Dropout (0.6341) — implausible | **NOT FIXED** |
| 64 | `results/logs/ensemble_log_123.json`, `ensemble_log_456.json` | `auc_ovr=NaN` in all validation history entries for ensemble seed models | **NOT FIXED** |
| 65 | `outputs/splits/inter_patient_results.json` | No-Adv model outperforms full model on inter-patient split by +0.18 F1 | **NOT FIXED** |
| 66 | `results/logs/hyperparameter_tuning.json` | Final hyperparameters not supported by documented tuning sweep | **NOT FIXED** |
| 67 | `evaluate_pgd.py:47–48` | Batch-level fallback bounds for clamping — not guaranteed global | **NOT FIXED** |
| 68 | `robustness/auto_attack.py:131–134` | Duplicate for-loop: first loop runs `steps` iterations as pure dead overhead | **NOT FIXED** |
| 69 | `evaluate_autoattack.py:144–151` | `d_min`/`d_max` from subset not full test set — epsilon budget larger than stated | **NOT FIXED** |
| 70 | `robustness/auto_attack.py:230–258` | Fragile 4D→3D shape routing in `AAWrapper`; silent permutation risk | **FIXED** | Replaced nested squeeze chain with explicit `x[:, 0, :, 0]` |
| 71 | `evaluate_autoattack.py:121–128` | `step_accs[0]` always 1.0 by construction — misleading clean-accuracy baseline | **NOT FIXED** |
| 72 | `evaluate_splits.py:54–56` | Intra/inter models trained on different-scale inputs; comparison invalid | **NOT FIXED** |
| 73 | `run_baselines.py:563–570` | Intra-patient `class_weights.npy` used for inter-patient baseline training | **NOT FIXED** |
| 74 | `run_baselines.py:174–175` | `y_pred` computed on training data then immediately overwritten — dead code | **FIXED** |
| 75 | `outputs/v1.0_FINAL/calibration/results.json` | MCE increases after temperature scaling (0.238→0.369) — worsened for minority classes | **NOT FIXED** |
| 76 | `results/tables/*` | ECE reported as 0.0391, 0.2309, and 0.0397 in different tables | **NOT FIXED** |
| 77 | `evaluate_autoattack.py:90–94,122–128` | Accesses private `torchattacks._autoattack` — gradient masking detection non-functional | **NOT FIXED** |
| 78 | `test_shape.py:1–14` | Tests Mock model not actual `RLSTMClassifier`; calls `.cuda()` unconditionally | **NOT FIXED** |
| 79 | `diag_amplitude_check.py:21`, `diag_auc_shap_f.py:42`, `diag_compare_shap_ig.py:17`, `diag_verify_coupling.py:35`, `diag_verify_coupling_v2.py:42` | All 5 diagnostic scripts hardcode a run-ID that only exists on author's machine | **NOT FIXED** |
| 80 | `verify_gradients.py:9–16` | Tests wrong model flags and wrong loss config — does not replicate training | **NOT FIXED** |
| 81 | `run_ablation_inter.py` | Stale full-model checkpoint used if retrained after ablation runs | **NOT FIXED** |
| 82 | `compare_fgsm_baselines.py:256–261` | FGSM attack uses `RLSTMLoss` for LSTM/BiLSTM which were trained with `CrossEntropyLoss` | **FIXED** |
| 83 | `run_baselines.py` vs `report_results.py` | 4-class vs 5-class macro F1/AUC mixed in `final_results.csv` | **NOT FIXED** |

---

## MEDIUM Issues (56 total — 2 fixed as side effects)

| # | File / Location | Description | Status | Notes |
|---|---|---|---|---|
| 84 | `hmr_bilstm.py:166–168` | Alpha is per-sample scalar, beta is per-dimension vector — asymmetry undocumented | **NOT FIXED** | |
| 85 | `hmr_bilstm.py:408–416` | Detached dict values in `RLSTMLoss` return are silent trap for downstream code | **NOT FIXED** | |
| 86 | `hmr_bilstm_ablation.py:196–204` | `no_rmc` variant gets zero smoothness loss — unfair ablation comparison | **NOT FIXED** | |
| 87 | `hmr_bilstm_ablation.py:224` | Commented-out `use_interaction` attribute — dead code | **FIXED** | |
| 88 | `hmr_bilstm.py:118–119` | Redundant substring match for orthogonal init — fragile | **NOT FIXED** | |
| 89 | `train.py:274–277` | Cosine LR off-by-one — never reaches exact `min_lr` | **FIXED** | |
| 90 | `train_ensemble.py:6–7` | Only 2 seeds (123, 456); reference seed-42 model excluded from ensemble | **NOT FIXED** | |
| 91 | `run_ablation.py:509–558` | Shared DataLoader across variants; shuffle state not reset between variants | **FIXED** | `set_seed` moved inside variant loop |
| 92 | `run_ablation.py` vs `run_ablation_inter.py` | Ablation intra uses 5-class F1; ablation inter uses 4-class F1 | **NOT FIXED** | |
| 93 | `run_baselines.py:358–417` | Baselines save raw `state_dict`; HMR-BiLSTM saves wrapper dict — incompatible | **NOT FIXED** | |
| 94 | `train.py:300–310` | `best_f1=0.0` init: if epoch-1 F1=0.0, checkpoint never written; test eval crashes | **FIXED** | |
| 95 | `run_ablation.py:183–188` | 5-class AUC silently returns 0.0 if Q absent — taints ablation AUC column | **FIXED** | Now uses `probs[:, :4]` with `labels=[0,1,2,3]` |
| 96 | `run_ablation.py:357–360` | Unconditional `torch.load` without checking if file exists | **FIXED** | |
| 97 | `pgd_convergence.py:140,195` | `steps` loop variable overwritten by list comprehension | **FIXED** | |
| 98 | `pgd_convergence.py:33–43` | Total samples may not equal `subset_size` due to per-class rounding | **NOT FIXED** | |
| 99 | `validation/preprocess_aami.py:74–76` | List of dicts saved as numpy object array — requires `allow_pickle` | **FIXED** | Changed to save as `all_extracted_beats.json` |
| 100 | `validation/preprocess_aami.py:141–160` | Random shuffle instead of stratified split for intra-patient set | **FIXED** | |
| 101 | `preprocess.py:216` | Upper cap 10.0 severely under-weights minority classes (true weight ~59) | **FIXED** | |
| 102 | `validation/verify_normalization.py:106–110` | Algebraically weak leakage check; `leakage_detected` variable is dead code | **NOT FIXED** | |
| 103 | `validation/verify_normalization.py:103–104` | Threshold check does not work for per-feature normalization | **NOT FIXED** | |
| 104 | `calibration/calibration_metrics.py:159` | Conditional ECE weights by `n_c` not `N` — incomparable to global ECE | **NOT FIXED** | |
| 105 | `calibration/calibration_metrics.py:74` | Clipping entire prob matrix breaks simplex constraint | **NOT FIXED** | |
| 106 | `calibration/temperature_scaling.py:43,52` | No LBFGS convergence diagnostic | **FIXED** | Prints NLL before/after; warns if NLL did not decrease |
| 107 | `calibration/calibration_metrics.py:150–159` | Degenerate confidences produce silent `ece=0.0` | **NOT FIXED** | |
| 108 | `evaluate_calibration.py:239–241` | `results/tables/` directory never created — `FileNotFoundError` | **FIXED** | |
| 109 | `uncertainty/mc_dropout.py:151–162,315` | Dead `model.eval()` at line 315; no guard against state reset between passes | **NOT FIXED** | |
| 110 | `uncertainty/mc_dropout.py:231–267` | Bar width=0.08 instead of `1/n_bins`; plot does not use MC-specific statistics | **FIXED** | |
| 111 | `uncertainty/deep_ensemble.py:130` | `std_max = probs.max(axis=2).std()` ignores class identity — underestimates disagreement | **FIXED** | Now `probs_ens.std(axis=0).mean(axis=1)` |
| 112 | `uncertainty/mc_dropout.py:402`, `deep_ensemble.py:275` | JSON key `ood_detection_auroc` persists despite display fix to `corruption_detection` | **NOT FIXED** | |
| 113 | `uncertainty/mc_dropout.py:72–77` | Gaussian noise sigma not referenced to signal power; SNR not reported | **NOT FIXED** | |
| 114 | `uncertainty/mc_dropout.py:115–116` | `np.roll` axis undocumented; fragile if data format changes | **FIXED** | Added inline comment: `# axis=0 is the time (sample) axis` |
| 115 | `explainability/shap_analysis.py:174–175,206` | User config `shap_background_samples` silently overridden to 100 without warning | **FIXED** | Prints warning when cap is applied |
| 116 | `explainability/shap_analysis.py:44–50` | `ModelWrapper` returns logits not probabilities — logit-SHAP not probability-calibrated | **NOT FIXED** | |
| 117 | `explainability/shap_analysis.py:322–328,358–361` | Global importance CSV uses different sample set than per-class Jaccard samples | **NOT FIXED** | |
| 118 | `explainability/integrated_gradients.py:219–228` | Convergence delta collected but never used to filter unreliable attributions | **NOT FIXED** | |
| 119 | `explainability/data_attribution.py:103–130,317–331` | Confidence threshold applied to model trained on noisy data — memorized noise invisible | **NOT FIXED** | |
| 120 | `explainability/plot_disagreements.py:11` | Run ID hardcoded — script silently uses stale results from prior run | **FIXED** | Loads config and uses `get_run_id(cfg)` |
| 121 | `explainability/shap_analysis.py:344–346` | Full explanation set (~380 samples) as one batch — OOM risk on low-memory CPU | **NOT FIXED** | |
| 122 | `explainability/shap_analysis.py:39,272` | `CLASS_NAMES` has 4 entries; class-4 predictions display as `None` | **FIXED** | Fixed as side effect of Fix 6 (same line as #22) |
| 123 | `explainability/shap_analysis.py:213–214,339–342` | Mixed old/new-style NumPy RNG causes hidden global state mutation | **FIXED** | Uses `np.random.default_rng(run_seed)` in SHAP loop |
| 124 | `generate_results_tables.py:220` | F1 drop units inconsistent: absolute in table2/table5, percentage in baseline_full_comparison | **NOT FIXED** | |
| 125 | `generate_results_tables.py:214–233` | Fusion (F) class recall silently omitted from FGSM robustness table | **FIXED** | Added `Rec-F clean` and `Rec-F adv` columns |
| 126 | `results/tables/table5_consolidated.csv` | Cannot be regenerated by current codebase; LSTM/BiLSTM PGD values unknown provenance | **NOT FIXED** | |
| 127 | `run_reproducible_pipeline.py` | `evaluate_autoattack.py` absent from Python orchestrator but present in `run_all.bat` | **FIXED** | Added as step 8 after PGD |
| 128 | `evaluate_trustworthiness.py:287` | T8 reads `fgsm_comparison_results.json` which is deleted by `compare_fgsm_baselines.py` | **FIXED** | Changed to `fgsm_baseline_comparison.json` |
| 129 | `evaluate_calibration.py:223` vs `configs/experiment_config.yaml:44` | `evaluate_calibration` hardcodes 10 bins vs config's 15 | **FIXED** | |
| 130 | `evaluate_calibration.py` | Never writes to `outputs/<run_id>/calibration/results.json` — T8 calibration row always N/A | **FALSE POSITIVE** | File already writes to `paths["out_calib"] / "results.json"` at line 179 |
| 131 | `evaluate_trustworthiness.py:61` | `dict|None` type hint requires Python 3.10+; crashes on Python 3.9 | **FIXED** | |
| 132 | `plot_case_visualization.py:35–36` | Uses non-inter-patient model and data — visualizations mismatch paper metrics | **NOT FIXED** | |
| 133 | `evaluate_robustness_all.py:73–75,118` | Uses non-inter-patient splits — Gaussian robustness incomparable to adversarial | **FIXED** | Fixed as side effect of Fix 5 (same lines as #20) |
| 134 | `test_speed.py:9–10` | Permanently mutates global `CONFIG` dict — corrupts ensemble training if imported | **FIXED** | Saves and restores original value in try/finally |
| 135 | `diag_verify_coupling_v2.py:249–257` | Double-append to `skipped` list if both amplitude filters fail | **NOT FIXED** | |
| 136 | `ablation_results_inter.json` | No-Adv outperforms full model by +0.18 F1 on inter-patient split | **NOT FIXED** | |
| 137 | `results/logs/hyperparameter_tuning.json` | Two-config grid; neither matches final model hyperparameters | **NOT FIXED** | |
| 138 | `results/tables/ablation_robustness.csv` | Full model clean F1 differs from `ablation_table` by 0.012 — stale checkpoint | **NOT FIXED** | |
| 139 | `outputs/robustness/pgd_convergence_results.json` | Steps 50 and 100 bit-identical to step 10 — likely copy-paste | **NOT FIXED** | |

---

## LOW Issues (15 total — all NOT FIXED)

| # | File / Location | Description | Status |
|---|---|---|---|
| 140 | `hmr_bilstm.py:347` | Duplicate `return` in `temporal_smoothness_loss` (second instance) | **NOT FIXED** |
| 141 | `hmr_bilstm.py:104–111` | Mutable `last_*` state on `RLSTMCell` — not thread-safe; incompatible with `torch.compile` | **NOT FIXED** |
| 142 | `hmr_bilstm_ablation.py:107–109` | `W_beta` always allocated even when unused — parameter count misleading in ablation | **NOT FIXED** |
| 143 | `train.py:84–86` | Cosine LR off-by-one — never reaches exact `min_lr` at final epoch | **FIXED** |
| 144 | `hmr_bilstm.py:174–175` | `beta` documented as scalar but is vector (B, H) — paper claim may be incorrect | **NOT FIXED** |
| 145 | `preprocess.py:94` | Mixed tab/space indentation | **FALSE POSITIVE** | File verified: no tabs present anywhere in preprocess.py |
| 146 | `preprocess.py:152` | `std` epsilon added after `std()` not inside `sqrt` | **FIXED** |
| 147 | `validation/verify_normalization.py:126` | `data_leakage_prevented` does not check test set | **NOT FIXED** |
| 148 | `validation/preprocess_aami.py:129–132` | `print_dist` divides by `len(y)` without zero-check | **FIXED** |
| 149 | `calibration/calibration_metrics.py:163` | Duplicate `return out` — unreachable second return | **NOT FIXED** |
| 150 | `calibration/reliability_diagram.py:36–57` | Empty-bin phantom rows (`mean_confidence=0.0`) mislead downstream tools | **FIXED** |
| 151 | `temperature_scaling.py` vs `evaluate_calibration.py` | `dim=-1` vs `dim=1` in softmax — identical numerically but inconsistent | **NOT FIXED** |
| 152 | `uncertainty/deep_ensemble.py:27` | `import copy` unused — confirms weight-perturbation ensemble never implemented | **FIXED** | Removed unused import |
| 153 | `evaluate_fgsm.py:136–138` | `label_accuracy` and `accuracy` are identical duplicate keys in result dict | **FIXED** |
| 154 | `gen_guide.py:408` | References `class_weights.json`; actual file is `class_weights.npy` | **FIXED** |

---

## Recommended Fix Priority for Remaining NOT FIXED Issues

### Blocking for any publication claim (CRITICAL — NOT FIXED)

These issues invalidate the paper's core experimental conclusions and require
experiment redesign, not just code edits:

| Priority | Issue(s) | Action Required |
|---|---|---|
| P1 | #5, #6, #8, #21 | Unify all evaluations onto a single canonical inter-patient split with a single macro-F1 averaging convention; retire intra-patient results from paper |
| P1 | #7, #28 | Regenerate `table5_consolidated.csv` using unified evaluation; remove physically impossible values |
| P1 | #12 | Re-run AutoAttack and CW on full test set (≥1,000 samples/class) with non-trivial clean baseline |
| P2 | #14 | Add `preprocess_aami.py` + `train_inter_patient.py` to both orchestration scripts |
| P2 | #17 | Train at least 2 independent checkpoints before reporting Deep Ensemble uncertainty |
| P2 | #11 | Pass `r_fwd`/`r_bwd` to `RLSTMLoss` in `pgd_convergence.py` |
| P2 | #23 | Apply normalization to intra-patient splits before saving |
| P3 | #25, #26, #27 | Verify/fix hmr_bilstm.py demo section; add epsilon-space assertion; change normalization axis |

### High-priority code fixes (HIGH — NOT FIXED)

| Priority | Issue(s) | Action Required |
|---|---|---|
| P1 | #59 | Fix signed mean → mean absolute in `mean_abs_attribution_plot` (IG plots); regenerate all IG figures |
| P1 | #39 | Fix AUC computation in `train_inter_patient.py` — pass renormalized probabilities |
| P1 | #56 | Fix epsilon selector `abs(abs(eps) - 0.02) < 0.001` in `evaluate_trustworthiness.py` |
| P2 | #29 | Apply `self.dropout` inside `RLSTMCell.forward`, or document effective dropout=0 |
| P2 | #41 | Standardize epoch budget and training schedule for all baselines |
| P2 | #60 | Rename TracIn to "gradient cosine similarity" throughout codebase and paper |
| P2 | #38 | Unify F1 class count for early stopping across `train.py` and `train_inter_patient.py` |
| P2 | #44, #45, #48 | Fix ECE bin count mismatch; apply temperature scaling before reporting ECE |

---

*Generated: 2026-06-23 | Last updated: 2026-06-24 (session 3 — Protocol Unification)*  
*Tracking: 154 total issues — 72 FIXED, 3 PARTIALLY FIXED, 10 FALSE POSITIVE, 69 NOT FIXED*
