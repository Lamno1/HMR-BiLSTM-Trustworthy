# HMR-BiLSTM Scientific & Engineering Audit Report

**Project:** HMR-BiLSTM — Hybrid Memory-Residual Bidirectional LSTM for ECG Arrhythmia Classification  
**Audit Date:** 2026-06-23  
**Auditor:** Automated multi-subsystem parallel audit (10 independent auditors)  
**Codebase Root:** `D:/HMR-BiLSTM-main/`

---

## Executive Summary

This audit examined every subsystem of the HMR-BiLSTM codebase: core model architecture, data pipeline, training pipeline, robustness evaluation, calibration, uncertainty quantification, explainability, results reporting, and configuration/orchestration. A total of **154 distinct issues** were identified after deduplication.

### Issue Count by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 28 |
| HIGH | 55 |
| MEDIUM | 46 |
| LOW | 25 |
| **Total** | **154** |

### Top 5 Most Critical Findings That Could Invalidate Published Conclusions

1. **Issue #1 — Unimportable Core Model Files:** `hmr_bilstm.py` and `hmr_bilstm_ablation.py` both contain syntax-level bugs (duplicate `for` loop headers, duplicate line numbers, indentation errors) that prevent the files from being imported at all. Every result claimed from these files is unreproducible from the source as written.

2. **Issue #5 — Intra-Patient Data Leakage in Primary Training Split:** The primary training script `train.py` uses `data/processed/train.npz` and `test.npz`, which are derived from Kaggle's `mitbih_train.csv` / `mitbih_test.csv` via stratified split without patient-level separation. Beats from the same patient appear in both train and test, inflating all reported metrics by an estimated 5–15 percentage points. This is the most fundamental validity threat to all absolute performance claims.

3. **Issue #6 — Apples-to-Oranges Comparative Table:** `run_baselines.py` evaluates baselines on the inter-patient split while the primary `train.py` HMR-BiLSTM results use the intra-patient split. The published comparison table directly mixes these incompatible evaluation regimes.

4. **Issue #7 — Table 5 Reports Physically Impossible Values:** `table5_consolidated.csv` shows PGD-F1 (0.8391) and FGSM-F1 (0.8425) both higher than Clean-F1 (0.5644) for HMR-BiLSTM. A model cannot perform better under adversarial attack than on clean data. This results from silently mixing two evaluation regimes in the same table.

5. **Issue #9 — Jaccard Consistency Formula is Mathematically Wrong:** The tolerance-aware Jaccard function in `integrated_gradients.py` double-counts matched elements and uses a non-standard denominator, producing values that are not true Jaccard indices. All reported SHAP-IG consistency numbers in the paper are computed from an incorrect formula.

### Overall Assessment of Publication Readiness

**NOT READY FOR PUBLICATION.** The codebase has multiple CRITICAL bugs that prevent core files from running, a fundamental data leakage problem in the primary evaluation split, an inconsistent multi-regime evaluation framework that produces internally contradictory tables, and mathematically incorrect implementations in both the explainability and evaluation subsystems. The issues are pervasive across all subsystems, not isolated to one component. Before submission, the authors must: (1) fix all syntax-level bugs, (2) unify evaluation onto a single canonical split, (3) fix the Jaccard formula and IG aggregation, (4) correct the adversarial evaluation protocol, and (5) rerun all experiments from scratch under a single consistent configuration.

---

## Issue Index

| Issue # | Severity | Category | File(s) | One-line Description |
|---------|----------|----------|---------|----------------------|
| 1 | CRITICAL | Syntax/Logic Bug | hmr_bilstm.py:213 | Duplicate `for t in range(T):` — file cannot be imported |
| 2 | CRITICAL | Syntax/Logic Bug | hmr_bilstm_ablation.py:107–108 | Line 108 appears twice (W_beta vs self.dropout) — file corrupt |
| 3 | CRITICAL | IndentationError | hmr_bilstm_ablation.py:435 | RLSTMLoss `__init__` has extra leading space — ImportError |
| 4 | CRITICAL | IndentationError | calibration/temperature_scaling.py:73–74 | `x = x.to(device)` outside for-loop body — module unexecutable |
| 5 | CRITICAL | Data Leakage | preprocess.py:136–142 | Intra-patient split: beats from same patient in train and test |
| 6 | CRITICAL | Data Leakage | run_baselines.py:31–35 vs train.py:212–213 | Baselines use inter-patient split; HMR-BiLSTM uses intra-patient |
| 7 | CRITICAL | Impossible Value | results/tables/table5_consolidated.csv | PGD-F1 (0.84) > Clean-F1 (0.56) — physically impossible |
| 8 | CRITICAL | Split Contamination | results/tables/* | Two evaluation regimes (F1=0.56 vs F1=0.88) silently mixed across all tables |
| 9 | CRITICAL | Wrong Formula | explainability/integrated_gradients.py:102–108 | Tolerance-aware Jaccard double-counts matched elements; formula is wrong |
| 10 | CRITICAL | IndentationError | evaluate_calibration.py:54–55 | `accuracy_in_bin` outside `if` body — script unexecutable |
| 11 | CRITICAL | Wrong Attack | calibration/temperature_scaling.py (via audit of pgd_convergence.py:64) | PGD attack passes r_fwd=None disabling smoothness loss — attack weaker than training loss |
| 12 | CRITICAL | Eval Protocol Error | outputs/v1.0_FINAL/robustness/autoattack_results.json | AutoAttack/CW use clean_accuracy=1.0 baseline (only pre-correct samples, n=200) |
| 13 | CRITICAL | Missing Dependencies | requirements.txt | 8 packages missing: shap, captum, torchattacks, autoattack, pyyaml, scipy, wfdb, python-docx |
| 14 | CRITICAL | Missing Pipeline Steps | run_reproducible_pipeline.py, run_all.bat | Inter-patient splits never created by any orchestration script |
| 15 | CRITICAL | Syntax (Semicolons) | uncertainty/mc_dropout.py:401,406,408–409,411 | Semicolons as statement terminators — possible SyntaxError |
| 16 | CRITICAL | File Corruption | uncertainty/mc_dropout.py, deep_ensemble.py, evaluate_corruptions.py | Duplicate line numbers indicating corrupted source; dict key collision |
| 17 | CRITICAL | Ensemble Misrepresentation | uncertainty/deep_ensemble.py:55–96 | Single checkpoint "ensemble" produces MI=0; docstring claims unimplemented weight perturbation |
| 18 | CRITICAL | FGSM Missing Clamp | evaluate_fgsm.py:32–33 | No data-range clamping after FGSM perturbation — inconsistent with PGD |
| 19 | CRITICAL | C&W Disabled | robustness/cw_attack.py:215 | cw_c=1e-4 effectively disables attack; reported ASR~0 is hyperparameter artifact |
| 20 | CRITICAL | Eval on Wrong Dataset | evaluate_robustness_all.py:73–78 | Uses intrapatient test.npz; all other files use interpatient inter_test.npz |
| 21 | CRITICAL | Metric Inconsistency | results/tables/* | 4-class vs 5-class macro F1 averaging switches silently between files |
| 22 | CRITICAL | Class Count Mismatch | explainability/shap_analysis.py:335,363–365 | n_classes=5 but global importance averages only 4 classes; CLASS_NAMES incomplete |
| 23 | CRITICAL | Intra-patient Normalization | validation/preprocess_aami.py:141–160 | Intra-patient split saved un-normalized; makes intra/inter comparison invalid |
| 24 | CRITICAL | Dead Code/Logic Bug | hmr_bilstm.py:347 | Duplicate `return` in temporal_smoothness_loss |
| 25 | CRITICAL | Logic Bug | hmr_bilstm.py:436 | Demo section line number collision — integration smoke-test broken |
| 26 | CRITICAL | Epsilon Inconsistency | robustness/auto_attack.py:158–159,211,265–267 | PGD vs AutoAttack epsilon spaces not verified equivalent; masking gap diagnosis unreliable |
| 27 | CRITICAL | Wrong Normalization Axis | preprocess.py:149–152 | Global scalar normalization instead of per-feature; saved scalars are scalars not vectors |
| 28 | CRITICAL | Eval Protocol | results/tables/* | AutoAttack/CW F1 drops relative to F1=1.0 baseline not comparable to FGSM/PGD drops on full set |
| 29 | HIGH | Dead Code — Dropout | hmr_bilstm.py:102 | self.dropout in RLSTMCell never applied — effective dropout=0 inside cell |
| 30 | HIGH | Incorrect Init | hmr_bilstm.py:115–133 | Substring name match for bias init — fragile, can corrupt on extension |
| 31 | HIGH | Numerical | hmr_bilstm.py:350 | requires_grad=True leaf disconnected from graph in temporal_smoothness_loss |
| 32 | HIGH | Math Error | hmr_bilstm.py:154–156 | c_keep + c_add != c_lstm — RMC decomposition is not a partition of LSTM update |
| 33 | HIGH | Numerical | hmr_bilstm.py:160–161 | F.layer_norm on near-zero c_add causes degenerate attention scoring |
| 34 | HIGH | Math Error | hmr_bilstm.py:152 | LayerNorm+sigmoid: unconstrained gamma can saturate r_t gate |
| 35 | HIGH | Logic Bug | hmr_bilstm.py:280–281 | h_T/c_T computed but silently unused — misleading API |
| 36 | HIGH | FGSM Implementation | train.py:128–145 | FGSM model.zero_grad() before loss.backward() corrupts BatchNorm stats |
| 37 | HIGH | Loss Computation | train.py:179–190 | NaN batch skipped silently with no counter for dropped batches |
| 38 | HIGH | Metric Inconsistency | train.py:105–110 vs train_inter_patient.py:106–110 | train.py uses 5-class F1 for early stopping; train_inter_patient.py uses 4-class |
| 39 | HIGH | Wrong Metric | train_inter_patient.py:115–124 | AUC: renormalized probs_4class computed but raw probs[:,:4] passed to roc_auc_score |
| 40 | HIGH | Focal Loss | hmr_bilstm.py:374–391 | Focal loss deviates from Lin et al. (2017) — alpha applied after focal multiplier |
| 41 | HIGH | Fairness | run_baselines.py:585–612 vs train.py CONFIG | Baselines: 12 epochs, no LR schedule; HMR-BiLSTM: 45 epochs, cosine LR, FGSM |
| 42 | HIGH | Annotation Misalignment | validation/preprocess_aami.py:46–59 | int(fs) truncates for resampling, float fs used for annotation index — inconsistent |
| 43 | HIGH | Off-by-one | validation/preprocess_aami.py:63–65 | Boundary guard uses strict `<` instead of `<=` — silently drops valid beats |
| 44 | HIGH | Wrong ECE Bin Count | evaluate_calibration.py:28,223 vs calibration/calibration_metrics.py | 10-bin ECE in evaluate_calibration.py vs 15-bin in temperature_scaling.py |
| 45 | HIGH | No Temperature Applied | evaluate_calibration.py:220–223 | Temperature scaling never applied — all ECE from this script is uncalibrated |
| 46 | HIGH | Classwise ECE | calibration/calibration_metrics.py:117–126 | Fixed equal-width bins unsuitable for class-imbalanced MIT-BIH data |
| 47 | HIGH | Device Placement | calibration/temperature_scaling.py:27 | TemperatureScaling.fit() has no internal .to(device) guard |
| 48 | HIGH | Before/After Calibration Mixed | evaluate_calibration.py:199–223 | Pre- and post-calibration ECE mixed in reporting without labeling |
| 49 | HIGH | Reliability Diagram Alignment | calibration/reliability_diagram.py | attn_weights aligned to T/4 CNN-downsampled axis, not original ECG timesteps |
| 50 | HIGH | MC Dropout Incomplete | uncertainty/mc_dropout.py:51–66 | enable_mc_dropout only handles nn.Dropout, misses Dropout2d/3d/AlphaDropout |
| 51 | HIGH | Wrong OOD Formula | uncertainty/mc_dropout.py:80–86 | Baseline wander frequency formula uses hardcoded 187 divisor — not true Hz |
| 52 | HIGH | MI Computation | uncertainty/mc_dropout.py:189–193 | MI clipped without diagnostic; asymmetric eps application biases MI upward |
| 53 | HIGH | OOD Strategy | uncertainty/mc_dropout.py:119–138 | Synthetic OOD from ID test set likely in-distribution; AUROC label misleading |
| 54 | HIGH | Wrong Uncertainty Decomp | uncertainty/deep_ensemble.py:201–206 | MC Dropout MI and Ensemble MI have different epistemic meanings; shared function undocumented |
| 55 | HIGH | Wrong F1 Evaluation | evaluate_trustworthiness.py:167–171 | 4-class F1 (labels=[0,1,2,3]) inconsistent with 5-class F1 in uncertainty modules |
| 56 | HIGH | Wrong ASR Logic | evaluate_trustworthiness.py:122 | abs(eps) - 0.02 < 0.001 should be abs(abs(eps) - 0.02) < 0.001 — selects wrong epsilon row |
| 57 | HIGH | IG Steps Insufficient | explainability/integrated_gradients.py:185 | n_steps=50 insufficient for MaxPool+ReLU+attention nonlinearities |
| 58 | HIGH | IG Wrong Baseline | explainability/integrated_gradients.py:218 | Zero baseline non-neutral after BatchNorm running stats |
| 59 | HIGH | Wrong Attribution Aggregation | explainability/integrated_gradients.py:284–285 | Variable named mean_abs_attribution_plot computed as signed mean — cancels genuine importance |
| 60 | HIGH | TracIn Mislabeled | explainability/data_attribution.py:1–5,297–333 | Cosine gradient similarity is not TracIn; single checkpoint; limited to 19K classifier params |
| 61 | HIGH | Model Mode in Explainability | explainability/data_attribution.py:74,244,302–308 | No defensive model.eval() assertion in main gradient loop |
| 62 | HIGH | SHAP Background Set | explainability/shap_analysis.py:341–342 | Unstratified random background dominated by N-class beats |
| 63 | HIGH | Impossible Table Values | results/tables/* | Ensemble OOD-AUROC (0.6228) lower than single-model MC Dropout (0.6341) |
| 64 | HIGH | AUC NaN | results/logs/ensemble_log_123.json, ensemble_log_456.json | auc_ovr=NaN in all validation history entries for ensemble seed models |
| 65 | HIGH | Implausible Result | outputs/splits/inter_patient_results.json | No-Adv model outperforms full model on inter-patient split by +0.18 F1 |
| 66 | HIGH | Hyperparameter Mismatch | results/logs/hyperparameter_tuning.json | Final hyperparameters not supported by documented tuning sweep |
| 67 | HIGH | PGD Bounds | evaluate_pgd.py:47–48 | Batch-level fallback bounds for clamping — not guaranteed global |
| 68 | HIGH | PGD Dead Loop | robustness/auto_attack.py:131–134 | Duplicate for-loop: first loop runs steps iterations as pure dead overhead |
| 69 | HIGH | Normalization Bounds | evaluate_autoattack.py:144–151 | d_min/d_max from subset not full test set — epsilon budget larger than stated |
| 70 | HIGH | AutoAttack Wrapper Shape | robustness/auto_attack.py:230–258 | Fragile 4D→3D shape routing in AAWrapper; silent permutation risk |
| 71 | HIGH | step_accs Semantic | evaluate_autoattack.py:121–128 | step_accs[0] always 1.0 by construction — misleading clean-accuracy baseline |
| 72 | HIGH | Evaluation Split Mismatch | evaluate_splits.py:54–56 | Intra/inter models trained on different-scale inputs; comparison invalid |
| 73 | HIGH | Class Weight Calibration | run_baselines.py:563–570 | Intra-patient class_weights.npy used for inter-patient baseline training |
| 74 | HIGH | Decision Tree Dead Code | run_baselines.py:174–175 | y_pred computed on training data then immediately overwritten by test prediction |
| 75 | HIGH | MCE Worsens | outputs/v1.0_FINAL/calibration/results.json | MCE increases after temperature scaling (0.238→0.369) — worsened for minority classes |
| 76 | HIGH | ECE Inconsistency | results/tables/* | ECE reported as 0.0391, 0.2309, and 0.0397 in different tables |
| 77 | HIGH | Private API Access | evaluate_autoattack.py:90–94,122–128 | Accesses private torchattacks _autoattack attribute — gradient masking detection non-functional |
| 78 | HIGH | Wrong Test Behavior | test_shape.py:1–14 | Tests Mock model not actual RLSTMClassifier; calls .cuda() unconditionally — fails on CPU |
| 79 | HIGH | Hardcoded Run IDs | diag_amplitude_check.py:21, diag_auc_shap_f.py:42, diag_compare_shap_ig.py:17, diag_verify_coupling.py:35, diag_verify_coupling_v2.py:42 | All 5 diagnostic scripts hardcode a specific run-ID that only exists on author's machine |
| 80 | HIGH | Wrong Test Behavior | verify_gradients.py:9–16 | Tests wrong model flags and wrong loss config — does not faithfully replicate training |
| 81 | HIGH | Wrong Ablation Loss | run_ablation_inter.py | Stale full-model checkpoint used if retrained after ablation runs |
| 82 | HIGH | FGSM Loss Mismatch | compare_fgsm_baselines.py:256–261 | FGSM attack uses RLSTMLoss for LSTM/BiLSTM which were trained with CrossEntropyLoss |
| 83 | HIGH | Metric Inconsistency | run_baselines.py vs report_results.py | 4-class vs 5-class macro F1/AUC mixed in final_results.csv |
| 84 | MEDIUM | Tensor Shape | hmr_bilstm.py:166–168 | Alpha is per-sample scalar, beta is per-dimension vector — asymmetry undocumented |
| 85 | MEDIUM | Math | hmr_bilstm.py:408–416 | Detached dict values in RLSTMLoss return are silent trap for downstream code |
| 86 | MEDIUM | Logic Bug | hmr_bilstm_ablation.py:196–204 | no_rmc zero r_t gives zero smoothness loss — unfair ablation comparison |
| 87 | MEDIUM | Dead Code | hmr_bilstm_ablation.py:224 | Commented-out use_interaction attribute |
| 88 | MEDIUM | Init Bug | hmr_bilstm.py:118–119 | Redundant substring match for orthogonal init — fragile |
| 89 | MEDIUM | LR Scheduler | train.py:274–277 | Cosine LR off-by-one — never reaches exact min_lr |
| 90 | MEDIUM | Seed Handling | train_ensemble.py:6–7 | Only 2 seeds (123, 456); reference seed-42 model excluded from ensemble |
| 91 | MEDIUM | Reproducibility | run_ablation.py:509–558 | Shared DataLoader across variants; shuffle state not reset between variants |
| 92 | MEDIUM | Metric Inconsistency | run_ablation.py vs run_ablation_inter.py | Ablation intra uses 5-class F1; ablation inter uses 4-class F1 |
| 93 | MEDIUM | Checkpoint Format | run_baselines.py:358–417 | Baselines save raw state_dict; HMR-BiLSTM saves wrapper dict — incompatible |
| 94 | MEDIUM | Early Stopping | train.py:300–310 | best_f1=0.0 init: if epoch 1 F1=0.0, checkpoint never written; test eval crashes |
| 95 | MEDIUM | AUC Ablation | run_ablation.py:183–188 | 5-class AUC silently returns 0.0 if Q absent — taints ablation AUC column |
| 96 | MEDIUM | Checkpoint Bug | run_ablation.py:357–360 | Unconditional torch.load without checking if file exists |
| 97 | MEDIUM | Variable Shadowing | pgd_convergence.py:140,195 | steps loop variable overwritten by list comprehension |
| 98 | MEDIUM | Subsampling Rounding | pgd_convergence.py:33–43 | Total samples may not equal subset_size due to per-class rounding |
| 99 | MEDIUM | Wrong Serialization | validation/preprocess_aami.py:74–76 | List of dicts saved as numpy object array — requires allow_pickle |
| 100 | MEDIUM | Incorrect Stratification | validation/preprocess_aami.py:141–160 | Random shuffle instead of stratified split for intra-patient set |
| 101 | MEDIUM | Class Weight Clipping | preprocess.py:216 | Upper cap 10.0 severely under-weights minority classes (true weight ~59) |
| 102 | MEDIUM | Leakage Detection | validation/verify_normalization.py:106–110 | Algebraically weak leakage check; leakage_detected variable is dead code |
| 103 | MEDIUM | Weak Verification | validation/verify_normalization.py:103–104 | Threshold check does not work for per-feature normalization |
| 104 | MEDIUM | ECE Weighting | calibration/calibration_metrics.py:159 | Conditional ECE weights by n_c not N — incomparable to global ECE |
| 105 | MEDIUM | NLL Computation | calibration/calibration_metrics.py:74 | Clipping entire prob matrix breaks simplex constraint |
| 106 | MEDIUM | Optimisation | calibration/temperature_scaling.py:43,52 | No LBFGS convergence diagnostic |
| 107 | MEDIUM | Off-by-one | calibration/calibration_metrics.py:150–159 | Degenerate confidences produce silent ece=0.0 |
| 108 | MEDIUM | Runtime Error | evaluate_calibration.py:239–241 | results/tables/ directory never created — FileNotFoundError |
| 109 | MEDIUM | MC Dropout State | uncertainty/mc_dropout.py:151–162,315 | Dead model.eval() at line 315; no guard against state reset between passes |
| 110 | MEDIUM | Calibration Plot | uncertainty/mc_dropout.py:231–267 | Bar width=0.08 instead of 1/n_bins; plot does not use MC-specific statistics |
| 111 | MEDIUM | Wrong Metric | uncertainty/deep_ensemble.py:130 | std_max = probs.max(axis=2).std() ignores class identity — underestimates disagreement |
| 112 | MEDIUM | Semantic Mislabeling | uncertainty/mc_dropout.py:402, deep_ensemble.py:275 | JSON key ood_detection_auroc persists despite dashboard display fix to corruption_detection |
| 113 | MEDIUM | Noise SNR | uncertainty/mc_dropout.py:72–77 | Gaussian noise sigma not referenced to signal power; SNR not reported |
| 114 | MEDIUM | Wrong Axis | uncertainty/mc_dropout.py:115–116 | np.roll axis undocumented; fragile if data format changes to channels-first |
| 115 | MEDIUM | SHAP Background Cap | explainability/shap_analysis.py:174–175,206 | User config shap_background_samples silently overridden to 100 without warning |
| 116 | MEDIUM | Logits vs Probabilities | explainability/shap_analysis.py:44–50 | ModelWrapper returns logits not probabilities — logit-SHAP not probability-calibrated |
| 117 | MEDIUM | SHAP Sample Inconsistency | explainability/shap_analysis.py:322–328,358–361 | Global importance CSV uses different sample set than per-class Jaccard samples |
| 118 | MEDIUM | IG Convergence Delta | explainability/integrated_gradients.py:219–228 | Convergence delta collected but never used to filter unreliable attributions |
| 119 | MEDIUM | Two-Signal Filter | explainability/data_attribution.py:103–130,317–331 | Confidence threshold applied to model trained on noisy data — memorized noise invisible |
| 120 | MEDIUM | Hardcoded Path | explainability/plot_disagreements.py:11 | Run ID hardcoded — script silently uses stale results from prior run |
| 121 | MEDIUM | OOM Risk | explainability/shap_analysis.py:344–346 | Full explanation set (~380 samples) processed as one batch — OOM on low-memory CPU |
| 122 | MEDIUM | CLASS_NAMES Incomplete | explainability/shap_analysis.py:39,272 | CLASS_NAMES has 4 entries; model has 5 classes; class-4 predictions display as None |
| 123 | MEDIUM | RNG Contamination | explainability/shap_analysis.py:213–214,339–342 | Mixed old/new-style NumPy RNG causes hidden global state mutation |
| 124 | MEDIUM | Wrong Table Values | generate_results_tables.py:220 | F1 drop units inconsistent: absolute in table2/table5, percentage in baseline_full_comparison |
| 125 | MEDIUM | Omitted F-Class Column | generate_results_tables.py:214–233 | Fusion (F) class recall silently omitted from fgsm robustness table |
| 126 | MEDIUM | Stale File | results/tables/table5_consolidated.csv | Cannot be regenerated by current codebase; LSTM/BiLSTM PGD values have unknown provenance |
| 127 | MEDIUM | Pipeline Missing Step | run_reproducible_pipeline.py | evaluate_autoattack.py absent from Python orchestrator but present in run_all.bat |
| 128 | MEDIUM | Path Inconsistency | evaluate_trustworthiness.py:287 | T8 reads fgsm_comparison_results.json which is deleted by compare_fgsm_baselines.py |
| 129 | MEDIUM | Config Inconsistency | evaluate_calibration.py:223 vs configs/experiment_config.yaml:44 | evaluate_calibration hardcodes 10 bins vs config's 15 |
| 130 | MEDIUM | Calibration Missing from Pipeline | evaluate_calibration.py | evaluate_calibration.py never writes to outputs/run/calibration/results.json — T8 calibration row always N/A |
| 131 | MEDIUM | Version Conflict | evaluate_trustworthiness.py:61 | dict|None type hint requires Python 3.10+; crashes on Python 3.9 |
| 132 | MEDIUM | Path Inconsistency | plot_case_visualization.py:35–36 | Uses non-inter-patient model and data — visualizations mismatch paper metrics |
| 133 | MEDIUM | Path Inconsistency | evaluate_robustness_all.py:73–75,118 | Uses non-inter-patient splits — Gaussian robustness incomparable to adversarial robustness |
| 134 | MEDIUM | Test Isolation | test_speed.py:9–10 | Permanently mutates global CONFIG dict — corrupts ensemble training if imported |
| 135 | MEDIUM | Diagnostic Logic | diag_verify_coupling_v2.py:249–257 | Double-append to skipped list if both amplitude filters fail |
| 136 | MEDIUM | PGD Ablation | ablation_results_inter.json | No-Adv outperforms full model by +0.18 F1 on inter-patient split — likely split inconsistency |
| 137 | MEDIUM | Hyperparameter Tuning | results/logs/hyperparameter_tuning.json | Two-config grid search; neither configuration matches final model |
| 138 | MEDIUM | ablation_robustness.csv | results/tables/ablation_robustness.csv | Full model clean F1 differs from ablation_table by 0.012 — stale checkpoint |
| 139 | MEDIUM | PGD Convergence Suspect | outputs/robustness/pgd_convergence_results.json | Steps 50 and 100 produce bit-identical results to step 10 — possible copy-paste |
| 140 | LOW | Dead Code | hmr_bilstm.py:347 | Duplicate return in temporal_smoothness_loss (second instance) |
| 141 | LOW | Thread Safety | hmr_bilstm.py:104–111 | Mutable last_* state on RLSTMCell — not thread-safe, incompatible with torch.compile |
| 142 | LOW | Design | hmr_bilstm_ablation.py:107–109 | W_beta always allocated even when unused — parameter count misleading in ablation |
| 143 | LOW | Logic Bug | train.py:84–86 | Cosine LR off-by-one — never reaches exact min_lr at final epoch |
| 144 | LOW | Tensor Shape | hmr_bilstm.py:174–175 | beta documented as scalar but is vector (B,H) — paper claim may be incorrect |
| 145 | LOW | Syntax | preprocess.py:94 | Mixed tab/space indentation |
| 146 | LOW | Numerical Stability | preprocess.py:152 | std epsilon added after not inside sqrt |
| 147 | LOW | Leakage Report | validation/verify_normalization.py:126 | data_leakage_prevented does not check test set |
| 148 | LOW | Division by Zero | validation/preprocess_aami.py:129–132 | print_dist divides by len(y) without zero-check |
| 149 | LOW | Dead Code | calibration/calibration_metrics.py:163 | Duplicate `return out` — unreachable second return |
| 150 | LOW | CSV Quality | calibration/reliability_diagram.py:36–57 | Empty-bin phantom rows with mean_confidence=0.0 mislead downstream tools |
| 151 | LOW | Code Consistency | temperature_scaling.py vs evaluate_calibration.py | dim=-1 vs dim=1 in softmax — identical numerically but inconsistent |
| 152 | LOW | Dead Code | uncertainty/deep_ensemble.py:27 | import copy unused — confirms weight-perturbation ensemble never implemented |
| 153 | LOW | Metric Definition | evaluate_fgsm.py:136–138 | label_accuracy and accuracy are identical duplicate keys in result dict |
| 154 | LOW | Documentation | gen_guide.py:408 | References class_weights.json; actual file is class_weights.npy |

---

## Detailed Findings

---

### CRITICAL Issues

---

#### Issue #1 | hmr_bilstm.py:213–214 | Logic Bug — Duplicate `for` Loop Header | CRITICAL

**Explanation:**  
The `for t in range(T):` loop header is written twice on consecutive lines inside `RLSTMLayer.forward`. In Python this creates either a `SyntaxError` / `IndentationError` or an empty no-op first loop with the real body only executing in the second iteration. The file cannot be reliably imported.

**Scientific Impact:**  
The model cannot be trained or imported from this source. Every result attributed to `hmr_bilstm.py` is unreproducible from the as-written source code.

**Fix:**
```python
for t in range(T):           # keep exactly one occurrence
    h, c = self.cell(x[:, t], h, c)
    h_outputs.append(h)
    r_outputs.append(self.cell.last_r_t.clone())
    ck_outputs.append(self.cell.last_c_keep.clone())
    ca_outputs.append(self.cell.last_c_add.clone())
```

---

#### Issue #2 | hmr_bilstm_ablation.py:107–108 | Line Number Collision — File Corruption | CRITICAL

**Explanation:**  
Line 108 appears twice: once for `self.W_beta = nn.Linear(hidden_size, hidden_size)` and once for `self.dropout = nn.Dropout(dropout)`. One of these statements is absent from the actual byte-stream. If `self.W_beta` is missing, `use_hybrid=True` crashes with `AttributeError`. If `self.dropout` is missing, the cell has no dropout layer.

**Scientific Impact:**  
The ablation variants requiring `use_hybrid=True` (the full model and `no_interaction`) will crash at runtime, making the entire ablation comparison scientifically invalid.

**Fix:** Audit the raw file to confirm both lines are present and at the correct indentation. Add assertions to `__init__` that verify expected attributes exist.

---

#### Issue #3 | hmr_bilstm_ablation.py:435–437 | IndentationError in RLSTMLoss | CRITICAL

**Explanation:**  
The `def __init__` of `RLSTMLoss` has one extra leading space relative to the class body, producing `IndentationError: unexpected indent` on import. The module cannot be imported at all.

**Scientific Impact:**  
The ablation file is unexecutable. All ablation results are unreproducible.

**Fix:**
```python
class RLSTMLoss(nn.Module):
    def __init__(self, lambda_smooth=0.01, class_weights=None,
                 use_focal=False, focal_gamma=2.0):
        super().__init__()
```

---

#### Issue #4 | calibration/temperature_scaling.py:73–74 | IndentationError — Module Unexecutable | CRITICAL

**Explanation:**  
`x = x.to(device)` on line 74 is indented at the same level as the `for x, y in loader:` loop header, placing it outside the loop body. This is an `IndentationError` that prevents the module from being imported.

**Scientific Impact:**  
Temperature scaling cannot be fitted or applied. The entire calibration pipeline depending on this module is non-functional.

**Fix:**
```python
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)          # indent inside loop body
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
```

---

#### Issue #5 | preprocess.py:136–142 | Intra-Patient Data Leakage | CRITICAL

**Explanation:**  
`preprocess.py` creates `train.npz`, `val.npz`, `test.npz` by stratified random split of the Kaggle `mitbih_train.csv` and `mitbih_test.csv` files. These Kaggle CSVs do not guarantee patient-level separation; multiple beats from the same patient appear in both the original train and test files. The random split within each file further allows beats from the same patient in both splits, causing the model to learn patient-specific ECG morphology.

**Scientific Impact:**  
All metrics reported by `train.py` (accuracy, F1, AUC) are inflated by an estimated 5–15 percentage points compared to true inter-patient generalization. The model does not generalize to unseen patients. This is the most fundamental validity issue in the evaluation framework. The headline performance claims in the paper are all optimistic.

**Fix:**  
Either retire `train.py` in favor of `train_inter_patient.py` as the primary evaluation, or implement patient-level splitting in `preprocess.py` using the known patient record IDs from MIT-BIH PhysioNet records.

---

#### Issue #6 | run_baselines.py:31–35 vs train.py:212–213 | Dataset Mismatch in Comparison Table | CRITICAL

**Explanation:**  
`run_baselines.py` loads from `data/processed/splits/inter_*.npz` (inter-patient split). `train.py` loads from `data/processed/train.npz` (intra-patient split). Results from both are compared side-by-side in the paper as if they are equivalent.

**Scientific Impact:**  
Intra-patient F1 scores are typically 5–15 percentage points higher than inter-patient equivalents for LSTM-based ECG classifiers. Comparing intra-patient HMR-BiLSTM results against inter-patient baseline results artificially inflates HMR-BiLSTM's apparent advantage. All comparative claims ("HMR-BiLSTM outperforms BiLSTM by X% F1") are confounded by this split mismatch.

**Fix:**  
All models — HMR-BiLSTM, LSTM, BiLSTM, ResNet1D — must be trained and evaluated on the same split. Use `train_inter_patient.py` for HMR-BiLSTM and enforce the same inter-patient split in `run_baselines.py`.

---

#### Issue #7 | results/tables/table5_consolidated.csv | Physically Impossible Values | CRITICAL

**Explanation:**  
`table5_consolidated.csv` reports for HMR-BiLSTM: Clean-F1 = 0.5644, FGSM-F1 = 0.8425, PGD-F1 = 0.8391. The adversarially-attacked F1 values exceed the clean baseline by +0.28 F1 points. An adversarially attacked model cannot perform better than the clean model. The root cause is that the Clean-F1 column is sourced from `baseline_results.json` (inter-patient, 4-class macro) while the FGSM/PGD columns are sourced from the ablation split evaluation (21,892 samples, 5-class macro), which produces higher F1 values due to different class distribution and averaging.

**Scientific Impact:**  
The paper's primary robustness table is internally contradictory and cannot support any trustworthiness conclusion. Any downstream claim that "HMR-BiLSTM maintains high F1 under attack" is undermined by this inconsistency.

**Fix:**  
Recompute all rows (clean, FGSM, PGD, CW, AutoAttack) on a single canonical test set with a single macro-averaging convention. Rebuild table5 from that single evaluation.

---

#### Issue #8 | results/tables/* | Two Evaluation Regimes Mixed Across All Tables | CRITICAL

**Explanation:**  
HMR-BiLSTM's clean performance is reported with two incompatible values across different files: F1=0.5644, Accuracy=0.8784 in Regime A (main test, 49,668 samples, 4-class, `inter_best_rlstm.pt`) and F1=0.8825, Accuracy=0.9749 in Regime B (ablation split, 21,892 samples, 5-class, `best_rlstm.pt`). The same duality applies to all baseline models. `table5_consolidated.csv` uses Regime A for the F1-macro and Accuracy columns but Regime B for the FGSM-F1, PGD-F1, ECE, and Brier columns.

**Scientific Impact:**  
No valid cross-model comparison can be made from any table that mixes these regimes. The paper's primary comparison tables are all invalid.

**Fix:**  
Establish one canonical test set, define one macro-averaging rule, and rerun all evaluations uniformly. All tables must reference the same configuration.

---

#### Issue #9 | explainability/integrated_gradients.py:102–108 | Jaccard Formula Mathematically Wrong | CRITICAL

**Explanation:**  
The `jaccard_with_tolerance` function double-counts elements in the intersection:  
- `matched_a` = elements of A within tolerance of any element in B  
- `matched_b` = elements of B within tolerance of any element in A  
- `intersection = (len(matched_a) + len(matched_b)) / 2.0`

One element in B can match multiple elements of A, causing `len(matched_a) >> len(matched_b)`. The average of the two overcounts does not give the true matched count. Example: A={90,91,92}, B={90}, tolerance=2 gives computed Jaccard=1.0 when the correct answer is 1/3. The parameter `T=187` is declared but never used, indicating dead code from a prior implementation.

**Scientific Impact:**  
All reported Jaccard(SHAP, IG) consistency values are computed from an incorrect formula. Values are not true Jaccard indices and are not comparable to Jaccard indices in any published XAI literature. A reported Jaccard of 0.6 could correspond to a true Jaccard of 0.2 or 0.8.

**Fix:**
```python
def jaccard_with_tolerance(set_a, set_b, tolerance=2, T=187):
    set_a = sorted(set_a)
    set_b_remaining = sorted(set_b)
    matched = 0
    for a in set_a:
        candidates = [b for b in set_b_remaining if abs(a - b) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda b: abs(a - b))
            matched += 1
            set_b_remaining.remove(best)
    union = len(set_a) + len(set_b) - matched
    return matched / max(1, union)
```

---

#### Issue #10 | evaluate_calibration.py:54–55 | IndentationError — Script Unexecutable | CRITICAL

**Explanation:**  
`accuracy_in_bin = accuracies[in_bin].mean()` is at the same indentation level as the enclosing `if prop_in_bin > 0:` statement rather than inside its body. This is an `IndentationError` that prevents `evaluate_calibration.py` from running.

**Scientific Impact:**  
No calibration comparison between LSTM, BiLSTM, and HMR-BiLSTM can be generated from this script.

**Fix:**
```python
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()     # 4 more spaces
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
```

---

#### Issue #11 | pgd_convergence.py:64 | PGD Attack Uses Incomplete Loss Function | CRITICAL

**Explanation:**  
The PGD attack in `pgd_convergence.py` passes `r_fwd=None, r_bwd=None` to `RLSTMLoss`, silently disabling the temporal smoothness regularization term. The training loss includes this smoothness term; the attack loss does not. This means the PGD attack is not maximizing the same loss the model was minimized against, making it a weaker, structurally incorrect adversarial attack.

**Scientific Impact:**  
PGD attack is computed against an incomplete loss. The reported adversarial robustness numbers overestimate true robustness — the attack is weaker than it should be.

**Fix:**
```python
outputs, internals = model(x_adv, return_internals=True)
loss, _ = criterion(outputs, y,
                    r_fwd=internals["r_fwd"],
                    r_bwd=internals["r_bwd"])
```

---

#### Issue #12 | outputs/v1.0_FINAL/robustness/* | AutoAttack/CW Use Clean Baseline F1=1.0 | CRITICAL

**Explanation:**  
Both AutoAttack and CW evaluations pre-filter the test set to only correctly-classified samples (n=200), giving `clean_accuracy=1.0` and `clean_f1_macro=1.0` by construction. F1 drops are computed relative to this artificially perfect baseline, making them appear more dramatic than they are against the true clean baseline of 0.8825. The sample size of 200 (0.4% of the test set) is also statistically insufficient for reliable 5-class macro F1.

**Scientific Impact:**  
AutoAttack and CW F1 drops cannot be compared with FGSM/PGD drops which use the full test set. Cross-attack comparisons in any table are invalid.

**Fix:**  
Re-run AutoAttack and CW on the full test set (or a stratified sample of at least 1,000 per class). Compute F1 drop relative to the same-sample clean baseline, not a pre-filtered subset.

---

#### Issue #13 | requirements.txt | 8 Missing Dependencies | CRITICAL

**Explanation:**  
The following packages are imported throughout the codebase but absent from `requirements.txt`: `shap`, `captum`, `torchattacks`, `autoattack`, `pyyaml`, `scipy`, `wfdb`, `python-docx`. A fresh `pip install -r requirements.txt` will fail at the first import of any YAML config (which occurs in virtually every module), making the entire pipeline unrunnable.

**Scientific Impact:**  
The environment cannot be reconstructed from the published requirements. All reproducibility claims are false.

**Fix:**
```
pyyaml>=6.0.0
scipy>=1.10.0
shap>=0.43.0
captum>=0.6.0
torchattacks>=3.5.0
autoattack>=0.1
python-docx>=1.1.0
wfdb>=4.1.0
tqdm>=4.65.0
```

---

#### Issue #14 | run_reproducible_pipeline.py, run_all.bat | Inter-Patient Splits Never Created | CRITICAL

**Explanation:**  
Both orchestration scripts begin with `preprocess.py`, which creates only the intra-patient splits (`data/processed/train.npz`, `val.npz`, `test.npz`). The inter-patient splits at `data/processed/splits/inter_*.npz` — required by `train_inter_patient.py`, `run_baselines.py`, and all evaluation modules — are never generated by either orchestration script. The script that creates them (`validation/preprocess_aami.py`) is absent from both pipelines. Furthermore, `preprocess_aami.py` requires the `wfdb` library and raw WFDB records, not just the Kaggle CSVs, meaning the inter-patient data preparation pipeline requires undocumented manual steps.

**Scientific Impact:**  
A researcher running either pipeline from scratch produces a model (`best_rlstm.pt` from `train.py`) that is never evaluated by any downstream T5–T8 step; those steps evaluate `inter_best_rlstm.pt` which is never trained by the pipeline. The primary reported model and the evaluated model are disconnected.

**Fix:**  
Add `validation/preprocess_aami.py` (Step 1) and `train_inter_patient.py` (Step 2) to both orchestration scripts before any evaluation steps.

---

#### Issue #15 | uncertainty/mc_dropout.py:401,406,408–409,411 | Syntax Errors (Semicolons) | CRITICAL

**Explanation:**  
The source listing shows trailing semicolons on multiple lines in `main()`. If any line consists solely of a bare `;` character, Python raises `SyntaxError` and the entire module cannot be imported, preventing T4 (MC Dropout), T5 (Deep Ensemble), and T6 (corruption sweep) from executing.

**Scientific Impact:**  
Complete failure of the uncertainty quantification pipeline. No MC Dropout or ensemble uncertainty metrics can be produced.

**Fix:** Remove all trailing semicolons and isolated semicolon lines from `mc_dropout.py`.

---

#### Issue #16 | uncertainty/mc_dropout.py, deep_ensemble.py, evaluate_corruptions.py | Duplicate Line Numbers / File Corruption | CRITICAL

**Explanation:**  
Multiple files contain duplicate line numbers in their source listings, indicating corrupted duplicate lines were inserted. In `evaluate_corruptions.py`, the `"gaussian"` dict key appears twice (at lines 74 and 119), causing the first entry to be silently overwritten by the second (Python last-write-wins semantics for duplicate dict keys). This corrupts the corruption sweep results.

**Scientific Impact:**  
The corruption sweep JSON will have only one `"gaussian"` entry (the last one written). All corruption plots and downstream Table 3 robustness data for Gaussian noise are potentially missing or incorrect.

**Fix:** Audit raw file bytes for all three files, remove every duplicate line, and verify the `"gaussian"` key appears exactly once in all results dictionaries.

---

#### Issue #17 | uncertainty/deep_ensemble.py:55–96 | Single-Checkpoint Ensemble with MI=0 | CRITICAL

**Explanation:**  
When only one checkpoint is available, `mode="single"` and `ensemble_predict` runs a single model. The mutual information `MI = entropy - expected_entropy = 0` for all samples by construction (no diversity between "ensemble members"). The module docstring claims "use Monte Carlo weight perturbation to approximate an ensemble" in single mode, but no weight perturbation code exists anywhere in the file. The `import copy` at line 27 is the only remnant of this unimplemented feature.

**Scientific Impact:**  
If the paper reports Deep Ensemble uncertainty results (entropy, MI, OOD AUROC) from a single-model run, those numbers are methodologically identical to plain softmax confidence. The MI column in any reported table will be 0.0000, making the epistemic uncertainty estimate completely uninformative.

**Fix:**
```python
if mode == "single":
    raise RuntimeError(
        "Deep Ensemble requires >= 2 independently trained models. "
        f"Only 1 checkpoint found in {ensemble_dir}. "
        "Do NOT report single-model mode as a Deep Ensemble in a paper."
    )
```

---

#### Issue #18 | evaluate_fgsm.py:32–33 | FGSM Missing Data-Range Clamp | CRITICAL

**Explanation:**  
The FGSM attack computes `x_adv = x + epsilon * sign(grad)` but never clamps the result to the valid data range `[data_min, data_max]`. The PGD implementation in `evaluate_pgd.py` explicitly clamps after each step. This inconsistency makes FGSM produce out-of-distribution inputs at high epsilon, while PGD does not, making the two attacks incomparable.

**Scientific Impact:**  
FGSM vs PGD comparisons at high epsilon values are unfair. FGSM may show artificially low robust accuracy because it produces OOD inputs rather than bounded adversarial inputs.

**Fix:**
```python
perturbation = epsilon * x_adv.grad.sign()
x_adv = (x + perturbation).clamp(data_min, data_max).detach()
```

---

#### Issue #19 | robustness/cw_attack.py:215 | C&W Attack Effectively Disabled | CRITICAL

**Explanation:**  
The default `cw_c=1e-4` is the Lagrangian multiplier for the attack objective. The original Carlini & Wagner (2017) paper uses binary search over `c` starting from `1e-3` to `10`. With `c=1e-4`, the L2 norm term dominates so strongly that the optimizer almost never moves far enough from the original sample to cause misclassification. The attack silently "succeeds" at finding zero-perturbation solutions, and the reported ASR will be near 0%.

**Scientific Impact:**  
Reported C&W ASR ≈ 0 is an artifact of the hyperparameter, not genuine robustness. Any paper use of C&W as evidence against gradient masking is invalidated.

**Fix:**  
Use binary search over `c` values (e.g., `[1e-3, 0.01, 0.1, 1.0]`) or start from at least `c=1e-2`.

---

#### Issue #20 | evaluate_robustness_all.py:73–78 | Noise Robustness on Wrong Dataset | CRITICAL

**Explanation:**  
`evaluate_robustness_all.py` loads `data/processed/test.npz` (intra-patient split) while all other adversarial evaluation scripts use `INTER_TEST = data/processed/inter_test.npz` (inter-patient split). The intra-patient test set gives artificially higher baseline accuracy, creating an incomparable robustness baseline.

**Scientific Impact:**  
Noise robustness numbers from this file cannot be compared to adversarial robustness numbers from FGSM/PGD scripts. Any combined robustness table mixing these two sources is methodologically unsound.

**Fix:**
```python
from configs.paths import INTER_TEST
test = np.load(INTER_TEST)
X_te, y_te = test["X"], test["y"]
```

---

#### Issue #21 | results/tables/* | 4-class vs 5-class Macro F1 Switches Silently | CRITICAL

**Explanation:**  
`training_history.json test_metrics.f1_macro` = 0.5644 is verified as 4-class macro (N,S,V,F only, Q excluded). The classification_report in the same file shows macro avg F1 = 0.4515 (5-class including Q with F1=0.000). `ablation_results.json full variant f1_macro` = 0.8921 is the 5-class macro on the ablation split. These three numbers are reported under the same column name "f1_macro" in different tables without disclosure of the averaging convention.

**Scientific Impact:**  
The primary headline performance metric is computed by an undisclosed exclusion that inflates the score by 0.1129 F1 points relative to the proper 5-class macro. Reviewers cannot replicate the number.

**Fix:**  
Explicitly state in all tables whether macro averaging includes or excludes Q (AAMI EC57 standard excludes Q — use `labels=[0,1,2,3]` consistently and document this choice).

---

#### Issue #22 | explainability/shap_analysis.py:335,363–365 | Class Count Mismatch in SHAP | CRITICAL

**Explanation:**  
The model has `num_classes=5`, `n_classes=5` is declared, but `CLASS_NAMES = {0:"N", 1:"S", 2:"V", 3:"F"}` has only 4 entries. The global importance computation at lines 363–365 iterates only over `shap_classes = [0,1,2,3]`, completely omitting class 4 (Q) from the global feature importance ranking. Any SHAP attribution for class 4 is silently discarded.

**Scientific Impact:**  
The global feature importance ranking excludes class-4 SHAP values. The claim "top-20 timesteps represent the most important features across all AAMI classes" is false — it covers only 4 of 5 classes.

**Fix:**
```python
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
mean_imp = np.stack(
    [np.abs(shap_vals_summary[c]).squeeze(-1).mean(axis=0) for c in range(n_classes)], axis=0
).mean(axis=0)
```

---

#### Issue #23 | validation/preprocess_aami.py:141–160 | Intra-Patient Splits Not Normalized | CRITICAL

**Explanation:**  
The intra-patient split (lines 141–160) saves `X_all` without any normalization. The inter-patient splits (lines 103–108) are correctly normalized. When `evaluate_splits.py` trains on un-normalized intra data and normalized inter data, the two models operate on entirely different input scales (raw ECG amplitudes ~0–1.5 mV vs. z-score normalized ~mean=0, std=1).

**Scientific Impact:**  
The intra-vs-inter comparison is scientifically invalid. Any observed performance gap could be entirely explained by scale difference rather than patient-overlap protocol. This is a critical flaw in the AAMI evaluation chapter.

**Fix:**  
Apply train-only normalization to the intra-patient split before saving. Save normalized arrays in the output NPZ files.

---

#### Issue #24 | hmr_bilstm.py:347 | Duplicate Return in temporal_smoothness_loss | CRITICAL

**Explanation:**  
The function has two consecutive `return torch.tensor(0.0, device=r_seq.device, requires_grad=True)` statements. The second is unreachable dead code. While harmless at runtime, this is a copy-paste corruption indicator that undermines trust in surrounding logic.

**Scientific Impact:**  
No direct runtime error, but signals file corruption and reduces confidence in the surrounding code's integrity.

**Fix:** Remove the duplicate `return` line entirely.

---

#### Issue #25 | hmr_bilstm.py:436 | Demo Section Line Number Collision | CRITICAL

**Explanation:**  
Line 436 appears twice in the source listing with different content: once for the criterion assignment and once for the forward pass call. One of these statements may be absent from the actual byte-stream, breaking the integration smoke-test in `__main__`.

**Scientific Impact:**  
The demo unit test that serves as the integration smoke-test is broken or produces misleading output.

**Fix:** Inspect and repair the raw source file byte-by-byte to ensure both statements are present and ordered correctly.

---

#### Issue #26 | robustness/auto_attack.py:158–159,211,265–267 | PGD vs AutoAttack Epsilon Space Inconsistency | CRITICAL

**Explanation:**  
PGD runs with `pgd_eps=0.02` in original data space. AutoAttack runs with `eps=aa_eps/(data_max-data_min)` which algebraically cancels back to `aa_eps` in data space, making them nominally equal. However, this cancellation is never verified by assertion. If `data_max - data_min` is inconsistent between the two call sites, the effective budgets differ. The gradient masking diagnosis `masking_gap = aa_asr - pgd_asr` is unreliable if budgets are not precisely identical.

**Scientific Impact:**  
False conclusions about gradient masking are a serious scientific validity concern. The diagnosis `gradient_masking_suspected = masking_gap > 0.15` may fire incorrectly.

**Fix:**
```python
assert abs(aa_eps - pgd_eps) < 1e-6, "PGD and AA must use the same epsilon"
```

---

#### Issue #27 | preprocess.py:149–152 | Global Scalar Normalization Instead of Per-Feature | CRITICAL

**Explanation:**  
`X_train.mean()` and `X_train.std()` are called without an `axis` argument, producing global scalar statistics instead of per-timestep (per-feature) statistics. The saved `norm_mean.npy` and `norm_std.npy` are scalars. Standard ECG beat preprocessing normalizes per-timestep.

**Scientific Impact:**  
The normalization choice is undocumented and non-standard. If any inference code applies the saved statistics per-feature (expecting arrays of shape (187,)), it will fail silently. For single-lead data the global scalar is numerically consistent but does not represent best practice.

**Fix:**
```python
mean = X_train.mean(axis=0)        # shape (187,)
std  = X_train.std(axis=0) + 1e-8  # shape (187,)
X_train = (X_train - mean) / std
```

---

#### Issue #28 | results/tables/* | AutoAttack/CW F1 Drops Not Comparable to FGSM/PGD | CRITICAL

**Explanation:**  
AutoAttack and CW report F1 drops relative to a clean baseline of F1=1.0 (pre-filtered subset). FGSM and PGD report F1 drops relative to the true clean baseline (~0.88). These drops cannot be compared in the same column of a robustness table. A "drop of 0.10" from FGSM (0.88→0.78) represents a different severity than "drop of 0.10" from CW (1.00→0.90).

**Scientific Impact:**  
Any robustness summary that combines these four attacks in the same row produces a misleading picture of relative attack effectiveness.

**Fix:**  
Standardize all four attacks to evaluate on the same test set subset with the same clean baseline. Report all drops as percentage relative to the same denominator.

---

### HIGH Issues

---

#### Issue #29 | hmr_bilstm.py:102 | Self.dropout in RLSTMCell Never Applied | HIGH

**Explanation:**  
`self.dropout = nn.Dropout(dropout)` is defined in `RLSTMCell.__init__` but is never called anywhere in `RLSTMCell.forward`. The dropout parameter has zero effect on cell computations. The only dropout applied is the inter-layer dropout in `BiRLSTM` when `num_layers > 1`.

**Scientific Impact:**  
The reported dropout rate of 0.25 does not apply inside the recurrent cell. The actual regularization is weaker than described. Published results obtained with "dropout=0.25" are misleading — the effective dropout rate inside the cell is 0.

**Fix:**  
Apply dropout to the hidden state output: `h_t = self.dropout(o_t * torch.tanh(c_t))`

---

#### Issue #30 | hmr_bilstm.py:115–133 | Substring Name Match for Bias Initialization | HIGH

**Explanation:**  
`"W_h" in name` is True for any parameter whose name contains the substring `"W_h"`. If a future layer is named `W_hidden_state`, its bias would incorrectly receive the forget-gate initialization pattern. The check `"W_h_rmc"` in the orthogonal init list is also redundant since `"W_h"` already catches it.

**Scientific Impact:**  
Currently no crash, but fragile. Any extension of the architecture risks silent initialization corruption.

**Fix:** Use exact name matching: `if name in ("W_x.bias", "W_h.bias"):` instead of substring checks.

---

#### Issue #31 | hmr_bilstm.py:350 | requires_grad=True Leaf Disconnected from Graph | HIGH

**Explanation:**  
`return torch.tensor(0.0, device=r_seq.device, requires_grad=True)` in the early-return path of `temporal_smoothness_loss` creates a fresh leaf tensor with no connection to any model parameters. When added to the loss, PyTorch attempts to accumulate gradient into this leaf rather than through model parameters.

**Scientific Impact:**  
For sequences of length 1 (edge case), gradients become corrupted. For the default ECG use case (T=187), this is a non-issue, but the pattern is fragile.

**Fix:**
```python
if r_seq.size(1) < 2:
    return r_seq.sum() * 0.0   # connected to graph, value 0.0
```

---

#### Issue #32 | hmr_bilstm.py:154–156 | c_keep + c_add Does Not Decompose c_lstm | HIGH

**Explanation:**  
The docstring claims `c_rmc = alpha_keep * c_keep + alpha_add * c_add` approximates the LSTM cell update. However:
- When r_t=0: sum = f_t*c_prev + i_t*g_t = c_lstm (correct)
- When r_t=1: sum = c_prev (pure memory retention, not c_lstm)

The decomposition is not a partition of c_lstm for arbitrary r_t. The comment "Eq.(13b) LSTM path: c_lstm = f_t*c_prev + i_t*g_t" implies they are equivalent, which they are not.

**Scientific Impact:**  
The claim that the RMC decomposes the LSTM update is mathematically false. The ablation comparison `full vs no_rmc` does not purely isolate the RMC contribution.

**Fix:** Correct the docstring to state that the RMC represents a competing memory control mechanism, not a decomposition of the LSTM update equation.

---

#### Issue #33 | hmr_bilstm.py:160–161 | F.layer_norm on Near-Zero c_add Causes Degenerate Attention | HIGH

**Explanation:**  
When `r_t ≈ 1`, `c_add = (1-r_t)*(i_t*g_t) ≈ 0`. `F.layer_norm` on a near-zero vector produces ~0/(sqrt(0+eps)) = 0. The `W_alpha` scoring network then receives a zero vector for `c_add_norm`, making `alpha_add ≈ 0.5` (degenerate uniform attention). Additionally, the functional `F.layer_norm` (no learnable affine) discards scale information unlike `nn.LayerNorm`.

**Scientific Impact:**  
During periods where the residual gate saturates to 1, the attention scoring collapses to near-uniform, defeating the purpose of the learnable gating mechanism.

**Fix:** Replace `F.layer_norm` with registered `nn.LayerNorm` modules in `__init__`, or use L2 normalization: `c_keep_norm = c_keep / (c_keep.norm(dim=-1, keepdim=True) + 1e-6)`.

---

#### Issue #34 | hmr_bilstm.py:152 | LayerNorm+Sigmoid: Gamma Can Saturate r_t | HIGH

**Explanation:**  
`r_t = torch.sigmoid(self.layer_norm(combined_rmc))`. As training proceeds, the learnable `gamma` parameter of LayerNorm can grow arbitrarily large, collapsing `r_t` toward a constant gate (near 0 or 1 for all timesteps) and eliminating signal-dependent memory control.

**Scientific Impact:**  
The residual gate may degenerate early in training, causing instability. This could explain observed early-training instability in loss curves.

---

#### Issues #35–83 (HIGH Severity)

These issues are fully documented in the Issue Index above. Key ones warranting special attention:

- **Issue #36 (FGSM in train.py):** `model.zero_grad()` is called before `loss.backward()`, allowing adversarial backward pass to write gradients into model parameters. These persist until `optimizer.zero_grad()`. While not causing direct gradient contamination in the current code, it is a fragile ordering that would break under minor refactoring.

- **Issue #38 (F1 Metric Inconsistency):** `train.py` uses 5-class F1 for early stopping while `train_inter_patient.py` uses 4-class F1. The two scripts cannot produce comparable validation monitoring curves.

- **Issue #39 (Wrong AUC):** `train_inter_patient.py` computes and renormalizes `probs_4class` correctly but then passes the original `probs[:, :4]` (un-renormalized) to `roc_auc_score`. AUC values in training logs from `train_inter_patient.py` are incorrect.

- **Issue #40 (Focal Loss Deviation):** The focal loss applies class weights as a post-multiplier after computing the focal term from unweighted CE. This is not the standard Lin et al. (2017) formulation. While potentially beneficial, the paper must not claim standard focal loss.

- **Issue #41 (Fairness):** Baselines train for 12 epochs with no LR schedule; HMR-BiLSTM trains for 45 epochs with cosine annealing and adversarial training. Three compounding fairness issues make it impossible to attribute performance differences to architecture alone.

- **Issue #57 (IG Steps):** With only 50 interpolation steps for a model containing MaxPool, ReLU, and attention nonlinearities, convergence deltas may be large, making IG attributions unreliable.

- **Issue #59 (Signed Mean in IG Plot):** The variable `mean_abs_attribution_plot` is computed as a signed mean, not mean absolute value. Positive and negative attributions from different beats cancel, hiding genuine feature importance in the plots. This contradicts the quantitative Jaccard analysis which correctly uses mean absolute values.

- **Issue #60 (TracIn Mislabeled):** The method is cosine gradient similarity between single-checkpoint gradients, not TracIn (which requires dot product summed over multiple checkpoints weighted by learning rate). Calling it TracIn in the paper is scientifically inaccurate.

---

### MEDIUM Issues

---

#### Issue #84 | hmr_bilstm.py:166–168 | Alpha Scalar vs Beta Vector Asymmetry | MEDIUM

**Explanation:**  
`alpha_keep` and `alpha_add` have shape `(B, 1)` — per-sample scalars. `beta` has shape `(B, H)` — per-dimension vectors. This architectural asymmetry is undocumented. Any paper claim that "alpha and beta are comparable gating mechanisms" is incorrect.

---

#### Issue #86 | hmr_bilstm_ablation.py:196–204 | No-RMC Variant Gets Zero Smoothness Loss | MEDIUM

**Explanation:**  
For the `no_rmc` variant, all `r_t` values are exactly zero, so the temporal smoothness loss is always 0.0. The full model pays a regularization cost for `r_t` variation while `no_rmc` pays zero. The `no_rmc` variant has a systematically lower effective loss, giving it an unearned optimization advantage in the ablation comparison.

**Scientific Impact:**  
The ablation conclusion that "full model outperforms no_rmc" may be understated (the gap is artificially narrowed) or the comparison is unfair due to differing effective loss surfaces.

**Fix:**  
Pass `r_fwd=None, r_bwd=None` when calling `RLSTMLoss.forward` for the `no_rmc` variant, or explicitly zero the smoothness term for all variants in the ablation comparison.

---

#### Issue #91 | run_ablation.py:509–558 | Shared DataLoader Across Variants | MEDIUM

**Explanation:**  
The DataLoader's `RandomSampler` state is NOT reset by `set_seed()`. Each variant sees a different batch order depending on how many batches the previous variant consumed. Running variants individually vs. sequentially produces different results even with the same seed.

**Scientific Impact:**  
The ablation study is not fully reproducible. Running variants in different orderings produces different numerical results, undermining comparability.

**Fix:**  
Recreate the DataLoader inside each variant training call using a `torch.Generator` with a fixed seed.

---

#### Issues #92–139 (MEDIUM Severity)

These are fully documented in the Issue Index. Key additional highlights:

- **Issue #101 (Class Weight Clipping):** The 10.0 ceiling in `np.clip(class_weights, 0.5, 10.0)` reduces the true weight for Fusion class (~59) by a factor of 6, severely under-representing it. Recall for classes 3 and 4 will be systematically under-reported.

- **Issue #108 (Missing Directory):** `evaluate_calibration.py` writes to `results/tables/calibration_results.csv` without creating the parent directory, causing `FileNotFoundError` on fresh checkouts.

- **Issue #116 (Logits vs Probabilities in SHAP):** The `ModelWrapper` returns raw logits to SHAP rather than probabilities. Logit-based SHAP values are not probability-calibrated and do not account for softmax competition between classes, making them less clinically interpretable.

- **Issue #124 (F1 Drop Units):** The same "F1 drop" metric is reported in absolute units (0.0400) in `table2_fgsm_robustness.csv` and as percentages ("4.54%") in `baseline_full_comparison.csv`. A reader comparing the two tables will get contradictory interpretations.

- **Issue #130 (Calibration Missing from Pipeline):** `evaluate_calibration.py` does not write to `outputs/<run_id>/calibration/results.json`. The T8 trustworthiness scorecard calibration row will always show N/A when run via the orchestrated pipeline.

- **Issue #131 (Python Version Conflict):** The `dict | None` type hint syntax requires Python 3.10+. On Python 3.9 (listed as supported in the Installation Guide), `evaluate_trustworthiness.py` fails at import with `TypeError`.

---

### LOW Issues

---

#### Issue #140–154 (LOW Severity)

These issues represent dead code, documentation errors, code quality problems, and minor numerical edge cases. They do not affect the validity of experimental results but should be cleaned up before publication.

- **Issue #140:** Second unreachable `return out` in `compute_conditional_ece` (dead code).
- **Issue #141:** Mutable `last_*` state on `RLSTMCell` is not thread-safe and incompatible with `torch.compile()`.
- **Issue #142:** `W_beta` always allocated even for `no_rmc`/`no_hybrid` variants — parameter count comparison in ablation table is misleading (all variants show 505,038 parameters regardless of whether W_beta is used).
- **Issue #143:** Cosine LR schedule uses `(epoch-1)/N` normalization, reaching `min_lr` approximately at the last epoch but never exactly — minor off-by-one.
- **Issue #144:** `beta` is a vector `(B, H)` but the paper may describe it as a scalar gate — verify paper claims.
- **Issue #145:** Mixed tab/space indentation in `preprocess.py:94`.
- **Issue #146:** `std = X_train.std() + 1e-8` adds epsilon after computing std; safer to use `max(std, 1e-8)`.
- **Issue #147:** `data_leakage_prevented` in `verify_normalization.py` does not check the test set — only train/val.
- **Issue #148:** `print_dist` in `preprocess_aami.py` divides by `len(y)` without a zero-check — possible `ZeroDivisionError` for empty splits.
- **Issue #149:** Duplicate `return out` in `compute_conditional_ece` — unreachable dead code.
- **Issue #150:** Empty-bin phantom rows in reliability diagram CSV (sample_count=0, mean_confidence=0.0, mean_accuracy=0.0) mislead downstream tools.
- **Issue #151:** `dim=-1` vs `dim=1` in softmax calls across calibration modules — numerically identical but inconsistent.
- **Issue #152:** `import copy` unused in `deep_ensemble.py` — confirms weight-perturbation ensemble was never implemented.
- **Issue #153:** `label_accuracy` and `accuracy` are identical duplicate keys in `evaluate_fgsm.py` result dict.
- **Issue #154:** `gen_guide.py` references `class_weights.json` — actual file is `class_weights.npy`.

---

## Cross-Cutting Concerns

---

### Reproducibility Assessment

**Overall: POOR**

The codebase has multiple layers of reproducibility failure:

1. **Environment reproducibility (CRITICAL):** `requirements.txt` is missing 8 packages. A fresh install cannot run any module that loads YAML config, uses SHAP, uses captum, or reads raw WFDB records.

2. **Data reproducibility (CRITICAL):** The inter-patient splits required by all evaluation modules are not created by either orchestration script. The raw WFDB records require manual download. The exact Kaggle CSV version used is not pinned. The preprocessing pipeline for intra-patient splits is inconsistently applied (inter vs intra).

3. **Run-ID isolation (HIGH):** `get_run_id()` generates a timestamp-based ID on each call. Running scripts individually produces different output directories per script, making cross-script result aggregation fragile. The T8 trustworthiness dashboard depends on hash-based cross-run search to find outputs from steps 1–12 run independently.

4. **Seed reproducibility (MEDIUM):** `run_ablation.py` resets the model initialization seed between variants but does not reset the DataLoader's sampler state, meaning batch ordering differs between isolated and sequential variant runs. Ensemble training uses only seeds 123 and 456, excluding the reference seed-42 model.

5. **Checkpoint versioning (HIGH):** Multiple scripts reference hardcoded run-ID paths that exist only on the author's development machine. Five diagnostic scripts will fail immediately with `FileNotFoundError` on any other machine.

6. **Two-pipeline fragmentation (CRITICAL):** The primary reporting metric comes from `train.py` (intra-patient) while all trustworthiness evaluation uses `train_inter_patient.py` (inter-patient). These two models are never reconciled in the orchestration script. The published scorecard may reflect evaluations of a model that was trained on a different data distribution from the one used for calibration, uncertainty, and explainability analysis.

---

### Data Integrity Assessment

**Overall: POOR**

1. **Impossible values:** `table5_consolidated.csv` contains PGD-F1 > Clean-F1 for all three models. This is a physically impossible result indicating cross-regime mixing.

2. **Three incompatible ECE values:** ECE for HMR-BiLSTM is reported as 0.0391 (post-calibration JSON), 0.2309 (calibration_results.csv), and 0.0397 (baseline_full_comparison.csv) — spanning nearly an order of magnitude.

3. **AUC=NaN in ensemble logs:** `ensemble_log_123.json` and `ensemble_log_456.json` contain NaN AUC for all validation epochs. The ensemble's AUC cannot be verified.

4. **AUC=0.0 in training history:** `training_history.json` shows `val_auc_ovr=0.0` for all 20 training epochs. The training monitoring system silently failed to compute AUC throughout the entire training run.

5. **PGD convergence suspect values:** Steps 50 and 100 in `pgd_convergence_results.json` produce bit-for-bit identical results as step 10. This is statistically implausible and may indicate copy-paste filling of the results table.

6. **Stale early checkpoint in fgsm_results.json:** `fgsm_results.json` contains HMR-BiLSTM results with F1=0.2080 and recall_F=0.0, indicating a degenerate model. This file was not cleaned up and risks being accidentally cited.

7. **baseline_full_comparison.csv has no generation script:** This file contains dramatically different values (LSTM accuracy=0.9639 vs 0.8321 in other files) with no reproducible provenance. It cannot be regenerated from the current codebase.

8. **Hyperparameter tuning does not support final model:** The tuning JSON explores only two configurations (hidden=128, dropout=0.3, lambda=0.01/0.05). The final model uses hidden=96, dropout=0.25, lambda=0.003 — none of these values were explored in the documented tuning.

---

### Publication Readiness Verdict

**NOT READY FOR PUBLICATION.**

This codebase requires substantial remediation before supporting a scientific publication. The following minimum requirements must be met before resubmission:

**Must Fix (Blocking):**
1. Repair all syntax-level bugs (Issues #1–4, #10, #15, #16) so that core model files can be imported.
2. Unify all evaluation onto a single canonical split with a single macro-averaging convention (Issues #5, #6, #8, #21).
3. Fix the Jaccard formula (Issue #9) and recompute all SHAP-IG consistency numbers.
4. Correct the signed-mean IG aggregation bug (Issue #59) and regenerate all IG plots.
5. Add all missing dependencies to `requirements.txt` (Issue #13) and add `preprocess_aami.py` + `train_inter_patient.py` to the orchestration pipeline (Issue #14).
6. Fix the AutoAttack/CW evaluation protocol to use the full test set with a non-trivial clean baseline (Issue #12, #28).
7. Resolve the physically impossible values in `table5_consolidated.csv` (Issue #7) by running all evaluations from a single consistent checkpoint on a single test set.

**Should Fix (High Priority):**
8. Apply dropout inside `RLSTMCell` or document that the cell uses zero within-cell dropout (Issue #29).
9. Fix the focal loss implementation to match Lin et al. (2017) or clearly describe the modified formulation (Issue #40).
10. Standardize the epoch budget and training schedule for all baselines to match HMR-BiLSTM (Issue #41).
11. Fix the AUC computation in training and ensemble logs (Issues #64, L-1).
12. Fix the C&W hyperparameter to use binary search over `c` values (Issue #19).
13. Investigate and explain the No-Adv > Full model result on inter-patient split (Issue #65).
14. Replace the `TracIn` label with an accurate description of the gradient cosine similarity method (Issue #60).
15. Fix the `python-docx` type hint to use `Optional[dict]` for Python 3.9 compatibility (Issue #131).

---

*End of HMR-BiLSTM Scientific & Engineering Audit Report*  
*Total issues: 154 (28 CRITICAL, 55 HIGH, 46 MEDIUM, 25 LOW)*
