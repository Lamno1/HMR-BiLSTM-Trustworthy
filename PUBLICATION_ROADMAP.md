# Publication Roadmap — HMR-BiLSTM

**Source:** `FINAL_PUBLICATION_READINESS_REPORT.md`  
**Date:** 2026-06-23  
**Total remaining issues:** 138 across 6 phases  
**Estimated calendar time:** 8–10 weeks (single developer with GPU access)

---

## Sequencing Rationale

```
Phase 1 ──► Phase 4 ──► Phase 5 ──► Phase 6
   │                        ▲
   └──► Phase 2 ────────────┤
Phase 3 ──────────────► Phase 4
```

**Phase 1** and **Phase 2** can be worked in parallel.  
**Phase 3** decisions must be recorded before Phase 4 begins.  
**Phase 4** depends on Phase 1 + Phase 3 both being complete.  
**Phase 5** depends on Phase 2 + Phase 4 both being complete.  
**Phase 6** depends on Phase 5.

The single hard constraint is: **do not run any training (Phase 4) until Phase 1 and Phase 3 are complete.** Training on broken code or without a unified protocol wastes GPU time and produces results that must be discarded.

---

## Phase 1 — Must Fix Before Any Retraining

**Goal:** Correct every line of code that executes during training, preprocessing, and
loss computation. After this phase, running Phase 4 will produce scientifically valid
checkpoints.

**Duration estimate:** 5–7 days  
**Issue count:** 25  
**Parallelisable with:** Phase 2, Phase 3

---

### 1A — Core model and loss (hmr_bilstm.py, hmr_bilstm_ablation.py)
*Fix these before any model training run. They change what model architecture gets trained.*

| # | Sev | File | Fix Required |
|---|---|---|---|
| 29 | HIGH | `hmr_bilstm.py:102` | Apply `self.dropout` inside `RLSTMCell.forward`, or add an explicit comment stating within-cell dropout is intentionally zero |
| 31 | HIGH | `hmr_bilstm.py:350` | Replace disconnected leaf `torch.tensor(0.0, requires_grad=True)` with `r_seq.sum() * 0.0` |
| 33 | HIGH | `hmr_bilstm.py:160–161` | Replace `F.layer_norm` with a registered `nn.LayerNorm` module to avoid degenerate attention when `c_add ≈ 0` |
| 34 | HIGH | `hmr_bilstm.py:152` | Add `gamma` clamping on the `nn.LayerNorm` in RMC path to prevent `r_t` saturation |
| 85 | MEDIUM | `hmr_bilstm.py:408–416` | Remove `.detach()` from `r_fwd`/`r_bwd` dict values in `RLSTMLoss.forward` return |
| 86 | MEDIUM | `hmr_bilstm_ablation.py:196–204` | Pass `r_fwd=None, r_bwd=None` explicitly for `no_rmc` variant so smoothness loss is consistently zeroed; do not give `no_rmc` a free loss reduction |
| 88 | MEDIUM | `hmr_bilstm.py:118–119` | Replace substring init match with exact parameter name list |
| 30 | HIGH | `hmr_bilstm.py:115–133` | Replace `"W_h" in name` substring check with an explicit tuple of exact parameter names |

**Effort:** 2–3 days

---

### 1B — Training script fixes (train.py, train_inter_patient.py, run_ablation*.py, train_ensemble.py)
*Fix these before running any training loop. They change model checkpoint selection and training dynamics.*

| # | Sev | File | Fix Required |
|---|---|---|---|
| 36 | HIGH | `train.py:128–145` | Move `model.zero_grad()` to after `loss.backward()` in the FGSM adversarial training block |
| 38 | HIGH | `train.py:105–110` vs `train_inter_patient.py:106–110` | Unify early stopping to use 4-class macro F1 (`labels=[0,1,2,3]`) in both scripts |
| 40 | HIGH | `hmr_bilstm.py:374–391` | Decision: either fix alpha to be applied before the focal multiplier (standard Lin et al. formulation), or keep the modified formulation and add a docstring documenting the deviation |
| 73 | HIGH | `run_baselines.py:563–570` | Load `inter_class_weights.npy` (not intra-patient) for inter-patient baseline training |
| 89 | MEDIUM | `train.py:274–277` | Fix cosine LR scheduler: use `epoch / (T-1)` so final epoch reaches exactly `min_lr` |
| 91 | MEDIUM | `run_ablation.py:509–558` | Recreate the `DataLoader` inside each variant loop using `torch.Generator(seed=42)` |
| 94 | MEDIUM | `train.py:300–310` | Change `best_f1 = 0.0` initialisation to `-1.0` so epoch-1 checkpoint always saved |
| 143 | LOW | `train.py:84–86` | Fix the same cosine LR off-by-one in the secondary scheduler |

**Effort:** 1–2 days

---

### 1C — Preprocessing pipeline (preprocess.py, validation/preprocess_aami.py)
*Fix before re-running data generation in Phase 3/4. These bugs affect what training data looks like.*

| # | Sev | File | Fix Required |
|---|---|---|---|
| 23 | CRITICAL | `validation/preprocess_aami.py:141–160` | Apply the same train-only z-score normalization to the intra-patient split before saving |
| 27 | CRITICAL | `preprocess.py:149–152` | Change `X_train.mean()` and `X_train.std()` to `X_train.mean(axis=0)` and `X_train.std(axis=0)` for per-feature statistics |
| 42 | HIGH | `validation/preprocess_aami.py:46–59` | Use a single consistent `float(fs)` throughout; avoid `int(fs)` for resampling |
| 43 | HIGH | `validation/preprocess_aami.py:63–65` | Change strict `<` to `<=` in boundary guard to stop dropping valid beats at segment edges |
| 99 | MEDIUM | `validation/preprocess_aami.py:74–76` | Save metadata as a structured numpy array or JSON, not a numpy object array |
| 100 | MEDIUM | `validation/preprocess_aami.py:141–160` | Replace random shuffle with stratified sampling to preserve class distribution in intra-patient split |
| 101 | MEDIUM | `preprocess.py:216` | Raise or remove the class weight cap: the true weight for minority classes is ~59; cap 10.0 reduces it 6× |
| 146 | LOW | `preprocess.py:152` | Change `X_train.std() + 1e-8` to `np.maximum(X_train.std(axis=0), 1e-8)` |

**Effort:** 2–3 days

---

### Phase 1 Exit Condition

All of the following must be true before Phase 4 begins:

- [ ] `python -c "from hmr_bilstm import RLSTMClassifier"` runs without error
- [ ] `python -c "from hmr_bilstm_ablation import RLSTMClassifier"` runs without error
- [ ] `self.dropout` is either called in `RLSTMCell.forward` or documented as intentionally zero
- [ ] `train.py` and `train_inter_patient.py` both use `labels=[0,1,2,3]` for early-stop F1
- [ ] `model.zero_grad()` appears after `loss.backward()` in `train.py`
- [ ] `best_f1` initialised to `-1.0`
- [ ] `preprocess.py` uses per-feature mean/std (shape `(187,)`)
- [ ] `preprocess_aami.py` saves normalised intra-patient splits
- [ ] `preprocess_aami.py` uses `<=` boundary guard and float fs throughout
- [ ] Class weight cap raised or removed in `preprocess.py`

---

## Phase 2 — Safe Automatic Code Fixes

**Goal:** Patch all evaluation, analysis, calibration, uncertainty, explainability, robustness,
diagnostic, and utility scripts. These changes do not affect any trained model — they
only affect how the model is evaluated and how results are displayed.

**Duration estimate:** 3–4 days  
**Issue count:** 67  
**Parallelisable with:** Phase 1, Phase 3  
**Note:** Can be done by a second team member concurrently with Phase 1.

---

### 2A — Pipeline and orchestration (5 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 14 | CRITICAL | `run_reproducible_pipeline.py`, `run_all.bat` | Insert `validation/preprocess_aami.py` (Step 0) and `train_inter_patient.py` (Step 1) as the first two steps in both orchestration scripts |
| 25 | CRITICAL | `hmr_bilstm.py:436` | Inspect demo `__main__` section byte-by-byte; verify both `criterion` assignment and forward pass call are present |
| 127 | MEDIUM | `run_reproducible_pipeline.py` | Add `evaluate_autoattack.py` to the Python orchestrator (already present in `run_all.bat`) |
| 128 | MEDIUM | `evaluate_trustworthiness.py:287` | Fix T8 to read from the canonical robustness JSON path that `compare_fgsm_baselines.py` does not delete |
| 131 | MEDIUM | `evaluate_trustworthiness.py:61` | Change `dict\|None` to `Optional[dict]` for Python 3.9 compatibility |

---

### 2B — Robustness evaluation (8 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 11 | CRITICAL | `pgd_convergence.py:64` | Pass `r_fwd=internals["r_fwd"], r_bwd=internals["r_bwd"]` to `RLSTMLoss` in the PGD attack inner loop |
| 26 | CRITICAL | `robustness/auto_attack.py:211,265–267` | Add `assert abs(aa_eps_normalized - pgd_eps) < 1e-6` before computing gradient masking gap |
| 67 | HIGH | `evaluate_pgd.py:47–48` | Compute `data_min`, `data_max` from the full test set once before the attack loop, not per-batch |
| 68 | HIGH | `robustness/auto_attack.py:131–134` | Remove the duplicate for-loop that runs `steps` iterations of pure dead overhead |
| 69 | HIGH | `evaluate_autoattack.py:144–151` | Compute `d_min`/`d_max` from the full test set, not the evaluation subset |
| 70 | HIGH | `robustness/auto_attack.py:230–258` | Add explicit shape assertions before and after the 4D→3D permutation in `AAWrapper` |
| 71 | HIGH | `evaluate_autoattack.py:121–128` | Fix `step_accs[0]` so clean accuracy is computed on the actual subset, not forced to 1.0 |
| 77 | HIGH | `evaluate_autoattack.py:90–94,122–128` | Replace private `._autoattack` attribute access with the public API; if unavailable, remove gradient masking detection and document limitation |
| 82 | HIGH | `compare_fgsm_baselines.py:256–261` | Use `nn.CrossEntropyLoss` for LSTM/BiLSTM FGSM attack; use `RLSTMLoss` only for HMR-BiLSTM |

---

### 2C — Calibration evaluation (10 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 44 | HIGH | `evaluate_calibration.py:28,223` | Standardize to 15-bin ECE throughout; remove hardcoded 10 |
| 45 | HIGH | `evaluate_calibration.py:220–223` | Apply fitted temperature to logits before computing ECE: `scaled_logits = temp_model(logits)` |
| 46 | HIGH | `calibration/calibration_metrics.py:117–126` | Add option for equal-mass binning; default to it for imbalanced datasets |
| 47 | HIGH | `calibration/temperature_scaling.py:27` | Add `.to(device)` guard inside `TemperatureScaling.fit()` |
| 49 | HIGH | `calibration/reliability_diagram.py` | Map attention weights from T/4 CNN-downsampled axis back to original T=187 timestep axis |
| 104 | MEDIUM | `calibration/calibration_metrics.py:159` | Weight conditional ECE by `n_c / N` not `1/n_classes` to make it comparable to global ECE |
| 105 | MEDIUM | `calibration/calibration_metrics.py:74` | Clip probabilities per-row after softmax to preserve the simplex constraint |
| 106 | MEDIUM | `calibration/temperature_scaling.py:43,52` | Add LBFGS convergence check; warn if final gradient norm > 1e-3 |
| 107 | MEDIUM | `calibration/calibration_metrics.py:150–159` | Skip or warn on degenerate confidence bins to prevent silent ECE=0.0 |
| 108 | MEDIUM | `evaluate_calibration.py:239–241` | Add `Path("results/tables").mkdir(parents=True, exist_ok=True)` before writing CSV |
| 129 | MEDIUM | `evaluate_calibration.py:223` | Remove hardcoded `n_bins=10`; read from `cfg["calibration"]["num_bins"]` |
| 130 | MEDIUM | `evaluate_calibration.py` | Write post-calibration results to `outputs/<run_id>/calibration/results.json` so T8 picks them up |

---

### 2D — Uncertainty evaluation (10 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 50 | HIGH | `uncertainty/mc_dropout.py:51–66` | Extend `enable_mc_dropout` to set `train(True)` on `Dropout2d`, `Dropout3d`, and `AlphaDropout` layers |
| 51 | HIGH | `uncertainty/mc_dropout.py:80–86` | Replace hardcoded `/ 187` with `/ (len(beat) / fs)` to get correct Hz |
| 52 | HIGH | `uncertainty/mc_dropout.py:189–193` | Apply symmetric epsilon clipping `eps = 1e-10` to both numerator and denominator; add diagnostic log if any MI value was clipped |
| 55 | HIGH | `evaluate_trustworthiness.py:167–171` | Change `labels=[0,1,2,3]` F1 to the canonical convention decided in Phase 3 |
| 56 | HIGH | `evaluate_trustworthiness.py:122` | Fix condition to `abs(abs(eps) - 0.02) < 0.001` |
| 109 | MEDIUM | `uncertainty/mc_dropout.py:151–162,315` | Remove dead `model.eval()` at line 315; add a guard that restores original module training states after MC sampling |
| 110 | MEDIUM | `uncertainty/mc_dropout.py:231–267` | Fix bar width to `1.0 / n_bins`; use MC-derived `entropy`/`MI` statistics in the plot |
| 111 | MEDIUM | `uncertainty/deep_ensemble.py:130` | Replace `probs.max(axis=2).std()` with per-class disagreement: `probs.std(axis=0).mean()` |
| 112 | MEDIUM | `uncertainty/mc_dropout.py:402`, `deep_ensemble.py:275` | Rename JSON key `ood_detection_auroc` to `corruption_detection_auroc` consistently |
| 114 | MEDIUM | `uncertainty/mc_dropout.py:115–116` | Add explicit `axis=` argument to `np.roll`; add a comment explaining the roll semantics |

---

### 2E — Explainability evaluation (11 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 35 | HIGH | `hmr_bilstm.py:280–281` | Remove or comment out `h_T`/`c_T` computation since it is never used downstream |
| 59 | HIGH | `explainability/integrated_gradients.py:284–285` | Change `mean_abs_attribution_plot` to use `torch.abs(ig_tensor).mean(dim=0)` instead of signed mean |
| 61 | HIGH | `explainability/data_attribution.py:74,244,302–308` | Add `model.eval()` call at the start of the gradient loop; add assertion `not model.training` |
| 95 | MEDIUM | `run_ablation.py:183–188` | Use `average="macro", labels=[0,1,2,3]` consistently to prevent silent 0.0 AUC when Q-class absent |
| 115 | MEDIUM | `explainability/shap_analysis.py:174–175,206` | Emit a `warnings.warn` when `n_background` is overridden below the user-configured value |
| 116 | MEDIUM | `explainability/shap_analysis.py:44–50` | Change `ModelWrapper.forward` to return `torch.softmax(logits, dim=-1)` for probability-calibrated SHAP |
| 118 | MEDIUM | `explainability/integrated_gradients.py:219–228` | After collection, filter out samples with `convergence_delta > threshold` from Jaccard computation |
| 120 | MEDIUM | `explainability/plot_disagreements.py:11` | Replace hardcoded run-ID with `get_run_id(cfg)` |
| 121 | MEDIUM | `explainability/shap_analysis.py:344–346` | Add mini-batching: split the 380-sample explanation set into chunks of 64 |
| 123 | MEDIUM | `explainability/shap_analysis.py:213–214,339–342` | Replace `np.random.seed()` calls with `rng = np.random.default_rng(seed)` throughout |
| 39 | HIGH | `train_inter_patient.py:115–124` | Pass `probs_4class` (renormalised to sum=1) to `roc_auc_score`, not raw `probs[:, :4]` |

---

### 2F — Code quality, diagnostics, and tests (23 issues)

| # | Sev | File | Fix Required |
|---|---|---|---|
| 37 | HIGH | `train.py:179–190` | Add `nan_batches_skipped` counter; log warning if count exceeds 5% of total batches |
| 74 | HIGH | `run_baselines.py:174–175` | Remove dead `y_pred = model.predict(X_tr)` line |
| 78 | HIGH | `test_shape.py:1–14` | Replace Mock model with `RLSTMClassifier`; add `if torch.cuda.is_available():` guard |
| 79 | HIGH | Diagnostic scripts | Replace hardcoded run-IDs in all 5 diagnostic scripts with `get_run_id(cfg)` |
| 80 | HIGH | `verify_gradients.py:9–16` | Fix model flags and loss config to match `train_inter_patient.py` exactly |
| 81 | HIGH | `run_ablation_inter.py` | Fix checkpoint path to always point to the freshest inter-patient full-model checkpoint |
| 93 | MEDIUM | `run_baselines.py:358–417` | Standardize all baseline saves to the same wrapper dict format as HMR-BiLSTM |
| 96 | MEDIUM | `run_ablation.py:357–360` | Add `assert Path(ckpt_path).exists()` before `torch.load` |
| 97 | MEDIUM | `pgd_convergence.py:140,195` | Rename inner loop variable to `step_count` to avoid shadowing the outer `steps` list |
| 98 | MEDIUM | `pgd_convergence.py:33–43` | Replace floor-division per-class count with ceiling to ensure total >= `subset_size` |
| 102 | MEDIUM | `validation/verify_normalization.py:106–110` | Replace algebraic check with a sample-based leakage check; remove dead `leakage_detected` variable |
| 103 | MEDIUM | `validation/verify_normalization.py:103–104` | Fix threshold to compare against per-feature statistics |
| 119 | MEDIUM | `explainability/data_attribution.py:103–130` | Lower confidence threshold or remove it; document that high-confidence incorrect predictions on noisy data are the most informative TracIn examples |
| 132 | MEDIUM | `plot_case_visualization.py:35–36` | Point `model_path` to `RLSTM_CKPT` (inter-patient) from `configs/paths.py` |
| 134 | MEDIUM | `test_speed.py:9–10` | Replace global CONFIG mutation with a local copy: `cfg = copy.deepcopy(CONFIG)` |
| 135 | MEDIUM | `diag_verify_coupling_v2.py:249–257` | Use `elif` for the second amplitude filter branch to prevent double-append |
| 87 | MEDIUM | `hmr_bilstm_ablation.py:224` | Remove commented-out `use_interaction` attribute |
| 140 | LOW | `hmr_bilstm.py:347` | Remove unreachable second `return` |
| 141 | LOW | `hmr_bilstm.py:104–111` | Move `last_r_t`, `last_c_keep`, `last_c_add` out of instance state into the `forward` return tuple |
| 142 | LOW | `hmr_bilstm_ablation.py:107–109` | Conditionally allocate `W_beta` only when `use_hybrid=True` |
| 145 | LOW | `preprocess.py:94` | Normalise indentation to spaces |
| 147 | LOW | `validation/verify_normalization.py:126` | Extend leakage check to also verify the test set |
| 148 | LOW | `validation/preprocess_aami.py:129–132` | Add `if len(y) == 0: return` guard in `print_dist` |
| 149 | LOW | `calibration/calibration_metrics.py:163` | Remove duplicate `return out` |
| 150 | LOW | `calibration/reliability_diagram.py:36–57` | Filter out bins with `sample_count == 0` before writing CSV |
| 151 | LOW | Multiple calibration files | Standardise to `dim=-1` in all `torch.softmax` calls |
| 152 | LOW | `uncertainty/deep_ensemble.py:27` | Remove `import copy` |
| 153 | LOW | `evaluate_fgsm.py:136–138` | Remove duplicate `"label_accuracy"` key from result dict |
| 154 | LOW | `gen_guide.py:408` | Change `class_weights.json` reference to `class_weights.npy` |

---

### Phase 2 Exit Condition

- [ ] `python -m pytest` passes all tests (or failures are in known-broken tests documented as TODO)
- [ ] `python evaluate_calibration.py` runs end-to-end without `FileNotFoundError`
- [ ] `python evaluate_trustworthiness.py` runs end-to-end without `TypeError` on Python 3.9
- [ ] All 5 diagnostic scripts accept a `--run_id` argument instead of hardcoded values
- [ ] `pgd_convergence.py` passes `r_fwd`/`r_bwd` to the loss

---

## Phase 3 — Experimental Redesign

**Goal:** Make all binding design decisions and set up any new experimental infrastructure
before training begins. These are decisions, not code edits — they determine what Phase 4
will produce.

**Duration estimate:** 2–3 days  
**Issue count:** 13  
**Parallelisable with:** Phase 1, Phase 2

---

### 3A — Canonical evaluation regime (resolves #5, #6, #8, #21, #72, #83)

**Decision record — must be written down and agreed before Phase 4:**

| Decision | Choice | Rationale |
|---|---|---|
| Primary model | `train_inter_patient.py` + `inter_best_rlstm.pt` | Clinically valid; avoids patient leakage |
| Retire | `train.py` / `best_rlstm.pt` results | Intra-patient results cannot be compared with inter-patient baselines |
| F1 averaging | **4-class macro (labels=N,S,V,F)** per AAMI EC57 | Standard for clinical ECG classification; Q class excluded by convention |
| Canonical test set | `data/processed/splits/inter_test.npz` | All models evaluated on this set only |
| Baseline split | Same inter-patient split | Eliminates #6 split mismatch |
| Ablation split | Same inter-patient split | Eliminates #72 scale mismatch |

---

### 3B — Adversarial evaluation protocol (resolves #12, #28, #65)

| Decision | Choice |
|---|---|
| AutoAttack sample size | Full `inter_test.npz` or stratified 1,000 samples/class (whichever is feasible) |
| CW sample size | Same stratified 1,000 samples/class set |
| Clean baseline for F1 drop | Computed on the same sample subset (not a pre-filtered correct-only subset) |
| No-Adv anomaly | Treat as a split mismatch artifact; expect it to disappear after Block A unification. If it persists after Phase 4, add as a paper limitation. |

---

### 3C — Baseline training protocol (resolves #41)

| Parameter | HMR-BiLSTM | Required Baseline Setting |
|---|---|---|
| Epochs | 45 | **45** (equal) |
| LR schedule | Cosine annealing | **Cosine annealing** (equal) |
| FGSM augmentation | 30%, ε=0.02 | **30%, ε=0.02** (equal) |
| Class weights | From `inter_class_weights.npy` | **Same** (already fixed by #73) |
| Early stopping metric | 4-class macro F1 | **Same** |

---

### 3D — Ensemble training plan (resolves #17, #64, #90)

- Train **3 independent ensemble members** from scratch: seeds 42, 123, 456
- All three use identical architecture, data, and training config
- Verify after training: `ensemble_MI.mean() > 0.001`; if still 0, diagnose before reporting
- Retire the current `best_rlstm.pt` seed-42 checkpoint as the ensemble "single mode"

---

### 3E — OOD dataset design (resolves #53)

- Replace synthetic OOD (augmented ID test beats) with one or more of:
  - Gaussian noise–only samples (no real ECG content)
  - Beats from a different PhysioNet dataset not used in training (e.g., PTB-XL short segments)
  - Randomly phase-shuffled beats (destroys clinical morphology)
- Document the OOD data source in the paper; do not claim "OOD detection" with ID-derived data

---

### 3F — IG methodology decisions (resolves #57, #58)

| Parameter | Current | Required |
|---|---|---|
| `n_steps` | 50 | **≥ 200** (convergence delta < 0.01) |
| Baseline | Zero vector | **Training-set mean** (apply BN stats first) |
| Filtering | None | **Remove samples with convergence delta > 0.05** |

---

### 3G — SHAP background stratification (resolves #62)

- Stratify the 200-sample background set by class proportions matching the training
  distribution (use `sklearn.utils.resample` with stratify)
- This change alone may shift reported per-class SHAP attributions; regenerate all figures

---

### 3H — Method renaming (resolves #60)

- Throughout codebase and paper, rename every occurrence of "TracIn" to
  **"gradient cosine similarity (single-checkpoint)"**
- Add a footnote: "A multi-checkpoint TracIn approximation would require checkpoints at
  every epoch weighted by their learning rate; this is deferred to future work."

---

### Phase 3 Exit Condition

- [ ] Decision record above is agreed and committed (e.g., as a DECISIONS.md file)
- [ ] `inter_class_weights.npy` exists and was generated from the inter-patient training split
- [ ] 3 ensemble seed slots are reserved (GPU schedule exists for Phase 4)
- [ ] OOD dataset is prepared and saved (separate from the test set)
- [ ] IG `n_steps=200` and training-set-mean baseline code is written (Phase 2 dependency)

---

## Phase 4 — Retraining

**Goal:** Produce all new trained checkpoints using the corrected pipeline from Phases 1 and 3.

**Duration estimate:** 10–14 days (dominated by GPU training time; most steps can be parallelized with multiple GPUs)  
**Issue count (resolved by running this phase):** 21  
**Dependency:** Phase 1 complete AND Phase 3 decisions recorded

---

### 4A — Re-run data preprocessing (~2–4 hours CPU)

```bash
# Step 1: Re-generate intra-patient splits (now normalised, stratified, correct boundary)
python validation/preprocess_aami.py

# Step 2: Re-generate intra-patient class weights (now without hard cap)
python preprocess.py
```

**Resolves:** #23, #27, #42, #43, #99, #100, #101, #146

---

### 4B — Retrain primary HMR-BiLSTM (2–4 days GPU)

```bash
python train_inter_patient.py
```

Checkpoint to save: `results/checkpoints/inter_best_rlstm_v2.pt`  
Check during training:
- Early stopping uses 4-class macro F1
- `model.zero_grad()` after `loss.backward()`
- `self.dropout` is applied inside `RLSTMCell`
- NaN batch counter stays below 5%

**Resolves (by retraining with corrected code):** #29, #33, #34, #36, #38, #89, #94

---

### 4C — Retrain all baselines under fair conditions (3–5 days GPU)

```bash
python run_baselines.py --epochs 45 --lr_schedule cosine --fgsm_aug --inter_patient
```

Baselines: Logistic Regression, Decision Tree, LSTM, BiLSTM  
All use: `inter_train.npz`, 45 epochs, cosine LR, FGSM augmentation, `inter_class_weights.npy`

**Resolves:** #41, #73

---

### 4D — Retrain Deep Ensemble (3 independent seeds) (3–5 days GPU)

```bash
for seed in 42 123 456; do
    python train_inter_patient.py --seed $seed --out ensemble/$seed
done
```

Verify after training:
- `ensemble_MI > 0` on 100 test samples
- All 3 checkpoints have non-NaN `val_auc_ovr`

**Resolves:** #17, #64, #90

---

### 4E — Rerun ablation study (2–4 days GPU)

```bash
python run_ablation_inter.py
```

With Phase 1 fixes:
- `no_rmc` variant uses consistent smoothness loss (#86)
- DataLoader recreated with fixed seed per variant (#91)
- All variants use `inter_train.npz`

**Resolves:** #86, #91

---

### 4F — Investigate No-Adv > Full anomaly

After Block A unification, re-inspect `inter_patient_results.json`:
- If the anomaly disappears: document as resolved by split unification
- If it persists: add as a paper limitation — "adversarial training did not improve
  inter-patient F1 in this experiment; we hypothesise this is due to the diversity of
  ECG morphology across patients suppressing the adversarially learned features"

**Resolves:** #65, #136 (partially)

---

### Phase 4 Exit Condition

- [ ] `inter_best_rlstm_v2.pt` checkpoint exists with non-NaN val metrics
- [ ] All 4 baseline checkpoints saved and loadable
- [ ] 3 ensemble checkpoints saved; `ensemble_MI.mean() > 0.001`
- [ ] Ablation results JSON written with no NaN entries
- [ ] No-Adv anomaly documented (resolved or explained)
- [ ] All training logs show `nan_batches_skipped < 5%`

---

## Phase 5 — Regenerate All Tables and Figures

**Goal:** With fixed code (Phases 1+2) and new checkpoints (Phase 4), produce a complete,
internally consistent set of results.

**Duration estimate:** 5–7 days  
**Issue count (resolved by regenerating outputs):** 20  
**Dependency:** Phase 2 complete AND Phase 4 complete

---

### 5A — Delete all stale pre-computed outputs

```bash
# WARNING: verify Phase 4 checkpoints exist before running
rm -rf results/tables/*.csv
rm -rf outputs/v1.0_FINAL/
rm -rf results/logs/*.json
```

**Resolves provenance issues:** #126, #7, #8 (stale impossible values removed)

---

### 5B — Run robustness evaluation (1–2 days)

```bash
python evaluate_fgsm.py          # FGSM, all epsilon levels
python evaluate_pgd.py           # PGD (fixed bounds from #67)
python evaluate_autoattack.py    # AutoAttack on full inter_test.npz
python robustness/cw_attack.py   # C&W on full inter_test.npz (c=1e-2)
python pgd_convergence.py        # PGD step sweep (verify monotonicity)
python evaluate_robustness_all.py # Gaussian noise sweep (inter_test.npz)
```

**Resolves:** #11, #12, #28, #67, #69, #71, #77, #139

---

### 5C — Run calibration evaluation (0.5 days)

```bash
python calibration/temperature_scaling.py   # fit temperature
python evaluate_calibration.py              # 15-bin ECE, pre+post labelled
python calibration/reliability_diagram.py   # regenerate diagrams
```

**Resolves:** #44, #45, #46, #48, #75, #76, #104, #105, #129, #130

---

### 5D — Run uncertainty evaluation (0.5 days)

```bash
python uncertainty/mc_dropout.py       # MC Dropout (fixed enable function)
python uncertainty/deep_ensemble.py    # Ensemble (3 seeds, MI > 0)
python evaluate_corruptions.py         # Corruption sweep
```

**Resolves:** #50, #51, #52, #53, #54, #63, #64, #109, #110, #111, #112

---

### 5E — Run explainability evaluation (1–2 days)

```bash
python explainability/integrated_gradients.py   # 200 steps, mean baseline, abs mean
python explainability/shap_analysis.py          # stratified background, 5 classes
python explainability/data_attribution.py       # gradient cosine similarity
```

**Resolves:** #57, #58, #59, #60, #62, #116, #118

---

### 5F — Regenerate all tables and figures (0.5 days)

```bash
python generate_results_tables.py    # unified F1 averaging, consistent units
python evaluate_trustworthiness.py   # T8 scorecard
python evaluate_splits.py            # intra vs inter comparison (same normalization)
```

**Resolves:** #21, #55, #83, #92, #113, #124, #125, #126, #138

---

### Phase 5 Exit Condition

- [ ] `results/tables/table5_consolidated.csv`: every row has adversarial F1 ≤ clean F1
- [ ] `pgd_convergence_results.json`: accuracy decreases monotonically with steps
- [ ] ECE values consistent (within ±0.005) across all files that report the same model
- [ ] Ensemble MI column is non-zero in all uncertainty result files
- [ ] All figures regenerated from the new outputs (no figures referencing the old run IDs)
- [ ] `python run_reproducible_pipeline.py` completes end-to-end on a clean run

---

## Phase 6 — Final Validation

**Goal:** Cross-check all outputs for scientific coherence, test reproducibility on a clean
machine, and revise the paper to match the corrected results and methodology.

**Duration estimate:** 5–7 days  
**Issue count (resolved by paper corrections and validation):** 12

---

### 6A — Scientific coherence checks (1 day)

Run the following assertions on every table in `results/tables/`:

| Check | Assertion |
|---|---|
| Adversarial robustness | For every model: FGSM-F1 ≤ Clean-F1, PGD-F1 ≤ FGSM-F1, AutoAttack-F1 ≤ PGD-F1 |
| Calibration ordering | ECE post-calibration < ECE pre-calibration for the majority of classes |
| Ensemble vs single model | Ensemble MI ≥ MC Dropout MI (ensembles are more diverse by construction) |
| No-Adv anomaly | No-Adv-F1 ≤ Full-F1 on inter-patient split |
| PGD convergence | step-10 accuracy ≥ step-50 accuracy ≥ step-100 accuracy |
| Class weight sanity | All reported class weights > 0; no class excluded |

**Resolves:** Validates that #7, #8, #63, #65, #75, #139 are all resolved

---

### 6B — Reproducibility test (0.5 days)

On a clean virtual environment:

```bash
git clone <repo>
pip install -r requirements.txt
python run_reproducible_pipeline.py
```

Expected outcome: All 8 trustworthiness steps complete without ImportError or FileNotFoundError.

**Resolves:** Validates that #14, #25, #127, #128 are resolved

---

### 6C — Paper revision: model description (1–2 days)

Sections requiring correction:

| # | Section | Correction |
|---|---|---|
| 32 | Architecture section | Correct mathematical description of RMC: c_keep + c_add is a competing memory path, not a decomposition of c_lstm. Provide the correct equation. |
| 40 | Loss function section | Label focal loss as "modified focal loss (α applied as post-multiplier)" rather than citing Lin et al. (2017) directly |
| 29 | Implementation details | State explicitly: "within-cell dropout is zero; dropout is applied only between layers in BiRLSTM" (if #29 is not fixed) OR report the within-cell dropout rate if #29 is fixed |
| 84 | Architecture table | Note that α (scalar) and β (vector) are asymmetric gating mechanisms |
| 144 | Architecture table | Correct β description to "per-dimension vector of shape (H,), not a scalar" |

**Resolves:** #32, #40, #84, #144

---

### 6D — Paper revision: evaluation methodology (1–2 days)

| # | Section | Correction |
|---|---|---|
| 21 | Evaluation section | State explicitly: "All F1 scores are 4-class macro averages (N, S, V, F) per AAMI EC57; class Q is excluded" |
| 41 | Baseline comparison | State: "All baselines were retrained under identical conditions: 45 epochs, cosine LR, FGSM augmentation at ε=0.02" |
| 53 | Uncertainty section | Replace "OOD detection" with correct description of the OOD dataset source |
| 60 | Explainability section | Replace "TracIn" with "gradient cosine similarity (single-checkpoint approximation)" with the referenced footnote |
| 66 | Hyperparameter section | Document actual tuning methodology; if full grid search was not done, state that hyperparameters were selected by manual sweep |
| 113 | OOD section | Report noise levels as SNR (dB) rather than raw standard deviation |
| 137 | Hyperparameter table | Reconcile table with actual final model configuration |

**Resolves:** #21, #41, #53, #60, #66, #113, #137

---

### 6E — Paper revision: results tables (0.5 days)

| # | Table | Correction |
|---|---|---|
| 48 | Calibration table | Add column header "ECE (pre-cal)" and "ECE (post-cal)" |
| 54 | Uncertainty table | Add footnote distinguishing MC Dropout MI (aleatoric+epistemic) from Ensemble MI (primarily epistemic) |
| 125 | FGSM robustness table | Add Fusion (F) class recall column |
| 124 | FGSM/baseline comparison | Standardize F1 drop column to absolute units (0.0400, not 4.54%) |
| 92 | Ablation table | Use identical F1 averaging (4-class) in all ablation rows |

**Resolves:** #48, #54, #92, #124, #125

---

### Phase 6 Exit Condition

- [ ] All coherence assertions from 6A pass programmatically
- [ ] Clean-install reproducibility test passes
- [ ] Paper sections on model, loss, baselines, explainability, OOD corrected
- [ ] All tables use consistent F1 averaging with clear documentation
- [ ] No figure references a stale run-ID
- [ ] `SCIENTIFIC_AUDIT_REPORT.md` acknowledged in the paper's limitations section

---

## Full Issue-to-Phase Index

| Issue | Phase | Category | Sev |
|---|---|---|---|
| 5 | P3 | EP | CRITICAL |
| 6 | P3 | EP | CRITICAL |
| 7 | P5 | RI | CRITICAL |
| 8 | P3+P5 | RI | CRITICAL |
| 11 | P2 | CB | CRITICAL |
| 12 | P3+P5 | EP | CRITICAL |
| 14 | P2 | CB | CRITICAL |
| 17 | P3+P4 | RT | CRITICAL |
| 21 | P3+P6 | RI | CRITICAL |
| 23 | P1 | RT | CRITICAL |
| 25 | P2 | CB | CRITICAL |
| 26 | P2 | CB | CRITICAL |
| 27 | P1 | RT | CRITICAL |
| 28 | P3+P5 | RI | CRITICAL |
| 29 | P1+P4 | RT | HIGH |
| 30 | P1 | CB | HIGH |
| 31 | P1 | CB | HIGH |
| 32 | P6 | RI | HIGH |
| 33 | P1+P4 | RT | HIGH |
| 34 | P1+P4 | RT | HIGH |
| 35 | P2 | CB | HIGH |
| 36 | P1+P4 | RT | HIGH |
| 37 | P2 | CB | HIGH |
| 38 | P1+P4 | RT | HIGH |
| 39 | P2 | CB | HIGH |
| 40 | P1+P6 | EP | HIGH |
| 41 | P3+P4 | EP | HIGH |
| 42 | P1+P4 | RT | HIGH |
| 43 | P1+P4 | RT | HIGH |
| 44 | P2 | CB | HIGH |
| 45 | P2 | CB | HIGH |
| 46 | P2 | CB | HIGH |
| 47 | P2 | CB | HIGH |
| 48 | P2+P6 | RI | HIGH |
| 49 | P2 | CB | HIGH |
| 50 | P2 | CB | HIGH |
| 51 | P2 | CB | HIGH |
| 52 | P2 | CB | HIGH |
| 53 | P3+P5 | EP | HIGH |
| 54 | P6 | RI | HIGH |
| 55 | P2+P6 | RI | HIGH |
| 56 | P2 | CB | HIGH |
| 57 | P3+P5 | EP | HIGH |
| 58 | P3+P5 | EP | HIGH |
| 59 | P2 | CB | HIGH |
| 60 | P3+P6 | RI | HIGH |
| 61 | P2 | CB | HIGH |
| 62 | P3+P5 | EP | HIGH |
| 63 | P4+P5 | RI | HIGH |
| 64 | P4 | RT | HIGH |
| 65 | P3+P4 | EP | HIGH |
| 66 | P6 | RI | HIGH |
| 67 | P2 | CB | HIGH |
| 68 | P2 | CB | HIGH |
| 69 | P2 | CB | HIGH |
| 70 | P2 | CB | HIGH |
| 71 | P2 | CB | HIGH |
| 72 | P3 | EP | HIGH |
| 73 | P1+P4 | RT | HIGH |
| 74 | P2 | CB | HIGH |
| 75 | P5+P6 | RI | HIGH |
| 76 | P5 | RI | HIGH |
| 77 | P2 | CB | HIGH |
| 78 | P2 | CB | HIGH |
| 79 | P2 | CB | HIGH |
| 80 | P2 | CB | HIGH |
| 81 | P2 | CB | HIGH |
| 82 | P2 | CB | HIGH |
| 83 | P3+P6 | RI | HIGH |
| 84 | P6 | RI | MEDIUM |
| 85 | P1 | CB | MEDIUM |
| 86 | P1+P4 | EP | MEDIUM |
| 87 | P2 | CB | MEDIUM |
| 88 | P1 | CB | MEDIUM |
| 89 | P1+P4 | RT | MEDIUM |
| 90 | P3+P4 | RT | MEDIUM |
| 91 | P1+P4 | EP | MEDIUM |
| 92 | P2+P6 | RI | MEDIUM |
| 93 | P2 | CB | MEDIUM |
| 94 | P1+P4 | RT | MEDIUM |
| 95 | P2 | CB | MEDIUM |
| 96 | P2 | CB | MEDIUM |
| 97 | P2 | CB | MEDIUM |
| 98 | P2 | CB | MEDIUM |
| 99 | P1 | CB | MEDIUM |
| 100 | P1+P4 | EP | MEDIUM |
| 101 | P1+P4 | RT | MEDIUM |
| 102 | P2 | CB | MEDIUM |
| 103 | P2 | CB | MEDIUM |
| 104 | P2 | CB | MEDIUM |
| 105 | P2 | CB | MEDIUM |
| 106 | P2 | CB | MEDIUM |
| 107 | P2 | CB | MEDIUM |
| 108 | P2 | CB | MEDIUM |
| 109 | P2 | CB | MEDIUM |
| 110 | P2 | CB | MEDIUM |
| 111 | P2 | CB | MEDIUM |
| 112 | P2 | CB | MEDIUM |
| 113 | P6 | RI | MEDIUM |
| 114 | P2 | CB | MEDIUM |
| 115 | P2 | CB | MEDIUM |
| 116 | P2 | CB | MEDIUM |
| 117 | P3+P5 | EP | MEDIUM |
| 118 | P2 | CB | MEDIUM |
| 119 | P2 | EP | MEDIUM |
| 120 | P2 | CB | MEDIUM |
| 121 | P2 | CB | MEDIUM |
| 123 | P2 | CB | MEDIUM |
| 124 | P5+P6 | RI | MEDIUM |
| 125 | P5+P6 | RI | MEDIUM |
| 126 | P5 | RI | MEDIUM |
| 127 | P2 | CB | MEDIUM |
| 128 | P2 | CB | MEDIUM |
| 129 | P2 | CB | MEDIUM |
| 130 | P2 | CB | MEDIUM |
| 131 | P2 | CB | MEDIUM |
| 132 | P2 | CB | MEDIUM |
| 134 | P2 | CB | MEDIUM |
| 135 | P2 | CB | MEDIUM |
| 136 | P4+P6 | RI | MEDIUM |
| 137 | P6 | RI | MEDIUM |
| 138 | P5 | RI | MEDIUM |
| 139 | P5 | RI | MEDIUM |
| 140 | P2 | CB | LOW |
| 141 | P2 | CB | LOW |
| 142 | P2 | CB | LOW |
| 143 | P1 | RT | LOW |
| 144 | P6 | RI | LOW |
| 145 | P2 | CB | LOW |
| 146 | P1 | CB | LOW |
| 147 | P2 | CB | LOW |
| 148 | P2 | CB | LOW |
| 149 | P2 | CB | LOW |
| 150 | P2 | CB | LOW |
| 151 | P2 | CB | LOW |
| 152 | P2 | CB | LOW |
| 153 | P2 | CB | LOW |
| 154 | P2 | CB | LOW |

---

## Timeline Summary

| Phase | Duration | Issues | Parallelisable |
|---|---|---|---|
| Phase 1 — Must fix before retraining | 5–7 days | 25 | Yes (w/ P2, P3) |
| Phase 2 — Safe automatic code fixes | 3–4 days | 67 | Yes (w/ P1, P3) |
| Phase 3 — Experimental redesign | 2–3 days | 13 | Yes (w/ P1, P2) |
| Phase 4 — Retraining | 10–14 days | 21 | Mostly sequential |
| Phase 5 — Regenerate tables and figures | 5–7 days | 20 | Partially |
| Phase 6 — Final validation | 5–7 days | 12 | Sequential |
| **Total (sequential)** | **30–42 days** | **138** | |
| **Total (parallelised P1+P2+P3)** | **~8–10 weeks** | **138** | |

### Critical path

```
Week 1–2:  [P1 + P2 + P3 in parallel]
Week 3:    [P3 decisions finalized] → gate for P4
Week 3–6:  [P4: preprocessing + primary model + baselines + ensemble + ablation]
Week 7:    [P5: all evaluation re-runs + table generation]
Week 8–10: [P6: validation + paper revision]
```

The single longest activity is baseline retraining (P4C) + primary model retraining (P4B) run serially. With 2 GPUs these can be parallelised, reducing P4 to ~1 week.

---

*Generated: 2026-06-23*  
*Issues tracked: 138 remaining across Phases 1–6*  
*Issues already resolved: 16 (8 fixed + 8 false positives)*
