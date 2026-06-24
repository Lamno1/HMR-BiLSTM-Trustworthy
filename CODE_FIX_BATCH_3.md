# Code Fix Batch 3 — HMR-BiLSTM

**Session:** 3 (2026-06-24)  
**Scope:** All remaining code-level bugs that do not require retraining or experimental protocol changes.  
**Constraint:** No experimental protocol changes, no model retraining.  
**Files changed:** 16  
**Issues resolved:** 24 fixes + 2 new false positives identified

---

## Summary

| # | Issue | File | Change type |
|---|-------|------|-------------|
| B3-01 | #31 | `hmr_bilstm.py:350` | Disconnected leaf tensor — gradient never flows |
| B3-02 | #47 | `calibration/temperature_scaling.py:28` | Missing `self.to(device)` in `fit()` |
| B3-03 | #106 | `calibration/temperature_scaling.py:46-52` | No LBFGS convergence diagnostic |
| B3-04 | #50 | `uncertainty/mc_dropout.py:65` | `enable_mc_dropout` misses Dropout2d/3d/AlphaDropout |
| B3-05 | #51 | `uncertainty/mc_dropout.py:84` | Baseline wander freq unit ambiguous |
| B3-06 | #52 | `uncertainty/mc_dropout.py:192-195` | MI clipped without diagnostic |
| B3-07 | #114 | `uncertainty/mc_dropout.py:116` | `np.roll` axis undocumented |
| B3-08 | #110 | `uncertainty/mc_dropout.py:253` | Bar width hard-coded as 0.08 instead of `1/n_bins` |
| B3-09 | #152 | `uncertainty/deep_ensemble.py:27` | Unused `import copy` |
| B3-10 | #111 | `uncertainty/deep_ensemble.py:130` | `std_max` ignores class identity |
| B3-11 | #82 | `compare_fgsm_baselines.py:255-280` | Wrong loss for LSTM/BiLSTM FGSM |
| B3-12 | #91 | `run_ablation.py:543` | Seed not reset per variant |
| B3-13 | #95 | `run_ablation.py:184-186` | 5-class AUC taints ablation table |
| B3-14 | #127 | `run_reproducible_pipeline.py:57-66` | AutoAttack missing from pipeline |
| B3-15 | #128 | `evaluate_trustworthiness.py:287` | T8 reads wrong FGSM JSON filename |
| B3-16 | #134 | `test_speed.py:9-13` | CONFIG dict permanently mutated |
| B3-17 | #120 | `explainability/plot_disagreements.py:10-13` | Hardcoded run ID |
| B3-18 | #57 | `explainability/integrated_gradients.py:200` | n_steps=50 insufficient |
| B3-19 | #115 | `explainability/shap_analysis.py:206` | Silent cap on n_background |
| B3-20 | #123 | `explainability/shap_analysis.py:338-341` | Mixed old/new NumPy RNG |
| B3-21 | #125 | `generate_results_tables.py:215-232` | Rec-F columns missing from FGSM table |
| B3-22 | #87 | `hmr_bilstm_ablation.py:224` | Dead commented-out attribute |
| B3-23 | #70 | `robustness/auto_attack.py:247-255` | Fragile AAWrapper shape routing |
| B3-24 | #99 | `validation/preprocess_aami.py:78-80` | `all_beats` saved as pickle-dependent npz |

**New false positives identified (not fixed):**  
- #130: `evaluate_calibration.py` was claimed to not write `results.json` — file already does so at line 179.  
- #145: `preprocess.py:94` was claimed to have mixed tab/space — no tabs found in file.

---

## Detailed Changes

### B3-01 · `hmr_bilstm.py:350` · #31 — Disconnected leaf tensor

**Problem:** `torch.tensor(0.0, requires_grad=True)` creates a leaf tensor not connected to any model parameter. Gradients do not flow through it; backprop on the smoothness loss path is silently broken when `r_seq.size(1) < 2`.

**Fix:**
```python
# Before
return torch.tensor(0.0, device=r_seq.device, requires_grad=True)

# After
return r_seq.sum() * 0.0
```
`r_seq.sum() * 0.0` evaluates to zero but preserves the computation graph, allowing gradients to propagate correctly.

---

### B3-02 · `calibration/temperature_scaling.py:28` · #47 — Missing device guard

**Problem:** `TemperatureScaling.fit()` is called with `device` but never moves the module parameters to that device. The LBFGS optimizer will fail with a device mismatch when using CUDA.

**Fix:** Added `self.to(device)` as the first line of `fit()`.

---

### B3-03 · `calibration/temperature_scaling.py:46-52` · #106 — No convergence diagnostic

**Problem:** LBFGS may converge poorly but the caller has no way to detect this. Silent non-convergence leads to a sub-optimal temperature that degrades calibration.

**Fix:** Compute NLL before and after optimization; print both values and emit a warning if NLL did not decrease.

---

### B3-04 · `uncertainty/mc_dropout.py:65` · #50 — Incomplete Dropout variant coverage

**Problem:** `enable_mc_dropout` only activates `nn.Dropout` modules. If the model uses `Dropout2d`, `Dropout3d`, or `AlphaDropout`, those layers remain in `eval` mode during MC sampling and provide zero stochasticity.

**Fix:**
```python
# Before
if isinstance(m, nn.Dropout):

# After
if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
```

---

### B3-05 · `uncertainty/mc_dropout.py:84` · #51 — Ambiguous freq unit

**Problem:** The `freq` parameter to `ood_baseline_wander` is labelled without units. The formula `freq * T / 187` produces cycles-per-beat-window, which is not Hz and not intuitive.

**Fix:** Added inline comment: `# freq is in cycles per 187-sample beat window (not Hz)`.

---

### B3-06 · `uncertainty/mc_dropout.py:192-195` · #52 — Silent MI clipping

**Problem:** Negative mutual information values are clipped to zero without any diagnostic. A large number of negative values indicates too few MC samples, but there is no way to detect this.

**Fix:** Added warning before clipping:
```python
n_negative_mi = int((mi < 0).sum())
if n_negative_mi > 0:
    print(f"  Warning: {n_negative_mi}/{len(mi)} MI values were negative (clipped to 0); "
          f"consider increasing n_mc_samples.")
mi = np.clip(mi, 0, None)
```

---

### B3-07 · `uncertainty/mc_dropout.py:116` · #114 — `np.roll` axis undocumented

**Problem:** `np.roll(X[i], s, axis=0)` — it is unclear which axis is being rolled without knowing the data format `(T, C)`.

**Fix:** Added trailing comment: `# axis=0 is the time (sample) axis`.

---

### B3-08 · `uncertainty/mc_dropout.py:253` · #110 — Bar width hard-coded

**Problem:** `width=0.08` is fixed regardless of `n_bins`. For `n_bins=10`, bars should be `0.1` wide; the current `0.08` leaves gaps at every bin boundary.

**Fix:** `width=1.0/n_bins` (uses the function parameter directly).

---

### B3-09 · `uncertainty/deep_ensemble.py:27` · #152 — Unused import

**Problem:** `import copy` is never used. Its presence implies that weight perturbation (copy + perturb) was planned but never implemented, which misleads readers about what this "ensemble" actually does.

**Fix:** Removed the unused import.

---

### B3-10 · `uncertainty/deep_ensemble.py:130` · #111 — `std_max` ignores class identity

**Problem:** `probs_ens.max(axis=2).std(axis=0)` first takes the per-member max confidence (ignoring which class that confidence belongs to), then computes the std. Two ensemble members can disagree completely on the predicted class yet show the same max confidence, yielding `std ≈ 0` — the exact opposite of the intended "high disagreement" signal.

**Fix:**
```python
# Before
std_max = probs_ens.max(axis=2).std(axis=0)

# After — mean std across class probabilities per sample
std_max = probs_ens.std(axis=0).mean(axis=1)
```
`probs_ens.std(axis=0)` gives per-sample per-class probability std across members `(N, C)`. `.mean(axis=1)` reduces to a per-sample scalar, correctly capturing member disagreement at the class level.

---

### B3-11 · `compare_fgsm_baselines.py:255-280` · #82 — Wrong loss for LSTM/BiLSTM

**Problem:** One `RLSTMLoss(use_focal=True)` is created and shared across ALL models. LSTM and BiLSTM were trained with `CrossEntropyLoss`; using focal loss to generate FGSM perturbations biases the attack gradient direction and makes FGSM-drop values non-comparable across models.

**Fix:** Two separate criterion objects are created:
- `hmr_criterion = RLSTMLoss(use_focal=True)` — for all `HMR-BiLSTM` variants
- `baseline_criterion = RLSTMLoss(use_focal=False)` — effectively `CrossEntropyLoss`, for LSTM/BiLSTM

The correct criterion is selected inside the model loop based on `model_name`.

---

### B3-12 · `run_ablation.py:543` · #91 — Seed not reset per variant

**Problem:** `set_seed(cfg["seed"])` was called once before the variant for-loop. Subsequent variants inherited the RNG state left by the previous training run, making their initialization non-reproducible and dependent on execution order.

**Fix:** Moved `set_seed(cfg["seed"])` inside the loop, before each `train_variant` call.

---

### B3-13 · `run_ablation.py:184-186` · #95 — 5-class AUC

**Problem:** `roc_auc_score(y_true, probs, multi_class="ovr", average="macro")` uses all 5 class columns. Class Q (label=4) is nearly absent from the inter-patient test set; when Q is missing `roc_auc_score` raises `ValueError` and the except clause returns `0.0`, silently replacing valid AUC values with zero in the ablation table.

**Fix:**
```python
# Before
roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

# After
roc_auc_score(y_true, probs[:, :4], multi_class="ovr", average="macro",
              labels=[0, 1, 2, 3])
```

---

### B3-14 · `run_reproducible_pipeline.py:57-66` · #127 — AutoAttack missing

**Problem:** `evaluate_autoattack.py` was present in `run_all.bat` (step 7.5) but absent from `run_reproducible_pipeline.py`, the Python orchestrator. This means the Python pipeline produces incomplete robustness results without any error.

**Fix:** Added `evaluate_autoattack.py` as step 8 (between PGD step 7 and ablation robustness step 9). All subsequent step numbers incremented by 1 (9→15).

---

### B3-15 · `evaluate_trustworthiness.py:287` · #128 — Wrong FGSM JSON filename

**Problem:** T8 reads `results/logs/fgsm_comparison_results.json` but `compare_fgsm_baselines.py` writes to `results/logs/fgsm_baseline_comparison.json`. The file T8 looks for is never written by any script, so the FGSM robustness row in T8 is always N/A.

**Fix:** Changed to read `results/logs/fgsm_baseline_comparison.json`.

---

### B3-16 · `test_speed.py:9-13` · #134 — Permanent CONFIG mutation

**Problem:** `train_inter_patient.CONFIG["epochs"] = 1` mutates the module-level dict in place. If any other module imports `train_inter_patient` in the same process after `test_speed.py` is run, it sees `epochs=1`, silently corrupting training of any model that follows.

**Fix:** Save and restore the original value in a try/finally block:
```python
_original_epochs = train_inter_patient.CONFIG["epochs"]
train_inter_patient.CONFIG["epochs"] = 1
try:
    train_inter_patient.main()
finally:
    train_inter_patient.CONFIG["epochs"] = _original_epochs
```

---

### B3-17 · `explainability/plot_disagreements.py:10-13` · #120 — Hardcoded run ID

**Problem:** `out_dir = Path("outputs/v1.0_20260616_061207/explainability")` is hardcoded. Rerunning any experiment with a different run ID causes the script to silently read stale results from a prior session.

**Fix:** Load `configs/experiment_config.yaml`, call `get_run_id(cfg)`, and derive the path from `build_paths(run_id)["out_explain"]`.

---

### B3-18 · `explainability/integrated_gradients.py:200` · #57 — Insufficient n_steps

**Problem:** `n_steps=50` is below the commonly-cited minimum of 200 steps for stable Integrated Gradients attribution in models with non-smooth activations (MaxPool, sigmoid attention). The approximation error in the Riemann sum is O(1/n_steps).

**Fix:** `n_steps = 200`

---

### B3-19 · `explainability/shap_analysis.py:206` · #115 — Silent background cap

**Problem:** `n_background = min(n_background, 100)` silently ignores the user config if `shap_background_samples > 100`. No warning is emitted, making it impossible to distinguish "ran with 100 background samples" from "ran with the configured value" in logs.

**Fix:** Added warning print before capping:
```python
if n_background > 100:
    print(f"Warning: n_background capped from {n_background} to 100 (GradientExplainer budget limit).")
```

---

### B3-20 · `explainability/shap_analysis.py:338-341` · #123 — Legacy NumPy RNG

**Problem:** `np.random.seed(run_seed)` inside the SHAP averaging loop mutates the global NumPy RNG state. If any other function is called between iterations that uses the global RNG, the background sample selection becomes non-reproducible.

**Fix:** Replaced with `rng_run = np.random.default_rng(run_seed)` and `bg_idx = rng_run.choice(...)`, isolating the RNG state to the local variable.

---

### B3-21 · `generate_results_tables.py:215-232` · #125 — Rec-F missing from FGSM table

**Problem:** `table_fgsm()` reported Rec-S and Rec-V per model but omitted Rec-F (Fusion class). Fusion has the worst adversarial robustness in most ablations; its omission is misleading. The data already exists in the FGSM dict as `rec_f_c` and `rec_f_a`.

**Fix:** Added `"Rec-F clean"` and `"Rec-F adv"` to `header` and the corresponding `d.get("rec_f_c")` / `d.get("rec_f_a")` values to each row. Uses `.get(..., float("nan"))` to degrade gracefully on older CSVs that may lack the column.

---

### B3-22 · `hmr_bilstm_ablation.py:224` · #87 — Dead commented-out code

**Problem:** `# self.use_interaction = use_interaction` is a dead comment that suggests the attribute should exist but was intentionally removed. It confuses readers about whether `use_interaction` can be accessed on the instance.

**Fix:** Removed the comment.

---

### B3-23 · `robustness/auto_attack.py:247-255` · #70 — Fragile shape routing

**Problem:** The original `AAWrapper.forward` used a sequence of `squeeze` + `dim == N` checks to convert AutoAttack's 4D input `(N, 1, T, 1)` back to model input `(N, T, 1)`. The nested checks were easy to break silently if AutoAttack changed its shape convention.

**Fix:** Replaced with explicit indexing:
```python
if x.dim() == 4:
    x = x[:, 0, :, 0]   # (N, T)
elif x.dim() == 3 and x.shape[1] == 1:
    x = x[:, 0, :]      # (N, T)
x = x.unsqueeze(-1)      # (N, T, 1)
```
This is unambiguous about which dimension is being stripped.

---

### B3-24 · `validation/preprocess_aami.py:78-80` · #99 — Pickle-dependent npz

**Problem:** `np.savez_compressed(..., data=all_beats)` where `all_beats` is a list of dicts saves a numpy object array. Loading it later requires `allow_pickle=True`, which is a security risk and fails with the default `allow_pickle=False` in NumPy ≥ 1.16.3.

**Fix:** Saved as `all_extracted_beats.json` using `json.dump`. This file is an intermediate diagnostic artifact only; no other script loads it by name.

---

## False Positives Identified

### FP — #130 · `evaluate_calibration.py`

The audit claimed this file never writes to `outputs/<run_id>/calibration/results.json`. After reading the file: line 179 already writes `results_json_path = paths["out_calib"] / "results.json"` followed by `json.dump(...)`. The write was present and correct. No change needed; status updated to FALSE POSITIVE.

### FP — #145 · `preprocess.py:94`

The audit claimed mixed tab/space indentation. After scanning the file binary, no tab characters exist anywhere in `preprocess.py`. All indentation uses spaces. No change needed; status updated to FALSE POSITIVE.

---

## Files Changed

| File | Issues |
|------|--------|
| `hmr_bilstm.py` | #31 |
| `calibration/temperature_scaling.py` | #47, #106 |
| `uncertainty/mc_dropout.py` | #50, #51, #52, #114, #110 |
| `uncertainty/deep_ensemble.py` | #152, #111 |
| `compare_fgsm_baselines.py` | #82 |
| `run_ablation.py` | #91, #95 |
| `run_reproducible_pipeline.py` | #127 |
| `evaluate_trustworthiness.py` | #128 |
| `test_speed.py` | #134 |
| `explainability/plot_disagreements.py` | #120 |
| `explainability/integrated_gradients.py` | #57 |
| `explainability/shap_analysis.py` | #115, #123 |
| `generate_results_tables.py` | #125 |
| `hmr_bilstm_ablation.py` | #87 |
| `robustness/auto_attack.py` | #70 |
| `validation/preprocess_aami.py` | #99 |

---

## What Was Deliberately Skipped

| Issue | Reason |
|-------|--------|
| #5, #6, #7, #8, #12, #21 | Experimental protocol changes — covered in `P1_PROTOCOL_UNIFICATION_PLAN.md` |
| #121 | SHAP batching — complex to add correctly without knowing available RAM; OOM is a risk not a correctness bug; deferred to when shap_analysis.py is refactored |
| All "Requires Retraining" issues | Per task constraint |

---

*Generated: 2026-06-24 (session 3)*  
*Issue tracking: 154 total — 69 FIXED, 10 FALSE POSITIVE, 75 NOT FIXED*
