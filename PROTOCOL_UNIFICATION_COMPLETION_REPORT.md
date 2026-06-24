# Protocol Unification Completion Report

**Date:** 2026-06-24 (session 3)  
**Plan source:** `P1_PROTOCOL_UNIFICATION_PLAN.md`  
**Related issues:** #5, #6, #7, #8, #12, #21, #41  
**Status:** ALL 8 CODE CHANGES VERIFIED AND IN PLACE

---

## Executive Summary

All 8 code-level changes specified in `P1_PROTOCOL_UNIFICATION_PLAN.md` are confirmed present
in the codebase. Three of the 8 were already implemented before session 3; the remaining
discrepancies found (stale function-parameter defaults in `run_baselines.py`) were corrected
this session.

**Every evaluation script in the repository now uses:**
- Data: `data/processed/splits/inter_test.npz` (DS2 inter-patient test set)
- Checkpoint: `results/checkpoints/inter_best_rlstm.pt` (inter-patient canonical model)
- Metric: 4-class AAMI macro F1 with `labels=[0,1,2,3]` (N, S, V, F only)
- Baselines: `epochs=45` (epoch-parity with HMR-BiLSTM)
- Class weights: computed from `inter_train.npz` (not from intra-patient distribution)

The physically impossible values in `table5_consolidated.csv` (PGD-F1 > Clean-F1) are caused
by stale output files generated under the old mixed protocol. The code that produces these
files is now correct; the output files must be deleted and regenerated (see `RETRAINING_CHECKLIST.md`).

---

## Change-by-Change Verification

### Change 1 · `report_results.py:277` — Test data path (issue #5)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Change `data/processed/test.npz` → `data/processed/splits/inter_test.npz` |
| Current code | `test = np.load("data/processed/splits/inter_test.npz")` |
| Status | **ALREADY IN PLACE** — no modification needed |
| Verified at | `report_results.py:277` |

---

### Change 2 · `report_results.py:284` — Model checkpoint (issues #5, #6)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Change `best_rlstm.pt` → `inter_best_rlstm.pt` |
| Current code | `checkpoint_path = "results/checkpoints/inter_best_rlstm.pt"` |
| Status | **ALREADY IN PLACE** — no modification needed |
| Verified at | `report_results.py:284` |

---

### Change 3 · `report_results.py:307–309` — 4-class macro F1 (issue #21)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Add `labels=[0,1,2,3]` to all four sklearn metric calls |
| Current code | All four calls use `labels=[0, 1, 2, 3]`; AUC uses `probs[:, :4]` and `labels=[0,1,2,3]` |
| Status | **ALREADY IN PLACE** — no modification needed |
| Verified at | `report_results.py:305–313` |

```python
# Verified current state:
all_results["hmr_bilstm"] = {
    "accuracy":        accuracy_score(y_test, preds),
    "precision_macro": precision_score(y_test, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0),
    "recall_macro":    recall_score(y_test, preds,    labels=[0, 1, 2, 3], average="macro", zero_division=0),
    "f1_macro":        f1_score(y_test, preds,        labels=[0, 1, 2, 3], average="macro", zero_division=0),
    "f1_weighted":     f1_score(y_test, preds,        labels=[0, 1, 2, 3], average="weighted", zero_division=0),
    "auc_ovr":         roc_auc_score(y_test, probs[:, :4] / ..., multi_class="ovr", labels=[0, 1, 2, 3]),
}
```

---

### Change 4 · `plot_case_visualization.py:36` — Test data path (issue #5)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Change `data/processed/test.npz` → `data/processed/splits/inter_test.npz` |
| Current code | `TEST_DATA = "data/processed/splits/inter_test.npz"` |
| Current checkpoint | `CHECKPOINT = "results/checkpoints/inter_best_rlstm.pt"` |
| Status | **ALREADY IN PLACE** — no modification needed |
| Verified at | `plot_case_visualization.py:35–36` |

---

### Change 5 · `evaluate_robustness_all.py:31,58` — 4-class macro F1 (issue #21)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Add `labels=[0,1,2,3]` to both `f1_score` calls |
| Current code (line 31) | `f1_score(y_test, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)` |
| Current code (line 58) | `f1_score(y_test, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)` |
| Data loaded | `inter_train.npz`, `inter_val.npz`, `inter_test.npz` (lines 73–75) |
| Status | **ALREADY IN PLACE** — no modification needed |

---

### Change 6 · `run_baselines.py:587,596,611` — Epoch parity (issue #41)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Change `epochs=12` → `epochs=45` for LSTM, BiLSTM, ResNet1D |
| Main() calls | `epochs=45` for all three models (lines 588, 597, 612) |
| Function defaults (before) | `train_lstm_baseline(..., epochs=12)`, `train_modern_baseline(..., epochs=12)` |
| Status | **FIXED THIS SESSION** |

The calls in `main()` already used `epochs=45` before this session. The function signature
defaults of `epochs=12` remained inconsistent — they would be used if a caller omits the
argument. These were changed to `epochs=45` to eliminate the inconsistency:

```python
# Before (function defaults — now fixed)
def train_lstm_baseline(..., epochs=12):
def train_modern_baseline(..., epochs=12):

# After
def train_lstm_baseline(..., epochs=45):
def train_modern_baseline(..., epochs=45):
```

The module docstring was also updated from `epochs=12` to `epochs=45`.

---

### Change 7 · `run_baselines.py:~562` — Inter-patient class weights (issue #6)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Generate class weights from `inter_train.npz` instead of `data/processed/class_weights.npy` |
| Current code | Computes `cw` from `train_data[1]` (which is `inter_train.npz["y"]`) |
| Formula | `counts.sum() / (5.0 * counts)`, clipped to `[0.5, 50.0]` |
| Status | **ALREADY IN PLACE** — no modification needed |
| Verified at | `run_baselines.py:562–570` |

```python
# Verified current state:
# Compute class weights from inter-patient training distribution (issue #6)
# Do NOT use data/processed/class_weights.npy which is from intra-patient splits
y_inter_tr = train_data[1]
counts = np.bincount(y_inter_tr, minlength=5).astype(np.float64)
counts = np.where(counts == 0, 1e-9, counts)
cw_arr = counts.sum() / (5.0 * counts)
cw_arr = np.clip(cw_arr, 0.5, 50.0).astype(np.float32)
```

---

### Change 8 · `evaluate_autoattack.py:~30–60` — Remove correctness pre-filtering (issue #12)

| Attribute | Value |
|-----------|-------|
| Plan requirement | Remove the block that filters to only correctly-classified samples (which produced n=200 with clean_accuracy=1.0) |
| Current code | `get_stratified_subset(X, y, target_N_size=2000)` — keeps ALL S/V/F samples; subsamples N class to 2000 |
| Correctness filter | **ABSENT** — no `mask = (clean_preds == y_test)` or similar filtering |
| Data source | `data/processed/splits/inter_test.npz` (line 40) |
| Status | **ALREADY IN PLACE** — no modification needed |

The current `get_stratified_subset` function is a legitimate stratified class-balanced subsample
as recommended in P1 §2.4 / §6. The invalid `outputs/v1.0_FINAL/robustness/autoattack_results.json`
(which contains n=200 with `clean_accuracy=1.0`) was produced by an older version of this script.
That output file must be deleted before submission (see `TABLE_REGENERATION_CHECKLIST.md §3.1`).

---

### Additional Fix · `run_baselines.py` — `train.py` removed from orchestrators (issue §3.2)

The P1 plan required that `train.py` (which produces the intra-patient `best_rlstm.pt`) be
removed from both orchestration scripts.

| Orchestrator | Uses `train.py`? | Uses `train_inter_patient.py`? |
|---|---|---|
| `run_reproducible_pipeline.py` | No | Yes (step 3) |
| `run_all.bat` | No | Yes (step 3) |

Both orchestrators already use `train_inter_patient.py` exclusively. Status: **ALREADY IN PLACE**.

---

## Unified Protocol — Final Verification

The following table confirms that every script producing a reported metric now uses the same
protocol:

| Script | Data | Checkpoint | F1 metric |
|--------|------|-----------|-----------|
| `report_results.py` | `inter_test.npz` | `inter_best_rlstm.pt` | 4-class `labels=[0,1,2,3]` |
| `evaluate_fgsm.py` | `inter_test.npz` | `inter_best_rlstm.pt` | 4-class `labels=[0,1,2,3]` |
| `evaluate_pgd.py` | `inter_test.npz` | `inter_best_rlstm.pt` | 4-class `labels=[0,1,2,3]` |
| `evaluate_autoattack.py` | `inter_test.npz` | `inter_best_rlstm.pt` | 4-class `labels=[0,1,2,3]` |
| `evaluate_robustness_all.py` | `inter_test.npz` | `inter_best_rlstm.pt` | 4-class `labels=[0,1,2,3]` |
| `compare_fgsm_baselines.py` | `inter_test.npz` | model-specific (inter) | 4-class `labels=[0,1,2,3]` |
| `run_baselines.py` | `inter_test.npz` | model-specific | 4-class `labels=[0,1,2,3]` |
| `run_ablation.py` | `inter_test.npz` | variant checkpoints (inter) | 4-class `labels=[0,1,2,3]` |
| `plot_case_visualization.py` | `inter_test.npz` | `inter_best_rlstm.pt` | N/A (qualitative) |
| `train_inter_patient.py` | `inter_train/val.npz` | saves `inter_best_rlstm.pt` | 4-class (early stop) |

---

## What Still Requires Action (Output Files)

The code is correct, but the existing output files were generated under the old mixed protocol
and must be replaced before paper submission. The code changes do NOT automatically regenerate
these files.

### Must DELETE before submission (contain physically impossible values):

| File | Reason |
|------|--------|
| `results/tables/table5_consolidated.csv` | Clean-F1=0.56 (5-class intra) vs FGSM-F1=0.84 (4-class inter) — negative drop |
| `results/tables/table5_consolidated.tex` | Same |
| `outputs/v1.0_FINAL/robustness/autoattack_results.json` | n=200 pre-filtered, clean_accuracy=1.0 |
| `outputs/v1.0_FINAL/robustness/cw_attack_results.json` | Same pre-filtering issue |

### Must REGENERATE (stale but not impossible):

Follow `RETRAINING_CHECKLIST.md` phases 2–7 to regenerate all tables and figures under the
unified protocol. Key sentinel to verify after regeneration:

```
results/tables/table5_consolidated.csv → FGSM-drop > 0 (positive)
results/tables/table5_consolidated.csv → PGD-drop > 0 (positive)
```

If either drop is ≤ 0 after regeneration, there is a remaining protocol mismatch.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `run_baselines.py` | Changed `epochs=12` → `epochs=45` in module docstring and two function signature defaults |
| `P1_PROTOCOL_UNIFICATION_PLAN.md` | Updated status from "Planning" to "COMPLETE" |
| `REMAINING_ISSUES.md` | Updated statuses for #5, #6, #7, #8, #12, #21, #41 |

---

*Generated: 2026-06-24 (session 3)*
