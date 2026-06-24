# Retraining Checklist

**Purpose:** Ordered execution plan to reproduce all results under the unified inter-patient
evaluation protocol. Steps must be completed in the order listed because later steps consume
outputs produced by earlier ones.

**Prerequisite:** All code changes listed in `P1_PROTOCOL_UNIFICATION_PLAN.md` §3 must be
applied before running any step.

---

## Phase 0: Code Changes (Must Complete Before Anything Else)

Apply the following changes (not yet done — no code has been written for these):

- [ ] **`report_results.py:277`** — Change `data/processed/test.npz` → `data/processed/splits/inter_test.npz`
- [ ] **`report_results.py:284`** — Change `best_rlstm.pt` → `inter_best_rlstm.pt`
- [ ] **`report_results.py:307–309`** — Add `labels=[0,1,2,3]` to all four sklearn metrics
- [ ] **`plot_case_visualization.py:36`** — Change `TEST_DATA` to `inter_test.npz`
- [ ] **`evaluate_robustness_all.py:31,58`** — Add `labels=[0,1,2,3]` to `f1_score` calls
- [ ] **`run_baselines.py:587,596,611`** — Change `epochs=12` → `epochs=45`
- [ ] **`run_baselines.py:~562`** — Generate class weights from `inter_train.npz`
- [ ] **`evaluate_autoattack.py`** — Remove n=200 pre-filtering block

---

## Phase 1: Data Preparation

### Step 1.1 — Generate Inter-patient Splits
```
python validation/preprocess_aami.py
```
**Produces:**
- `data/processed/splits/inter_train.npz`
- `data/processed/splits/inter_val.npz`
- `data/processed/splits/inter_test.npz`
- `data/processed/splits/inter_norm_mean.npy`
- `data/processed/splits/inter_norm_std.npy`

**Prerequisite:** Raw MIT-BIH WFDB files must exist at `data/raw/mitdb/` (all 48 records).  
**Notes:** This script also regenerates intra-patient splits (now stratified, post Fix 28).
Verify class distributions in the printed output — DS2 test set should have heavy class imbalance
(N >> S, V, F, Q).

- [ ] Step complete

---

## Phase 2: Model Training

### Step 2.1 — Train HMR-BiLSTM on Inter-patient Split
```
python train_inter_patient.py
```
**Produces:**
- `results/checkpoints/inter_best_rlstm.pt` (canonical checkpoint)

**Config:** `configs/experiment_config.yaml` — epochs=45, lr=1e-3, dropout=0.3  
**Notes:** Checkpoint is saved on best val F1 (4-class macro, `labels=[0,1,2,3]`). Verify the
saved checkpoint metadata contains the val_f1_macro field before proceeding to Phase 3.

- [ ] Step complete

### Step 2.2 — Retrain All Baselines (Epoch Parity)
```
python run_baselines.py
```
**Produces:**
- `results/logs/baseline_results.json` (overwrites old version)
- Trained checkpoint files for LSTM, BiLSTM, Logistic Regression, Decision Tree

**Config:** After Phase 0 fix, baselines will use epochs=45 and inter-patient splits.  
**Notes:** Runtime will be ~3.75× longer than current 12-epoch run. The file
`baseline_results.json` is consumed by `generate_results_tables.py` → ensure it is complete
before Phase 7.

- [ ] Step complete

---

## Phase 3: Clean Performance Evaluation

### Step 3.1 — Evaluate HMR-BiLSTM (report_results.py)
```
python report_results.py
```
**Produces:**
- `results/logs/baseline_results.json` — adds `"hmr_bilstm"` entry (4-class macro F1, inter-patient)
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/gate_trajectories.png`
- `results/figures/comparison_bars.png`
- `results/figures/final_results_table.png`

**After Phase 0 fix:** Evaluates `inter_best_rlstm.pt` on `inter_test.npz` with `labels=[0,1,2,3]`.  
**Verify:** `hmr_bilstm.f1_macro` in `baseline_results.json` should be approximately 0.88.
If it is ≈ 0.56, Phase 0 code changes were not applied.

- [ ] Step complete

---

## Phase 4: Adversarial Robustness Evaluation

All robustness scripts use `inter_best_rlstm.pt` and `inter_test.npz` already (or were fixed
in session 2). Run in any order after Phase 2.1.

### Step 4.1 — FGSM Robustness
```
python evaluate_fgsm.py
```
**Produces:**
- `results/tables/fgsm_results.csv`
- `results/tables/fgsm_baseline_summary.csv`
- `results/tables/fgsm_baseline_comparison.csv`
- `results/figures/fgsm_*.png`

- [ ] Step complete

### Step 4.2 — PGD Robustness
```
python evaluate_pgd.py
```
**Produces:**
- `results/tables/pgd_baseline_comparison.csv`
- `results/figures/pgd_*.png`

- [ ] Step complete

### Step 4.3 — AutoAttack (Full Test Set)
```
python evaluate_autoattack.py
```
**After Phase 0 fix:** Runs on the full `inter_test.npz` set — no pre-filtering.  
**Produces:**
- `outputs/v1.0_FINAL/robustness/autoattack_results.json` (replaces n=200 subset version)

**Warning:** Estimated runtime 1–4 hours on GPU. Consider a stratified subsample of n=2000
(seed=42) if full-set runtime is prohibitive; document subsample size in the paper.

- [ ] Step complete

### Step 4.4 — C&W Attack (Full Test Set)
```
python robustness/cw_attack.py
```
**After Phase 0 fix:** Runs on full test set — no pre-filtering.  
**Produces:**
- `outputs/v1.0_FINAL/robustness/cw_attack_results.json` (replaces n=200 subset version)

- [ ] Step complete

### Step 4.5 — Noise Robustness Sweep
```
python evaluate_robustness_all.py
```
**After Phase 0 fix:** Now uses `labels=[0,1,2,3]` for F1 (5-class fixed).  
**Produces:**
- `results/figures/robustness_noise_all.png`
- `results/figures/robustness_summary.png`

- [ ] Step complete

---

## Phase 5: Calibration and Uncertainty Evaluation

### Step 5.1 — Calibration (ECE + Temperature Scaling)
```
python evaluate_calibration.py
```
**Produces:**
- `results/tables/calibration_results.csv`
- `results/figures/reliability_diagram.png`

**Notes:** After session 2 Fix 15, this now fits temperature scaling on the inter-patient
val set and reports both `ECE_uncalibrated` and `ECE_calibrated`.

- [ ] Step complete

### Step 5.2 — MC Dropout Uncertainty
```
python evaluate_trustworthiness.py
```
**Produces:**
- `outputs/v1.0_FINAL/trustworthiness/mc_dropout_results.json`
- `results/figures/trustworthy_ai_summary.png`

- [ ] Step complete

---

## Phase 6: Explainability Evaluation

Run after Phase 2.1 (needs the canonical checkpoint).

### Step 6.1 — SHAP Analysis
```
python explainability/shap_analysis.py
```
**Produces:**
- `outputs/v1.0_FINAL/explainability/shap_importance_ranking.csv`
- SHAP per-class figures

- [ ] Step complete

### Step 6.2 — Integrated Gradients
```
python explainability/integrated_gradients.py
```
**Produces:**
- `outputs/v1.0_FINAL/explainability/ig_importance.npy`

- [ ] Step complete

### Step 6.3 — Gradient Cosine Similarity (approx TracIn)
```
python explainability/data_attribution.py
```
- [ ] Step complete

---

## Phase 7: Table and Figure Generation

Run only after all preceding phases are complete.

### Step 7.1 — Generate All Results Tables
```
python generate_results_tables.py
```
**Produces (replaces all contaminated versions):**
- `results/tables/final_results.csv`
- `results/tables/final_results.tex`
- `results/tables/baseline_full_comparison.csv`
- `results/tables/table2_fgsm_robustness.csv` + `.tex`
- `results/tables/table3_calibration.csv` + `.tex`
- `results/tables/table4_pgd_robustness.csv` + `.tex`
- `results/tables/table5_consolidated.csv` + `.tex`
- `results/figures/trustworthy_ai_summary.png`

**Verify table5_consolidated.csv:** The FGSM-drop and PGD-drop columns must be positive
(Clean-F1 > adversarial-F1). If either is negative, the Phase 0 code changes were not applied
correctly to `report_results.py`.

- [ ] Step complete

### Step 7.2 — Ablation Study
```
python run_ablation.py
```
**Produces:**
- `results/tables/ablation_table_final.csv`
- `results/tables/ablation_robustness.csv`
- `results/figures/ablation_clean_vs_adv.png`
- `results/figures/ablation_f1_drop.png`

- [ ] Step complete

### Step 7.3 — Case Visualizations
```
python plot_case_visualization.py
```
**After Phase 0 fix:** Uses `inter_test.npz`.  
**Produces:**
- `results/figures/case_*.png`

- [ ] Step complete

### Step 7.4 — Final Output Collection
```
python gather_final.py
```
Copies all final tables/figures to the output directory for paper submission.

- [ ] Step complete

---

## Verification Checklist

After all phases complete, verify these sentinel values:

| File | Key | Expected range | Flag if |
|------|-----|---------------|---------|
| `results/logs/baseline_results.json["hmr_bilstm"]["f1_macro"]` | Clean F1 | 0.85–0.92 | < 0.70 or > 0.98 |
| `results/tables/table5_consolidated.csv` row HMR-BiLSTM | `FGSM-drop` | 0.01–0.20 | < 0 (impossible) |
| `results/tables/table5_consolidated.csv` row HMR-BiLSTM | `PGD-drop` | 0.01–0.25 | < 0 (impossible) |
| `results/tables/calibration_results.csv` row HMR-BiLSTM | `ECE_uncalibrated` | 0.05–0.20 | > 0.40 |
| `outputs/v1.0_FINAL/robustness/autoattack_results.json` | `n_samples` | > 1000 | == 200 (pre-filter not removed) |

---

*Generated: 2026-06-24 (session 2 — Part 2)*
