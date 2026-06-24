# HMR-BiLSTM: Comprehensive Architecture Report

**Project:** Hybrid Memory Residual Bidirectional LSTM for Trustworthy ECG Arrhythmia Classification  
**Dataset:** MIT-BIH Arrhythmia Database (Moody & Mark, 2001)  
**Report Date:** 2026-06-23  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Core Model Architecture](#4-core-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation & Trustworthiness Framework](#6-evaluation--trustworthiness-framework)
7. [Explainability Subsystem](#7-explainability-subsystem)
8. [Uncertainty Quantification Subsystem](#8-uncertainty-quantification-subsystem)
9. [Calibration Subsystem](#9-calibration-subsystem)
10. [Robustness Subsystem](#10-robustness-subsystem)
11. [Results & Visualization Pipeline](#11-results--visualization-pipeline)
12. [Orchestration & Reproducibility](#12-orchestration--reproducibility)
13. [Directory & Module Map](#13-directory--module-map)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Strengths](#15-strengths)
16. [Weaknesses & Limitations](#16-weaknesses--limitations)
17. [Publication Recommendations](#17-publication-recommendations)

---

## 1. Executive Summary

HMR-BiLSTM is a **trustworthy, explainable deep-learning framework** for 5-class ECG arrhythmia classification on the MIT-BIH dataset (N / S / V / F / Q). Its core contribution is the **Hybrid Memory Residual (RMC) Cell**: a novel LSTM variant that introduces a learnable blend gate β between a residual memory-control (RMC) path and a standard LSTM path. This design preserves LSTM expressiveness as a fallback while allowing the model to learn longer-range selective memory when beneficial.

Beyond the model architecture, the framework provides a **complete trustworthiness pipeline** spanning:

| Pillar | Methods Implemented |
|--------|-------------------|
| Accuracy | CNN + BiRLSTM + Attention, Focal Loss + class weights |
| Robustness | FGSM (train-time), FGSM / PGD / AutoAttack / CW (eval) |
| Calibration | ECE, Brier Score, NLL, Temperature Scaling |
| Uncertainty | MC Dropout, Deep Ensembles, OOD corruption sweep |
| Explainability | SHAP (GradientExplainer), Integrated Gradients (Captum), TracIn / data attribution |
| Reproducibility | Fixed seeds, deterministic CUDA, versioned run IDs |

---

## 2. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HMR-BiLSTM SYSTEM                            │
│                                                                     │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │  MIT-BIH    │   │   preprocess.py  │   │   data/processed/    │ │
│  │  CSV Data   │──▶│  Z-score / split │──▶│  train/val/test.npz  │ │
│  │  (raw/)     │   │  class weights   │   │  class_weights.npy   │ │
│  └─────────────┘   └──────────────────┘   └──────────┬───────────┘ │
│                                                        │             │
│                         ┌──────────────────────────────▼──────┐     │
│                         │         TRAINING LAYER               │     │
│                         │  ┌────────────────┐  ┌───────────┐  │     │
│                         │  │   train.py     │  │run_baselin│  │     │
│                         │  │  HMR-BiLSTM   │  │  es.py    │  │     │
│                         │  │  +FGSM train  │  │LR/DT/LSTM/│  │     │
│                         │  └──────┬─────────┘  │BiLSTM/Res │  │     │
│                         │         │             │Net1D      │  │     │
│                         │  ┌──────▼─────────┐  └──────┬────┘  │     │
│                         │  │ run_ablation.py│         │        │     │
│                         │  │  6 variants    │         │        │     │
│                         │  └──────┬─────────┘         │        │     │
│                         └─────────┼───────────────────┼────────┘     │
│                                   │                   │              │
│              ┌────────────────────▼───────────────────▼─────────┐   │
│              │                  CHECKPOINTS                       │   │
│              │  results/checkpoints/best_rlstm.pt                │   │
│              │  results/checkpoints/{lstm,bilstm,resnet1d}.pt    │   │
│              │  results/ablation/checkpoints/best_rlstm_*.pt     │   │
│              └──────────────────────────┬────────────────────────┘   │
│                                         │                            │
│              ┌──────────────────────────▼────────────────────────┐   │
│              │              EVALUATION LAYER                      │   │
│              │  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │   │
│              │  │Robustness│  │Calibration│  │Explainability│   │   │
│              │  │FGSM/PGD/ │  │ECE/Brier/ │  │SHAP/IG/TracIn│  │   │
│              │  │AutoAttack│  │Temp Scale │  │              │   │   │
│              │  └────┬─────┘  └─────┬─────┘  └──────┬───────┘   │   │
│              │       │              │                │            │   │
│              │  ┌────▼──────────────▼────────────────▼─────────┐ │   │
│              │  │             Uncertainty                        │ │   │
│              │  │       MC Dropout / Deep Ensemble               │ │   │
│              │  └───────────────────────────────────────────────┘ │   │
│              └──────────────────────────┬────────────────────────┘   │
│                                         │                            │
│              ┌──────────────────────────▼────────────────────────┐   │
│              │              RESULTS LAYER                         │   │
│              │  results/tables/  results/figures/  outputs/v*/   │   │
│              │  *.csv  *.tex     *.png             *.json        │   │
│              └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

**File:** `preprocess.py`

### 3.1 Input Format

MIT-BIH arrhythmia beats, pre-segmented into fixed-length windows of **187 samples** at 360 Hz. Labels follow the AAMI EC57 mapping:

| Class | Label | Condition | Approx. count |
|-------|-------|-----------|---------------|
| N | 0 | Normal / Left/Right bundle branch | ~90,000 |
| S | 1 | Supraventricular ectopic | ~2,500 |
| V | 2 | Ventricular ectopic | ~7,000 |
| F | 3 | Fusion | ~800 |
| Q | 4 | Unknown / Paced | ~8,000 |

Imbalance ratio: **~1:100** (F vs N). This is the dominant challenge for the dataset.

### 3.2 Processing Steps

```
Raw CSV (train/test)
       │
       ▼
  Load & inspect class distribution
       │
       ▼
  [Optional] Stratified subsampling
       │
       ▼
  Train → 85% train_split + 15% val_split  (stratified)
       │
       ▼
  Z-score normalization
  ┌───────────────────────────────────┐
  │  μ = train.mean(), σ = train.std()│
  │  Applied to train, val, AND test  │
  │  Saved: norm_mean.npy, norm_std.npy│
  │  → No data leakage from val/test  │
  └───────────────────────────────────┘
       │
       ▼
  Reshape: (N, 187) → (N, 187, 1)
       │
       ▼
  Class weights: w_c = N_total / (C * N_c)
  Clipped to [0.5, 10.0]
       │
       ▼
  Save:
  ├── data/processed/train.npz
  ├── data/processed/val.npz
  ├── data/processed/test.npz
  └── data/processed/class_weights.npy
```

### 3.3 Validation Submodule

`validation/` contains independent verification scripts:
- `verify_normalization.py` — asserts μ≈0, σ≈1 on train split
- `evaluate_splits.py` — checks stratification ratios across splits
- `download_mitdb.py` — utility to fetch raw data
- `preprocess_aami.py` — alternate AAMI-standard preprocessing path

---

## 4. Core Model Architecture

**File:** `hmr_bilstm.py`

### 4.1 Full Architecture Diagram

```
Input: (B, 187, 1)
       │
       ▼
┌──────────────────────────────────────┐
│       ECGFeatureExtractor (CNN)       │
│                                      │
│  Conv1d(1→32, k=5, pad=2) + BN      │
│       ↓ MaxPool(2) → (B, 93, 32)    │
│  Conv1d(32→64, k=3, pad=1) + BN     │
│       ↓ MaxPool(2) → (B, 46, 64)    │
└──────────────────┬───────────────────┘
                   │ (B, 46, 64)
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    BiRLSTM (num_layers=2)                      │
│                                                              │
│  Layer 1:                                                    │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  RLSTMLayer (forward)   │  │  RLSTMLayer (backward)  │   │
│  │  input=64, hidden=96    │  │  input=64, hidden=96    │   │
│  └────────────┬────────────┘  └────────────┬────────────┘   │
│               │ (B,46,96)                  │ (B,46,96)      │
│               └──────────────┬─────────────┘                │
│                              │ cat → (B, 46, 192)           │
│                         Dropout(0.25)                        │
│                                                              │
│  Layer 2:                                                    │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │  RLSTMLayer (forward)   │  │  RLSTMLayer (backward)  │   │
│  │  input=192, hidden=96   │  │  input=192, hidden=96   │   │
│  └────────────┬────────────┘  └────────────┬────────────┘   │
│               │ (B,46,96)                  │ (B,46,96)      │
│               └──────────────┬─────────────┘                │
│                              │ cat → (B, 46, 192)           │
└──────────────────────────────┼───────────────────────────────┘
                               │ (B, 46, 192)
                               ▼
┌──────────────────────────────────────┐
│          AttentionPooling             │
│                                      │
│  a_t = softmax(Linear(tanh(Linear(h_t))))  │
│  c   = Σ_t a_t * h_t                │
│  output: (B, 192)                    │
└──────────────────┬───────────────────┘
                   │ (B, 192)
                   ▼
┌──────────────────────────────────────┐
│           Classifier Head            │
│                                      │
│  Linear(192 → 96) + ReLU            │
│  Dropout(0.25)                       │
│  Linear(96 → 5)                     │
└──────────────────┬───────────────────┘
                   │
                   ▼
            Logits: (B, 5)
```

### 4.2 RLSTMCell — The Core Innovation

The **Residual Memory Control (RMC) Cell** augments the standard LSTM with a learnable hybrid path.

```
Input at step t: x_t ∈ R^d,  h_{t-1} ∈ R^H,  c_{t-1} ∈ R^H
                                      │
               ┌──────────────────────▼──────────────────────────┐
               │              STANDARD LSTM GATES                 │
               │                                                  │
               │  [i_t, f_t, o_t, g_t] = W_x · x_t + W_h · h_{t-1}
               │  i_t = σ(·),  f_t = σ(·),  o_t = σ(·),  g_t = tanh(·)
               │                                                  │
               │  c_lstm = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t          │
               │                                                  │
               └─────────────────────┬────────────────────────────┘
                                     │ c_lstm
                                     │
               ┌─────────────────────▼────────────────────────────┐
               │               RMC PATH                           │
               │                                                  │
               │  m_t = LayerNorm(W_c · c_{t-1} + W_h_rmc · h_{t-1})
               │  r_t = σ(m_t)            ← residual gate        │
               │                                                  │
               │  c_keep = r_t   ⊙ c_{t-1}   (preserve old)     │
               │  c_add  = (1-r_t) ⊙ c_lstm  (admit new)        │
               │                                                  │
               │  α = softmax(W_alpha · [c_keep; c_add])         │
               │  c_rmc = α[0] * c_keep + α[1] * c_add          │
               │                                                  │
               └─────────────────────┬────────────────────────────┘
                                     │ c_rmc
               ┌─────────────────────▼────────────────────────────┐
               │             BLEND GATE (β)                        │
               │                                                  │
               │  β = σ(W_beta · c_{t-1})   [bias init: -5.0]   │
               │                                                  │
               │  c_t = β ⊙ c_rmc  +  (1-β) ⊙ c_lstm           │
               │                                                  │
               │  h_t = o_t ⊙ tanh(c_t)                         │
               └──────────────────────────────────────────────────┘

Interpretable outputs preserved per step:
  r_t     ← residual gate trajectory  (how much old memory kept)
  c_keep  ← preserved component
  c_add   ← new information component
  α       ← soft selection weights
  β       ← RMC vs LSTM blend ratio
```

**Key design choice:** `W_beta` bias initialized to **-5.0** → `σ(-5) ≈ 0.007`. At the start of training the model is nearly pure LSTM, and the RMC path is activated only as the model learns it is beneficial. This provides a **stable training curriculum**.

### 4.3 Loss Function Stack

```
RLSTMLoss
├── Task Loss: FocalLoss(γ=1.5, α=class_weights)
│       FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
│       Purpose: down-weight easy majority-class samples
│
└── Smooth Loss: temporal_smoothness_loss
        L_smooth = mean(‖r_{t+1} - r_t‖²)
        λ_smooth = 0.003
        Purpose: prevent erratic gate switching

Total: L = L_task + λ_smooth * L_smooth
```

### 4.4 Ablation Variants

| Variant | Change from Full Model | Purpose |
|---------|----------------------|---------|
| `full` | Reference model | Baseline |
| `no_rmc` | `c_t = c_lstm` (RMC path disabled) | Measures RMC contribution |
| `no_cnn` | Raw (B,187,1) input to BiRLSTM | Measures CNN contribution |
| `mean_pool` | Global mean instead of attention | Measures attention contribution |
| `no_smooth` | λ_smooth = 0 | Measures smoothness regularization |
| `no_adv` | No FGSM training | Measures adversarial training contribution |
| `no_hybrid` | β = 1 (RMC only, no LSTM fallback) | Measures hybrid blend contribution |

---

## 5. Training Pipeline

**Files:** `train.py`, `run_baselines.py`, `run_ablation.py`, `train_ensemble.py`, `train_inter_patient.py`

### 5.1 Main Training Flow

```
Load data/processed/*.npz
       │
       ▼
ECGDataset → DataLoader(batch_size=128, shuffle=True)
       │
       ▼
Initialize RLSTMClassifier
  hidden=96, layers=2, dropout=0.25, num_classes=5
       │
       ▼
Optimizer: Adam(lr=1e-3)
Scheduler: CosineAnnealingLR(T_max=45, eta_min=1e-5)
       │
       ▼
┌─────────────────────── Per Epoch ───────────────────────────┐
│                                                              │
│  For each batch (x, y):                                      │
│    ┌─────────────────────────────┐                          │
│    │ 70% of batch: clean x      │                          │
│    │ 30% of batch: FGSM x_adv  │                          │
│    │   x_adv = x + ε·sign(∇_x L)                          │
│    │   ε = 0.02                                            │
│    └─────────────────────────────┘                          │
│    Forward: logits = model(concat([x_clean, x_adv]))        │
│    Loss = FocalLoss + λ·smooth_loss                         │
│    Backward + clip_grad_norm(1.0)                           │
│    Adam step                                                │
│                                                              │
│  Validate → compute macro-F1                                │
│  If F1 improved: save checkpoint                            │
│  Early stop after patience=8 epochs of no improvement      │
│  Scheduler step                                             │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Load best checkpoint → evaluate on test set
Save: results/checkpoints/best_rlstm.pt
      results/logs/training_history.json
```

### 5.2 Baseline Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| Logistic Regression | Flatten 187→LR | Statistical baseline |
| Decision Tree | max_depth=20 | Interpretable baseline |
| LSTM | hidden=96, 1-layer | Ablation: directionality |
| BiLSTM | hidden=96, 1-layer | Ablation: RMC contribution |
| ResNet1D | 3 residual blocks | Strong temporal CNN baseline |

---

## 6. Evaluation & Trustworthiness Framework

```
┌────────────────────────────────────────────────────────────────┐
│              TRUSTWORTHINESS EVALUATION FRAMEWORK              │
│                                                                │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│  │  ROBUSTNESS │   │ CALIBRATION  │   │  EXPLAINABILITY     │ │
│  │             │   │              │   │                     │ │
│  │ evaluate_   │   │ evaluate_    │   │ shap_analysis.py    │ │
│  │ fgsm.py     │   │ calibration  │   │ integrated_         │ │
│  │ evaluate_   │   │ .py          │   │ gradients.py        │ │
│  │ pgd.py      │   │              │   │ data_attribution.py │ │
│  │ evaluate_   │   │              │   │                     │ │
│  │ autoattack  │   │              │   │                     │ │
│  │ .py         │   │              │   │                     │ │
│  └──────┬──────┘   └──────┬───────┘   └──────────┬──────────┘ │
│         │                 │                       │            │
│  ┌──────▼─────────────────▼───────────────────────▼──────────┐ │
│  │              UNCERTAINTY QUANTIFICATION                    │ │
│  │         mc_dropout.py  /  deep_ensemble.py                │ │
│  │              evaluate_corruptions.py                       │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────▼─────────────────────────────────┐ │
│  │          evaluate_trustworthiness.py (dashboard)           │ │
│  │    Aggregates ALL metrics into a single CSV report         │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. Explainability Subsystem

**Directory:** `explainability/`

### 7.1 SHAP Analysis (`shap_analysis.py`)

- **Method:** `shap.GradientExplainer` (chosen over DeepExplainer for LayerNorm compatibility)
- **Background:** 200 random training samples
- **Analysis scope:** 10 correct + 5 misclassified samples per class
- **Output:** Per-timestep importance scores, class-level summary plots, overlaid on ECG waveforms

### 7.2 Integrated Gradients (`integrated_gradients.py`)

- **Method:** Captum `IntegratedGradients` with 50 interpolation steps
- **Anatomical mapping:** P-wave (10-40ms), QRS complex (60-100ms), T-wave (110-155ms)
- **Output:** Attribution heatmaps aligned to ECG morphology

### 7.3 Data Attribution (`data_attribution.py`)

- **Method:** TracIn (gradient cosine similarity as influence proxy)
- **Two-signal filter:** Training sample flagged as "label noise candidate" only if:
  1. Influence score > threshold AND
  2. Model-predicted label ≠ true label with confidence > 0.9
- **Output:** Top-30 confusable pairs as waveform plots

### 7.4 Cross-Method Consistency

`plot_disagreements.py` computes the **Jaccard similarity** between SHAP and IG importance rankings, quantifying how consistently the two methods agree on which timesteps matter.

---

## 8. Uncertainty Quantification Subsystem

**Directory:** `uncertainty/`

### 8.1 MC Dropout (`mc_dropout.py`)

```
enable_mc_dropout():
  model.eval()               ← freeze BatchNorm running stats
  for m in model.modules():
    if isinstance(m, Dropout): m.train()  ← keep dropout stochastic

T forward passes per sample (T=50):
  p_t = softmax(model(x))   for t in 1..T

Metrics:
  p̄ = mean(p_t over T)
  Predictive Entropy H = -Σ_c p̄_c log(p̄_c)
  Mutual Information = H(p̄) - mean(H(p_t))   ← epistemic uncertainty
```

**OOD evaluation:** Four synthetic corruptions applied to test data, then entropy AUROC (ID=low entropy, OOD=high entropy) measures detection ability.

### 8.2 Deep Ensembles (`deep_ensemble.py`)

- Trains 3 independent models from seeds [42, 123, 456]
- Ensemble prediction = mean of all member softmax outputs
- Inter-member variance = epistemic uncertainty proxy

### 8.3 Corruption Sweep (`evaluate_corruptions.py`)

Tests model under: Gaussian noise (σ ∈ [0.05, 1.0]), baseline wander, crop-pad time shift, amplitude shift.

---

## 9. Calibration Subsystem

**Directory:** `calibration/`

### 9.1 Metrics

| Metric | Formula | Ideal |
|--------|---------|-------|
| ECE (15 bins) | Σ_b (n_b/N) ‖acc_b - conf_b‖ | 0.0 |
| MCE | max_b ‖acc_b - conf_b‖ | 0.0 |
| Brier Score | mean(‖p - y_one_hot‖²) | 0.0 |
| NLL | -Σ log p_{true_class} | 0.0 |

### 9.2 Temperature Scaling

```
T* = argmin_T  NLL(val; logits / T)
Solved via LBFGS (50 iterations)

Calibrated output: p_calib = softmax(logits / T*)
T* > 1.0 → model was overconfident before scaling
```

---

## 10. Robustness Subsystem

**Files:** `evaluate_fgsm.py`, `evaluate_pgd.py`, `evaluate_autoattack.py`, `robustness/cw_attack.py`

### 10.1 Attack Taxonomy

```
┌─────────────────────────────────────────────────────────────┐
│                     ATTACKS EVALUATED                       │
│                                                             │
│  White-box (gradient-based):                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FGSM   x_adv = x + ε·sign(∇_x L)  (1 step)       │   │
│  │  PGD    iterated FGSM + random start (20 steps)     │   │
│  │  CW     minimize ‖δ‖ + c·f(x+δ) (optimization)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Black-box / adaptive:                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  AutoAttack (ensemble of 4 attacks):                │   │
│  │    APGD-CE:  adaptive step, CE loss                 │   │
│  │    APGD-DLR: adaptive step, DLR margin loss         │   │
│  │    FAB:      decision-boundary based                │   │
│  │    Square:   score-based (black-box)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Evaluated at ε ∈ {0.01, 0.02, 0.05}                      │
│  Metrics: Clean Acc, Adv Acc, ASR, per-class recall         │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Results & Visualization Pipeline

**Key files:** `report_results.py`, `generate_results_tables.py`, `plot_and_export.py`, `evaluate_trustworthiness.py`

### 11.1 Figures Generated

| Figure | File | Method |
|--------|------|--------|
| 5×5 Confusion Matrix (normalized) | `confusion_matrix.png` | `report_results.py` |
| Multi-class ROC curves | `roc_curves.png` | One-vs-rest AUC |
| Gate trajectory visualization | `gate_trajectories.png` | r_t over time per class |
| Reliability diagrams (before/after temp. scaling) | `reliability_*.png` | `calibration_metrics.py` |
| MC Dropout entropy histograms | `mc_entropy_distribution.png` | `mc_dropout.py` |
| SHAP summary plot | `shap_summary_plot.png` | GradientExplainer |
| IG attribution overlays | `ig_{class}.png` | Captum |
| Baseline comparison bar chart | `fgsm_baseline_comparison.png` | `evaluate_fgsm.py` |

### 11.2 Tables Generated

| Table | Format | Content |
|-------|--------|---------|
| `baseline_comparison.csv/.tex` | CSV + LaTeX | Accuracy/Precision/Recall/F1/AUC |
| `ablation_table_final.csv/.tex` | CSV + LaTeX | Clean + robustness per ablation variant |
| `calibration_results.csv` | CSV | ECE/Brier/NLL before and after T-scaling |
| `fgsm_baseline_summary.csv` | CSV | Per-epsilon ASR for all models |
| `trustworthiness_dashboard.csv` | CSV | All metrics, one row per model |

---

## 12. Orchestration & Reproducibility

**Files:** `run_reproducible_pipeline.py`, `run_all.bat`, `configs/`

### 12.1 Pipeline Execution Order

```
run_reproducible_pipeline.py
  Step 0: preprocess.py
  Step 1: run_baselines.py
  Step 2: train.py
  Step 3: run_ablation.py
  Step 4: report_results.py       (core figures)
  Step 5: evaluate_fgsm.py
  Step 6: evaluate_pgd.py
  Step 7: evaluate_ablation_robustness.py
  Step 8: evaluate_robustness_all.py
  Step 9: evaluate_calibration.py
  Step 10: combine_ablation_tables.py
  Step 11: generate_results_tables.py
  Step 12: plot_and_export.py
  Step 13: evaluate_trustworthiness.py
```

### 12.2 Run ID & Versioning

```
configs/paths.py:
  run_id = f"v1.0_{datetime.now():%Y%m%d_%H%M%S}"
  outputs/{run_id}/...
```

All evaluation outputs are namespaced under a timestamped run ID, enabling comparison across multiple runs.

### 12.3 Seed Control

```python
set_seed(42):
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
```

---

## 13. Directory & Module Map

```
D:\HMR-BiLSTM-main/
│
├── Core Model
│   ├── hmr_bilstm.py          ECGFeatureExtractor, RLSTMCell, BiRLSTM,
│   │                          RLSTMClassifier, FocalLoss, RLSTMLoss
│   └── hmr_bilstm_ablation.py Ablation-aware version of above
│
├── Data
│   └── preprocess.py          MIT-BIH loading, normalization, splitting
│
├── Training
│   ├── train.py               Main HMR-BiLSTM training loop
│   ├── run_baselines.py       LR, DT, LSTM, BiLSTM, ResNet1D baselines
│   ├── run_ablation.py        7-variant ablation sweep
│   ├── train_ensemble.py      Multi-seed ensemble training
│   ├── train_inter_patient.py Inter-patient generalization split
│   └── run_ablation_inter.py  Ablation on inter-patient split
│
├── Evaluation
│   ├── evaluate_fgsm.py
│   ├── evaluate_pgd.py
│   ├── evaluate_autoattack.py
│   ├── evaluate_calibration.py
│   ├── evaluate_robustness_all.py
│   ├── evaluate_ablation_robustness.py
│   └── evaluate_trustworthiness.py
│
├── explainability/
│   ├── shap_analysis.py
│   ├── integrated_gradients.py
│   ├── data_attribution.py
│   └── plot_disagreements.py
│
├── uncertainty/
│   ├── mc_dropout.py
│   ├── deep_ensemble.py
│   └── evaluate_corruptions.py
│
├── calibration/
│   ├── calibration_metrics.py
│   ├── reliability_diagram.py
│   └── temperature_scaling.py
│
├── robustness/
│   ├── auto_attack.py
│   └── cw_attack.py
│
├── validation/
│   ├── download_mitdb.py
│   ├── evaluate_splits.py
│   ├── pgd_convergence.py
│   ├── preprocess_aami.py
│   └── verify_normalization.py
│
├── configs/
│   ├── experiment_config.yaml
│   └── paths.py
│
├── Results
│   ├── report_results.py
│   ├── generate_results_tables.py
│   ├── plot_and_export.py
│   ├── combine_ablation_tables.py
│   └── gather_final.py
│
├── Orchestration
│   ├── run_reproducible_pipeline.py
│   └── run_all.bat
│
├── Diagnostics
│   ├── verify_checkpoint.py
│   ├── verify_gradients.py
│   ├── test_shape.py
│   ├── test_speed.py
│   └── diag_*.py
│
└── Docs
    ├── README.md
    ├── requirements.txt
    └── HMR-BiLSTM_Installation_Guide.docx
```

---

## 14. Hyperparameter Reference

| Parameter | Value | Location | Rationale |
|-----------|-------|----------|-----------|
| `hidden_size` | 96 | `train.py` | Balance capacity vs. compute |
| `num_layers` | 2 | `train.py` | Hierarchical temporal patterns |
| `dropout` | 0.25 | `train.py` | Regularization without over-damping |
| `batch_size` | 128 | `train.py` | Stable gradients on imbalanced data |
| `learning_rate` | 1e-3 | `train.py` | Standard Adam starting point |
| `min_lr` | 1e-5 | `train.py` | Cosine annealing floor |
| `epochs` | 45 | `train.py` | Sufficient convergence for dataset size |
| `early_stop_patience` | 8 | `train.py` | Prevent overfitting |
| `lambda_smooth` | 0.003 | `train.py` | Gentle gate regularization |
| `focal_gamma` | 1.5 | `train.py` | Focus on hard minority-class samples |
| `adv_epsilon` | 0.02 | `train.py` | L∞ perturbation budget |
| `adv_ratio` | 0.3 | `train.py` | 30% adversarial batch mix |
| `W_beta_bias_init` | -5.0 | `hmr_bilstm.py` | Start near pure LSTM (σ(-5)≈0.007) |
| `class_weight_clip` | [0.5, 10.0] | `preprocess.py` | Prevent loss explosion on F class |
| `mc_samples` | 50 | `experiment_config.yaml` | MC Dropout Monte Carlo samples |
| `ensemble_size` | 3 | `experiment_config.yaml` | Ensemble members |
| `shap_background` | 200 | `experiment_config.yaml` | GradientExplainer background set |
| `pgd_steps` | 20 | `evaluate_pgd.py` | Sufficient attack convergence |
| `temperature_bins` | 15 | `calibration_metrics.py` | ECE binning resolution |

---

## 15. Strengths

### 15.1 Architectural

**RMC Cell design is principled and interpretable.**  
The blend gate β with bias initialized to -5.0 creates a natural training curriculum: the model starts as pure LSTM and gradually learns when the RMC path is beneficial. This avoids the instability of initializing a novel path at full strength. The residual gate r_t is directly interpretable as "how much old memory the model retains" and can be plotted per class per timestep — a genuinely useful diagnostic for cardiologists.

**CNN + BiRLSTM hierarchy matches ECG structure.**  
The two-stage architecture (morphological features via CNN → temporal context via BiRLSTM) mirrors the clinical ECG interpretation process: first identify beat shapes, then contextualize within rhythm. This inductive bias is appropriate for the domain.

**Attention pooling adds temporal localization.**  
Sequence pooling via learned attention weights is superior to global mean/max pooling and provides a secondary interpretability channel (which timesteps the model focuses on).

### 15.2 Training

**FGSM adversarial training at train time is lightweight and effective.**  
30% batch ratio with ε=0.02 provides robustness without destabilizing training. The implementation correctly keeps the model in `train()` mode during FGSM generation to preserve BatchNorm statistics.

**Focal Loss + class weights is a well-motivated combination.**  
For a dataset with ~100:1 class imbalance, using both per-class weights (inverse frequency) AND focal down-weighting of easy samples is appropriate. The weight clipping [0.5, 10.0] prevents numerical instability.

**Cosine annealing with early stopping is robust.**  
This combination avoids the need for manual learning rate tuning while still terminating training before overfitting.

### 15.3 Evaluation

**The trustworthiness framework is genuinely comprehensive.**  
Most published ECG classification papers evaluate only accuracy and F1. This framework additionally provides: 4 attack types × 3 ε values, 3 calibration metrics with temperature scaling, 2 uncertainty methods with OOD detection, and 3 complementary explainability methods. This is publication-grade.

**MC Dropout implementation is correct.**  
`enable_mc_dropout()` correctly sets `model.eval()` first (to freeze BatchNorm), then selectively re-enables only Dropout layers. Many public implementations incorrectly call `model.train()`, which introduces spurious randomness from BatchNorm.

**AutoAttack inclusion detects gradient masking.**  
By combining white-box (APGD) with black-box (Square) attacks, the framework can detect whether apparent robustness is due to genuine generalization or gradient masking — a critical validity check.

**TracIn two-signal filter reduces false positives.**  
Requiring both high influence AND high-confidence disagreement before labeling a sample as "label noise" is a sound filtering strategy that prevents noisy training examples from overwhelming the analysis.

### 15.4 Software Engineering

- Strict seed control (numpy, torch, CUDA) with deterministic cuDNN
- Versioned run IDs prevent output clobbering across experiments
- Independent validation scripts (`validation/`) can be run separately from the main pipeline
- Consistent use of `.npz` for intermediate data prevents silent dtype mismatches
- 14-step reproducible pipeline in both Python and Windows batch form

---

## 16. Weaknesses & Limitations

### 16.1 Architectural

**RMC path adds parameters with uncertain added value.**  
The RMC cell adds W_c, W_h_rmc, W_alpha, W_beta (~5 × H² extra parameters per layer). Whether these yield statistically significant improvement over a well-tuned standard BiLSTM should be validated rigorously — the ablation study (`no_rmc` variant) answers this, but the margin should be checked for statistical significance across multiple seeds.

**CNN downsampling may lose P-wave detail.**  
Two rounds of MaxPool(2) reduce 187 → 46 timesteps. At 360 Hz, this loses temporal resolution to ~31 Hz, which may be insufficient to resolve fine-grained P-wave morphology that distinguishes S (APC) from N beats. Adaptive average pooling or strided convolutions would be more principled.

**Input length is fixed at 187 samples.**  
The architecture cannot handle variable-length beats without padding/truncation. Real-world ECG signals have variable QRS-to-QRS intervals, so any deployment beyond MIT-BIH preprocessing would need adaptation.

**BatchNorm in the CNN interacts with small batch sizes.**  
During evaluation with MC Dropout, BatchNorm is correctly frozen. However, the implementation does not use `num_features` sync across BatchNorm layers — if batch size < 4, BN statistics may be unreliable during training on rare classes.

### 16.2 Training

**Inter-patient generalization is an add-on, not the primary split.**  
The main results use the standard MIT-BIH intra-patient split, where train and test patients overlap. Several papers (Chazal et al., 2004; De Chazal & Reilly, 2006) have shown that intra-patient accuracy is significantly higher than inter-patient accuracy. The `train_inter_patient.py` script addresses this, but it should be the primary reported result for clinical relevance.

**Adversarial training (FGSM) alone does not provide certified robustness.**  
FGSM-trained models are often vulnerable to stronger iterative attacks (PGD, AutoAttack), which the evaluation correctly demonstrates. However, the training does not use PGD-based adversarial training, which would provide stronger guarantees.

**No data augmentation beyond adversarial perturbations.**  
Standard ECG augmentation (time warping, amplitude scaling, electrode noise simulation) is not applied during training. This may limit generalization to new ECG recording devices or patient populations.

**The Deep Ensemble in "diversity mode" (single checkpoint) is not a real ensemble.**  
`deep_ensemble.py` gracefully degrades to single-model evaluation if only one checkpoint is present. The uncertainty estimates in this mode are not ensemble-based and should be clearly distinguished from true ensemble uncertainty.

### 16.3 Evaluation

**SHAP GradientExplainer is an approximation.**  
`GradientExplainer` computes expected gradients, which approximates Shapley values but is not exact. For time-series with temporal dependencies, the SHAP values for individual timesteps are not independent and can be misleading if interpreted as isolated contributions.

**TracIn influence approximation using gradient cosine similarity is heuristic.**  
The implemented TracIn variant uses cosine similarity of full-model gradients rather than the sum-over-checkpoints formulation from the original paper. This is a common approximation but may miss training dynamics.

**Calibration evaluated only on test set (not separate calibration set).**  
Temperature scaling is fitted on the validation set, which is correct. However, the reliability of ECE estimates with 15 bins depends on having sufficient samples in each bin for each class — the F class (rare) may have sparse bins.

**AutoAttack requires `autoattack` package that is not in `requirements.txt`.**  
The `autoattack` library, `captum` (for IG), and `shap` are missing from `requirements.txt`, creating reproducibility issues for others.

### 16.4 Dataset & Scope

**MIT-BIH is a small, well-curated benchmark.**  
109 recording segments from 47 patients is a limited clinical dataset. Performance on MIT-BIH does not necessarily translate to real-world ECG databases (PTB-XL, PhysioNet, hospital EHR data).

**5-class AAMI mapping conflates clinically distinct rhythms.**  
Class N includes both normal sinus rhythm and left/right bundle branch block beats. Class Q includes paced beats. This grouping may mask subclass-level performance differences.

**No external validation set.**  
All models are trained and tested on MIT-BIH only. A held-out external dataset (e.g., INCART, E-HOL) would strengthen clinical validity claims.

---

## 17. Publication Recommendations

### 17.1 Target Venues

| Tier | Venue | Rationale |
|------|-------|-----------|
| Top | IEEE Transactions on Biomedical Engineering (TBME) | Strong precedent for ECG DL; impact factor ~4.5 |
| Top | IEEE Journal of Biomedical and Health Informatics (JBHI) | Explicitly covers clinical AI trustworthiness |
| High | Pattern Recognition | Novel architecture + ablation + comprehensive eval |
| High | Computers in Biology and Medicine | Applied clinical AI audience |
| Conference | EMBC 2026 / CINC 2026 | Workshop-level fast turnaround for cardiac AI |

### 17.2 Paper Framing

The paper's primary contribution should be framed as a **trustworthiness framework** with the HMR-BiLSTM model as the vehicle — not merely a new LSTM variant. The differentiation from prior work is the *combination* of:
1. Interpretable memory control (r_t gate trajectory)
2. Adversarial robustness evaluation with 4 attack types
3. Calibration analysis with temperature scaling
4. Uncertainty quantification with MC Dropout and Deep Ensembles
5. Three complementary explainability methods with cross-method consistency checks

This framing aligns with the growing "trustworthy AI in healthcare" literature (Obermeyer et al., 2019; Topol, 2019; Rajpurkar et al., 2022).

### 17.3 Critical Results to Lead With

1. **Macro-F1 on test set** (all 5 classes including minority F and S) — emphasize this over accuracy due to class imbalance.
2. **ASR gap between HMR-BiLSTM and BiLSTM baseline** at ε=0.02 — quantifies the adversarial robustness benefit of FGSM training.
3. **ECE before and after temperature scaling** — demonstrates calibration analysis maturity.
4. **AUROC (ID vs OOD)** from MC Dropout — clinical deployment readiness indicator.
5. **Gate trajectory visualization** — the most visually compelling result for reviewers; shows when the RMC path activates during P-wave, QRS, and T-wave segments.

### 17.4 Required Additions Before Submission

**High Priority**

1. **Statistical significance testing.** Run each model variant from ≥5 independent seeds. Report mean ± std for all metrics. Use paired t-test or Wilcoxon signed-rank test for ablation comparisons. Without this, reviewers will correctly challenge whether ablation differences are meaningful.

2. **Inter-patient results as primary evaluation.** Shift `train_inter_patient.py` results to Table 1 (primary results) and move intra-patient to Table 2 (comparison with prior work). This addresses the strongest methodological critique.

3. **Fix `requirements.txt`** to include `shap`, `captum`, `autoattack` with pinned versions. Add a `setup.py` or `pyproject.toml`. Without this, independent replication is not possible.

4. **External validation.** Test the trained model (without retraining) on at least one external ECG dataset. PTB-XL or E-HOL are publicly available. Even qualitative analysis strengthens the claim that HMR-BiLSTM generalizes.

**Medium Priority**

5. **Ablation on inter-patient split.** The current ablation uses intra-patient splits. Repeat the ablation on the inter-patient split to validate that each component's contribution holds under distribution shift.

6. **Ensemble uncertainty as true ensemble.** Train 3 models from seeds [42, 123, 456] and report true ensemble uncertainty. The current single-checkpoint "diversity mode" should not appear in the paper.

7. **Per-class calibration reliability diagrams.** The overall reliability diagram may hide poor calibration on the F class (only ~800 samples). Show per-class reliability curves.

8. **Complexity analysis.** Report FLOPs, parameter count, and inference time vs. baselines. Clinical deployment feasibility depends on these.

**Lower Priority (strengthens paper but not blocking)**

9. **Compare SHAP vs. IG attribution on anatomically labeled beats.** Use cardiologist-labeled beats where the discriminating feature is known (e.g., P-wave absence in APC) to validate whether attributions agree with clinical ground truth.

10. **Ablation on CNN kernel size and pooling strategy.** The choice of k=5, k=3, MaxPool(2) × 2 is not ablated. A single experiment varying this would validate the design.

11. **Visualization of β gate values.** Show how the blend gate β evolves over training (does it increase, indicating growing reliance on RMC?). This would validate the training curriculum intuition behind the W_beta bias initialization.

### 17.5 Writing Recommendations

- **Section order:** Related Work → Problem Formulation → HMR-BiLSTM Architecture → Trustworthiness Framework → Experiments → Results → Discussion → Conclusion
- **Figure 1:** Full architecture diagram (Section 4.1 format above) — must appear in paper
- **Figure 2:** RMC Cell detailed diagram (Section 4.2 format) — core contribution visualization
- **Figure 3:** Gate trajectory r_t per class — most compelling interpretability result
- **Table 1:** Inter-patient results vs. baselines (accuracy, macro-F1, AUC per class)
- **Table 2:** Ablation results (clean + adversarial robustness)
- **Table 3:** Trustworthiness metrics (ECE, Brier, MC-AUROC, Ensemble-var)
- Cite the adversarial training literature (Madry et al., 2018; Goodfellow et al., 2015), calibration work (Guo et al., 2017; Platt, 1999), and clinical ECG DL surveys (Hannun et al., 2019; Ribeiro et al., 2020).
- Acknowledge the intra-patient vs. inter-patient limitation explicitly in the Discussion — reviewers familiar with ECG literature will look for it.

---

*Report generated from full repository scan of D:\HMR-BiLSTM-main — 2026-06-23*
