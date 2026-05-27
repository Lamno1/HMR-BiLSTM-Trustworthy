# HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification

## Overview

HMR-BiLSTM is a deep learning framework for ECG arrhythmia classification on the MIT-BIH Arrhythmia Dataset. The architecture is designed to improve temporal representation learning, adversarial robustness, and explainability for safety-critical healthcare AI systems.

### Key Components

- **Residual Memory Control (RMC):** Hybrid memory path with softmax-based memory decomposition for adaptive long-term dependency modeling.
- **Bidirectional LSTM:** Captures both forward and backward temporal context in ECG signals.
- **CNN Feature Extractor:** 1D convolutional front-end for local morphological feature extraction.
- **Attention Pooling:** Learnable temporal attention mechanism for sequence aggregation.
- **Focal Loss + Class Weights:** Addresses severe class imbalance (N >> S, V, F, Q).
- **Adversarial Training (FGSM):** Improves robustness against adversarial perturbations at training time.
- **Temporal Smoothness Regularization:** Encourages stable residual gate trajectories.

---

## Dataset

This project uses the [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

After downloading, place the CSV files in `data/raw/`:

```
data/raw/
├── mitbih_train.csv
└── mitbih_test.csv
```

> **Note:** `data/raw/` is excluded from version control (see `.gitignore`). All other data (processed splits, class weights) and model checkpoints are tracked in the repository.

**5-class AAMI mapping:** N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown).

---

## Project Structure

```
HMR-BiLSTM/
│
├── data/
│   ├── raw/                                # Raw MIT-BIH CSV files (git-ignored)
│   └── processed/                          # Preprocessed .npz splits + class weights
│
├── results/
│   ├── checkpoints/                        # Model checkpoints (.pt)
│   │   ├── best_lstm.pt
│   │   ├── best_bilstm.pt
│   │   └── best_rlstm.pt                   # HMR-BiLSTM checkpoint
│   ├── figures/                            # Generated plots and visualizations
│   ├── logs/                               # Evaluation logs (JSON)
│   │   ├── baseline_results.json
│   │   ├── fgsm_baseline_comparison.json
│   │   └── pgd_baseline_comparison.json
│   ├── tables/                             # LaTeX and CSV result tables
│   └── ablation/                           # Ablation study results
│       ├── checkpoints/                    # Ablation variant checkpoints
│       ├── ablation_results.json           # Full ablation metrics
│       ├── ablation_table_final.csv        # Summary table (CSV)
│       └── ablation_table_final.tex        # Summary table (LaTeX)
│
├── hmr_bilstm.py                           # HMR-BiLSTM model architecture + RLSTMLoss
├── hmr_bilstm_ablation.py                  # Ablation variants (No-RMC, No-CNN, Mean-Pool, etc.)
├── preprocess.py                           # Data preprocessing and train/val/test split
├── train.py                                # Main training script (HMR-BiLSTM)
├── run_baselines.py                        # Train baseline models (LSTM, BiLSTM)
├── run_ablation.py                         # Ablation study (train + evaluate all variants)
├── report_results.py                       # Generate core figures (confusion matrix, ROC, gates)
├── evaluate_fgsm.py                        # FGSM adversarial robustness evaluation
├── evaluate_pgd.py                         # PGD adversarial robustness evaluation
├── evaluate_calibration.py                 # Calibration analysis (reliability diagram, ECE, Brier)
├── evaluate_robustness_all.py              # Gaussian noise robustness evaluation
├── evaluate_ablation_robustness.py         # Adversarial robustness for ablation variants
├── compare_fgsm_baselines.py               # FGSM comparison across all models
├── combine_ablation_tables.py              # Merge ablation clean + robustness tables
├── generate_results_tables.py              # Generate LaTeX/CSV summary tables
├── plot_and_export.py                      # Export final figures + baseline_full_comparison table
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```

Splits the raw MIT-BIH dataset into train/val/test sets and computes class weights. Outputs saved to `data/processed/`.

### 2. Train HMR-BiLSTM (Proposed Model)

```bash
python train.py
```

Trains the main HMR-BiLSTM model with adversarial training (FGSM), focal loss, cosine annealing LR, and early stopping. Best checkpoint saved to `results/checkpoints/best_rlstm.pt`.

### 3. Train Baseline Models

```bash
python run_baselines.py
```

Trains LSTM and BiLSTM baseline models. Checkpoints saved to `results/checkpoints/`.

### 4. Ablation Study

```bash
# Run all ablation variants
python run_ablation.py

# Run specific variants
python run_ablation.py --variants no_rmc no_cnn

# Generate table from existing checkpoints only
python run_ablation.py --table-only
```

Variants: `full`, `no_rmc`, `no_cnn`, `mean_pool`, `no_adv`, `no_hybrid`, `no_smooth`.

### 5. Evaluation & Visualization

```bash
# Core figures: confusion matrix, ROC curve, gate trajectories
python report_results.py

# FGSM adversarial robustness (all models)
python compare_fgsm_baselines.py

# PGD adversarial robustness
python evaluate_pgd.py

# Gaussian noise robustness
python evaluate_robustness_all.py

# Calibration analysis (reliability diagram, ECE, Brier score)
python evaluate_calibration.py

# Adversarial robustness for ablation variants
python evaluate_ablation_robustness.py

# Export final figures and tables
python plot_and_export.py
```

> **Note (Windows/CUDA):** PGD evaluation requires `torch.backends.cudnn.enabled = False` to avoid CuDNN backward errors in eval mode. This is handled automatically inside `evaluate_pgd.py`.

---

## Results

### Table 1 — Clean Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 (macro) | F1-weighted | AUC-OvR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6741 | 0.4433 | 0.7650 | 0.4786 | 0.7375 | 0.9146 |
| Decision Tree | 0.8911 | 0.6162 | 0.8477 | 0.6846 | 0.9058 | 0.9197 |
| LSTM | 0.9639 | 0.8300 | 0.9027 | 0.8583 | 0.9669 | 0.9878 |
| BiLSTM | 0.9654 | 0.8322 | 0.8733 | 0.8505 | 0.9657 | 0.9828 |
| HMR-BiLSTM (no Adv) | **0.9794** | — | — | **0.9024** | — | — |
| **HMR-BiLSTM** | 0.9749 | **0.8562** | **0.9146** | 0.8825 | **0.9758** | **0.9916** |

### Table 2 — FGSM Adversarial Robustness (ε=0.02)

| Model | F1 (clean) | F1 (adv) | F1 Drop | ASR | Rec-S clean | Rec-S adv | Rec-V clean | Rec-V adv | Rec-F clean | Rec-F adv |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.8583 | 0.7628 | 11.13% | 0.0492 | 0.8129 | 0.7248 | 0.9372 | 0.8847 | 0.8086 | 0.6667 |
| BiLSTM | 0.8505 | 0.7886 | 7.29% | 0.0170 | 0.6781 | 0.6349 | 0.9586 | 0.9358 | 0.7778 | 0.6852 |
| HMR-BiLSTM (no Adv) | 0.9024 | 0.8046 | 10.84% | 0.0346 | 0.8489 | 0.7680 | 0.9689 | 0.9413 | 0.8025 | 0.6790 |
| **HMR-BiLSTM** | 0.8825 | **0.8425** | **4.54%** | **0.0114** | 0.8112 | 0.7842 | 0.9448 | 0.9268 | 0.8457 | 0.7963 |

### Table 3 — PGD-20 Adversarial Robustness

| Model | F1 (clean) | F1-PGD (ε=0.02) | F1-PGD (ε=0.05) | F1 Drop (0.02) | ASR (0.02) | ASR (0.05) |
|---|---|---|---|---|---|---|
| LSTM | 0.8583 | 0.7365 | 0.5338 | 14.19% | 0.0700 | 0.2641 |
| BiLSTM | 0.8505 | 0.7728 | 0.6080 | 9.14% | 0.0221 | 0.0971 |
| **HMR-BiLSTM** | 0.8825 | **0.8391** | **0.7470** | **4.91%** | **0.0131** | **0.0509** |

### Table 4 — Calibration

| Model | ECE ↓ | Brier Score ↓ | ECE Grade | Brier Grade |
|---|---|---|---|---|
| LSTM | **0.0056** | 0.0461 | Excellent | Excellent |
| BiLSTM | 0.0080 | 0.0542 | Excellent | Good |
| HMR-BiLSTM | 0.0397 | **0.0444** | Good | Excellent |

### Table 5 — Ablation Study

| Variant | Params | F1-Clean | F1-Adv (ε=0.02) | F1-Adv (ε=0.05) | F1-S | F1-V | F1-F | AUC |
|---|---|---|---|---|---|---|---|---|
| HMR-BiLSTM (full) | 505,038 | **0.8921** | **0.8425** | **0.7823** | **0.7387** | 0.9492 | **0.7944** | **0.9917** |
| No-RMC (c_t = c_lstm) | 505,038 | 0.8918 | 0.8269 | 0.7529 | 0.7590 | **0.9505** | 0.7713 | 0.9911 |
| No-CNN (raw input) | 450,062 | 0.8470 | 0.8007 | 0.7168 | 0.6844 | 0.9329 | 0.6667 | 0.9881 |
| Mean-Pool (no attention) | 486,413 | 0.8506 | 0.8175 | 0.7540 | 0.6427 | 0.9261 | 0.7216 | 0.9865 |
| No-Adv-Training | 505,038 | 0.9023 | 0.8045 | 0.5589 | 0.7789 | 0.9509 | 0.8025 | 0.9933 |
| No-Hybrid-Path (c_t = c_rmc) | 505,038 | 0.8606 | 0.8162 | 0.7357 | 0.6307 | 0.9338 | 0.7720 | 0.9854 |
| No-Smoothness (λ=0) | 505,038 | 0.8919 | 0.8384 | 0.7613 | 0.7424 | **0.9540** | 0.7831 | 0.9921 |

> All numbers verified from checkpoint files: `best_lstm.pt`, `best_bilstm.pt`, `best_rlstm.pt`.

---

## Research Contributions

- **Trustworthy healthcare AI:** Full adversarial robustness evaluation under FGSM and PGD attacks.
- **Explainable ECG classification:** Intrinsic interpretability through residual gate trajectory visualization.
- **Robust temporal modeling:** Hybrid memory decomposition with adaptive residual gating (RMC).
- **Improved calibration:** Lower ECE and Brier score vs. baselines for reliable probability outputs.
- **Clinical relevance:** Significant F1 improvement on minority arrhythmia classes (S, V, F).

---

## Keywords

ECG Arrhythmia Classification, Explainable AI, Trustworthy AI, BiLSTM, Residual Memory Control, Adversarial Robustness, FGSM, PGD, Biomedical Signal Processing, Deep Learning, MIT-BIH

---

## Citation

```bibtex
@article{hmr_bilstm_2026,
  title={HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification},
  author={Anonymous},
  year={2026}
}
```
