# HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification

## Overview

HMR-BiLSTM is a deep learning framework for ECG arrhythmia classification on the MIT-BIH Arrhythmia Dataset. The architecture is designed to improve temporal representation learning, adversarial robustness, and explainability for safety-critical healthcare AI systems.

### Key Components

- **Residual Memory Control (RMC):** Hybrid memory path with softmax-based memory decomposition for adaptive long-term dependency modeling.
- **Bidirectional LSTM:** Captures both forward and backward temporal context in ECG signals.
- **CNN Feature Extractor:** 1D convolutional front‑end for local morphological feature extraction.
- **Attention Pooling:** Learnable temporal attention mechanism for sequence aggregation.
- **Focal Loss + Class Weights:** Addresses severe class imbalance (N >> S, V, F, Q).
- **Adversarial Training (FGSM):** Improves robustness against adversarial perturbations at training time.
- **Temporal Smoothness Regularization:** Encourages stable residual gate trajectories.

---

## Dataset

The project uses the [MIT‑BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat). After downloading, place the CSV files in `data/raw/`:

```
data/raw/
├── mitbih_train.csv
└── mitbih_test.csv
```

> **Note:** `data/raw/` is ignored by version control (see `.gitignore`). All processed data, class weights, and model checkpoints are stored within the repository under `data/processed/` and `results/`.

---

## Project Structure

```
HMR-BiLSTM/
│
├── data/
│   ├── raw/                # Raw MIT‑BIH CSV files (git‑ignored)
│   └── processed/          # Pre‑processed splits + class weights
│
├── results/
│   ├── checkpoints/        # Model checkpoints (.pt)
│   ├── figures/            # Generated plots and visualizations
│   ├── tables/             # LaTeX and CSV result tables (final)
│   │   ├── ablation_table_final.csv
│   │   └── ablation_table_final.tex
│   └── logs/               # Evaluation logs (JSON)
│
├── hmr_bilstm.py                       # Model architecture + RLSTMLoss
├── hmr_bilstm_ablation.py              # Ablation variants
├── preprocess.py                       # Data preprocessing and split
├── train.py                            # Main training script
├── run_baselines.py                    # Train baseline models
├── run_ablation.py                     # Ablation study driver
├── report_results.py                   # Core figures (confusion matrix, ROC)
├── evaluate_fgsm.py                    # FGSM robustness evaluation
├── evaluate_pgd.py                     # PGD robustness evaluation
├── evaluate_calibration.py             # Calibration analysis
├── evaluate_robustness_all.py          # Gaussian noise robustness
├── evaluate_ablation_robustness.py     # Ablation robustness evaluation
├── combine_ablation_tables.py          # Merge clean + robustness tables
├── generate_results_tables.py          # Generate LaTeX/CSV summary tables
├── plot_and_export.py                  # Export final figures & tables
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Lamno1/HMR-BiLSTM-Trustworthy.git
cd HMR-BiLSTM-Trustworthy

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

### Option A: Run the Complete Reproducible Pipeline (Recommended)

You can reproduce the entire pipeline (from preprocessing, baseline training, ablation study to exporting all final tables and figures) in a single command using the master script:

```bash
python run_reproducible_pipeline.py
```

To view the commands that will be executed without actually running them:
```bash
python run_reproducible_pipeline.py --dry-run
```

### Option B: Run Steps Manually

If you prefer to run the components individually:

#### 1. Data Preprocessing
```bash
python preprocess.py
```
Splits the raw MIT‑BIH dataset into train/val/test sets and computes class weights. Outputs saved to `data/processed/`.

#### 2. Train HMR‑BiLSTM (Proposed Model)
```bash
python train.py
```
Trains the main HMR‑BiLSTM model with adversarial training (FGSM), focal loss, cosine annealing LR, and early stopping. Best checkpoint saved to `results/checkpoints/best_rlstm.pt`.

#### 3. Train Baseline Models
```bash
python run_baselines.py
```
Trains LSTM and BiLSTM baseline models. Checkpoints saved to `results/checkpoints/`.

#### 4. Ablation Study
```bash
# Run all ablation variants
python run_ablation.py

# Run specific variants
python run_ablation.py --variants no_rmc no_cnn

# Generate table from existing checkpoints only
python run_ablation.py --table-only
```
Variants: `full`, `no_rmc`, `no_cnn`, `mean_pool`, `no_adv`, `no_hybrid`, `no_smooth`.

#### 5. Evaluation & Visualization
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

## Reproducibility & Consistency

### Strict Seed Control
All scripts use a fixed seed (`42`) for random number generators (`numpy.random`, `torch.manual_seed`, etc.) and data shufflers to ensure that all partitions, weight initializations, and sample selections are identical across runs.

### Deterministic CUDA execution
During training and evaluation, PyTorch's deterministic execution flags are explicitly enabled:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Expectation of Results
If you or anyone else clones the repository and runs the pipeline from scratch on the same hardware architecture (e.g., CPU or CUDA GPU) and library versions, the outputs will **exactly match** the published metrics (including the gold 4-class macro F1 score of **0.5644** for HMR-BiLSTM). 

*Note: Minor round-off variations (e.g., in the range of $10^{-6}$) might occasionally occur due to floating-point differences across different GPU architectures or PyTorch versions.*

---

## Results

*Tables 1‑5 are stored in `results/tables/` as LaTeX and CSV files.*

---

## Research Contributions

- **Trustworthy healthcare AI:** Full adversarial robustness evaluation under FGSM and PGD attacks.
- **Explainable ECG classification:** Intrinsic interpretability through residual gate trajectory visualization.
- **Robust temporal modeling:** Hybrid memory decomposition with adaptive residual gating (RMC).
- **Improved calibration:** Lower ECE and Brier score vs. baselines for reliable probability outputs.
- **Clinical relevance:** Significant F1 improvement on minority arrhythmia classes (S, V, F).

---

## Keywords

ECG Arrhythmia Classification, Explainable AI, Trustworthy AI, BiLSTM, Residual Memory Control, Adversarial Robustness, FGSM, PGD, Biomedical Signal Processing, Deep Learning, MIT‑BIH

---

## Citation

```bibtex
@article{hmr_bilstm_2026,
  title={HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification},
  author={Anonymous},
  year={2026}
}
```
