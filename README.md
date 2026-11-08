# ResMemECG — controlled ECG evaluation

**Study title** · *Weak Agreement Between Validation and Test Rankings of Recurrent ECG Classifiers: A Controlled, Subject-Grouped Evaluation*

This repository examines how training recipe, recurrent-cell implementation,
random seed and input selection affect ECG classification on MIT-BIH. The public
snapshot contains aggregate results from **129 runs**, exact fold manifests and
the numerical inputs used for the main comparisons.

## Start here

| What you want to do | Open or run |
|---|---|
| Check the reported numbers and regenerate the main diagnostic plots | Follow the quick start below, then run `python -m reviewer` |
| Inspect exact train/validation/test records and N/S/V/F/Q counts | [Fold manifest](paper_results/v1/results/tables/sgcv_fold_manifest.csv) · [Record assignments](paper_results/v1/results/tables/sgcv_record_partition.csv) |
| Find the data behind a table or figure | [Evidence map](docs/REPOSITORY_MAP.md#evidence-map) |
| Read metric definitions and snapshot provenance | [Snapshot documentation](paper_results/v1/README.md) |
| Recompute one analysis or train models | [Reproduction guide](REPRODUCING.md) |
| Locate the architecture and training implementation | [Repository map](docs/REPOSITORY_MAP.md) |

## Quick start on CPU

The verified environment is Windows with Python 3.13.2. Package versions,
including dependencies, are pinned in [requirements-reproduce.txt](requirements-reproduce.txt).
Run commands from the repository root.

```bash
git clone --depth 1 https://github.com/Lamno1/HMR-BiLSTM-Trustworthy.git
cd HMR-BiLSTM-Trustworthy
```

**PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduce.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m reviewer
```

**Bash on Linux or macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python -m pip check
.venv/bin/python -m reviewer
```

The same entry point runs the table check, ranking analysis and figure export.
Platform and package versions are recorded in the generated report.

```text
outputs/reviewer/
├── reviewer_report.json
├── logs/
│   ├── tables.txt
│   ├── ranking.txt
│   └── figures.txt
├── ranking/
│   ├── ranking_cells.csv
│   └── ranking_diagnostics.json
└── figures/
    ├── confusion_matrix_seeds.{pdf,svg,png}
    ├── transfer_auc.{pdf,svg,png}
    └── validation_agreement.{pdf,svg,png}
```

### Expected checks

- **Tables** — `150 file checksums verified; 729 values reproduced`.
- **Ranking** — diagnostic JSON matches the snapshot. Mean validation-selection
  regret is approximately **0.016231**, compared with **0.030136** for uniform
  random selection and **0.024693** for fixed No-RMC.
- **Figures** — three plots exported as PDF, SVG and PNG.
- **Report** — `outputs/reviewer/reviewer_report.json` finishes with `"status": "passed"`.

Run a single task with `python -m reviewer --step tables`, `--step ranking` or
`--step figures`. See [reviewer/](reviewer/README.md) for input/output options.

## How the repository is organised

```text
HMR-BiLSTM-Trustworthy/
├── README.md                       # Reviewer entry page
├── REPRODUCING.md                  # Detailed commands and training paths
├── requirements-reproduce.txt      # Pinned CPU verification environment
├── requirements.txt                # Training and broader experiment dependencies
├── reviewer/                       # Single entry point for the public checks
│   ├── __main__.py
│   └── README.md
├── paper_results/v1/               # Versioned aggregate evidence
│   ├── manifest.json               # File hashes and implementation provenance
│   ├── SHA256SUMS
│   ├── table_values.json            # 729 unrounded numeric table entries
│   └── results/
│       ├── matched_recipe/         # 20 configurations × 3 seeds
│       ├── matched_recipe_mlii/    # 3 cores × 3 seeds
│       ├── controlled_mit_subject_cv/  # Full model, 5 folds × 2 seeds
│       ├── sgcv_arch/              # Five other cores on the same fold–seed cells
│       ├── tables/                 # Fold manifests, ranking and transfer summaries
│       └── p2_final/               # Selective-prediction aggregate diagnostics
├── validation/                     # Preprocessing, protocols and analysis implementations
├── docs/                           # Repository/evidence map and protocol descriptions
├── hmr_bilstm.py                    # ResMemECG implementation, RLSTMClassifier
├── train_inter_patient.py          # Official training recipe
├── train_matched_recipe.py         # Architecture controls and paired-seed experiments
├── configs/                        # Experiment configuration and path helpers
├── calibration/                    # Calibration analyses
├── explainability/                 # Attribution analyses
├── robustness/                     # Perturbation evaluation
├── selective/                      # Selective-prediction policy
└── uncertainty/                    # Uncertainty analyses
```

The tree highlights the public reviewer path and core implementation. The
[full map](docs/REPOSITORY_MAP.md) identifies additional training and evaluation
modules. Generated outputs are written to the ignored `outputs/` directory.

## Main results at a glance

Macro F1 is averaged over N/S/V/F. Q remains a fifth output and predictions into
Q count as errors. Values below are mean macro F1 with sample SD.

| Evaluation | ResMemECG | No-RMC | Standard BiLSTM | Replicates used for SD |
|---|---|---|---|---|
| DS1/DS2, original channel-0 inputs | 0.559 (0.061) | 0.536 (0.045) | 0.473 (0.015) | Three training seeds |
| DS1/DS2, MLII selected by name | 0.532 (0.014) | 0.496 (0.016) | 0.480 (0.011) | Three training seeds |
| Subject-grouped CV | 0.475 (0.107) | 0.477 (0.112) | 0.449 (0.098) | Five fixed folds × two seeds |

These comparisons use the matched harness. The source run files and table
definitions are linked in the [evidence map](docs/REPOSITORY_MAP.md#evidence-map).

### Interpretation boundaries

- The RMC contrast depends on implementation and input conditions. The
  original-input no-dropout mean contrast is about −0.002. The MLII contrast is
  about +0.036, with corresponding MLII no-dropout controls unavailable.
- Fusion-beat generalisation remains weak. Mean F1-F under subject-grouped CV
  is at most 0.012 across the evaluated cores.
- Validation/test exact rankings agree weakly in these runs. Validation-based
  selection still has lower observed regret than the two comparators above.
- Record-bootstrap intervals condition on fitted models. The fixed-fold
  summaries also depend on the chosen partition and training seeds.

## Protocol details to check

| Item | Implementation and interpretation |
|---|---|
| DS1/DS2 | Record-disjoint. Records 201 and 202 belong to one subject and lie on opposite sides of this split |
| Subject-grouped CV | Five folds over 43 groups from 44 retained records. Records 201/202 are bound together |
| Record 208 | Outer-test in fold 0 and inner-validation in folds 1–4 |
| Channel selection | Original channel 0 is V5 for record 114. The MLII sensitivity selects the lead by name and refits training normalization |
| Rare-class support | Inspect N/S/V/F/Q counts before interpreting fold scores. Training N is capped at 37,208 per fold |

See the [controlled protocol](docs/CONTROLLED_MIT_SUBJECT_CV_PROTOCOL.md) and
[fold manifest](paper_results/v1/results/tables/sgcv_fold_manifest.csv).

## Data, versions and attribution

The aggregate snapshot is pinned by tag
[`results-v1-2026-11-07`](https://github.com/Lamno1/HMR-BiLSTM-Trustworthy/tree/results-v1-2026-11-07/paper_results/v1).
Its authored aggregate files and manifests use [CC BY 4.0](paper_results/v1/DATA_LICENSE.md).
The manifest records the model implementation commit and original artifact hashes.
The reviewer report records the code commit actually executed.

Original ECG recordings are available from
[PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/).
Full training uses those recordings and the preprocessing/training commands in
[REPRODUCING.md](REPRODUCING.md#training-from-source-data).

When citing or reporting a reproduction, record the study title, repository URL,
code commit, snapshot tag and environment from `reviewer_report.json`.
