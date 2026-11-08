# Reproducing the published results

The versioned data snapshot is `paper_results/v1/` at tag
`results-v1-2026-11-07`. Start from the latest `main` for the reviewer commands
below. The snapshot files keep their original digests.

## Prepare a CPU environment

The commands were verified on Windows with Python 3.13.2. The pinned packages
support the table, ranking and figure checks. Run all commands from the repository
root. Use the shell's corresponding activation command.

```bash
git clone --depth 1 https://github.com/Lamno1/HMR-BiLSTM-Trustworthy.git
cd HMR-BiLSTM-Trustworthy
python -m venv .venv
```

PowerShell activation is `.venv\Scripts\Activate.ps1`. In bash on Linux or macOS,
use `source .venv/bin/activate`. If PowerShell execution policy prevents activation,
replace `python` below with `.venv\Scripts\python.exe`.

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip check
```

## Verify the numeric tables

```bash
python -m validation.check_bel_tables --public-snapshot paper_results/v1
```

Expected final output is `150 file checksums verified; 729 values reproduced`.
This recomputes the published numeric entries from saved aggregate run metrics,
confusion counts and archived bootstrap summaries. The local manuscript check
uses `python -m validation.check_bel_tables` when manuscript sources and full
working artifacts are available.

## Recompute model-selection diagnostics

```bash
python -m validation.audit_selection_rankings --results-dir paper_results/v1/results --out-dir outputs/reproduced-ranking
```

Outputs are `ranking_cells.csv` and `ranking_diagnostics.json`. Expected mean
regrets are approximately 0.016231 for validation selection, 0.030136 for uniform
random selection and 0.024693 for fixed No-RMC. Mean within-cell Spearman is 0.08
and validation selects the observed test-best core in 2/10 cells. No maximum ties
occur in these runs. The ten cells share subjects and folds.

## Recreate the three revised figures

```bash
python -m validation.plot_submission_figures --tables-dir paper_results/v1/results/tables --out-dir outputs/reproduced-figures
```

Expected outputs are PDF, SVG and PNG versions of `confusion_matrix_seeds`,
`transfer_auc` and `validation_agreement`. The script uses all stored seed
results. The optional `--qa-tools` argument accepts an installed nature-figure
scripts directory to run its alignment checks. Standard figure generation uses
the packages in `requirements-reproduce.txt`.

## Training and scope

The results snapshot covers 129 saved runs and selected diagnostics. Training uses
the separate `requirements.txt`, public PhysioNet input data and the preprocessing
scripts named in the project README. `train_matched_recipe.py` imports the
official recipe from `train_inter_patient.py`. Training checkpoints support
epoch-level resume. Training cost and GPU memory should be assessed separately.

N/S/V/F macroF1 retains predictions into output Q as errors. DS1/DS2 has subject
overlap through 201/202. Subject-grouped CV binds those records. Results for F
remain weak and vary with seed, held-out subjects and input protocol. In SG-CV,
mean F1-F is at most 0.012 across the evaluated cores. The study supports a
controlled benchmark evaluation rather than reliable rare-F automation.

Metric definitions, row ordering, class support and original artifact hashes are
in `paper_results/v1/README.md`, its manifests and provenance file. Original
datasets retain their PhysioNet terms. The authored aggregate data and manifests
use CC BY 4.0.
