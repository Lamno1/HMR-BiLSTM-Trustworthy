# Reproducing the study

## Choose the task

| Task | Environment | Input | Output |
|---|---|---|---|
| Verify tables, recompute ranking and recreate figures | Pinned CPU environment | `paper_results/v1/` | `outputs/reviewer/` |
| Run a single saved-results analysis | Same CPU environment | Same snapshot | Directory selected below |
| Train models from source recordings | Training dependencies and suitable compute | PhysioNet recordings and preprocessed arrays | `results/` and checkpoints |

The CPU workflow was verified on Windows with Python 3.13.2. Full package
versions are recorded in `requirements-reproduce.txt`. All commands below run
from the repository root.

## Install the reviewer environment

```bash
git clone --depth 1 https://github.com/Lamno1/HMR-BiLSTM-Trustworthy.git
cd HMR-BiLSTM-Trustworthy
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in bash. Alternatively, call the environment's
interpreter directly as shown in the main README.

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip check
python -m reviewer
```

Expected results are 150 verified file checksums, 729 reproduced table values,
ranking diagnostics equal to the snapshot, and three figures in PDF/SVG/PNG.
The final `outputs/reviewer/reviewer_report.json` records a passed status,
the exact code commit, Python/platform details and package versions.

## Run individual reviewer tasks

```bash
python -m reviewer --step tables
python -m reviewer --step ranking
python -m reviewer --step figures
```

To change the output directory, append `--output outputs/my-review`.
Detailed logs for each step are retained under the output directory's `logs/`.

## Direct analysis module commands

These are the same implementations called by `reviewer/`.

```bash
python -m validation.check_bel_tables --public-snapshot paper_results/v1
python -m validation.audit_selection_rankings --results-dir paper_results/v1/results --out-dir outputs/reproduced-ranking
python -m validation.plot_submission_figures --tables-dir paper_results/v1/results/tables --out-dir outputs/reproduced-figures
```

The table check reproduces numeric entries from aggregate run metrics, confusion
counts and stored bootstrap summaries. The ranking check recomputes the ten
fold–seed cells and paired-seed contrasts. The figure exporter uses stored
confusion matrices, per-checkpoint AUCs and per-core validation/test scores.

Expected ranking values are mean within-cell Spearman 0.08, test-best selected
in 2/10 cells, and mean regrets of approximately 0.016231 for validation,
0.030136 for uniform random selection and 0.024693 for fixed No-RMC.

The optional plotting argument `--qa-tools PATH` enables alignment checks from
an installed nature-figure scripts directory. The standard reviewer workflow
uses the pinned public Python dependencies.

## Training from source data

Training uses `requirements.txt` and the official configuration in
`train_inter_patient.py`. Assess the installed PyTorch build and compute
requirements on the training machine. Run one training job at a time. The
trainers save resume state each epoch and skip completed run summaries.

### Original-input DS1/DS2 comparisons

```bash
python -m pip install -r requirements.txt
python -m validation.download_mitdb
python -m validation.preprocess_aami
python train_matched_recipe.py --arch full --seeds 42 123 456
python train_matched_recipe.py --arch normc --seeds 42 123 456
python train_matched_recipe.py --arch bilstm --seeds 42 123 456
```

Data are read from `data/raw/mitdb/`, and the split arrays are written to
`data/processed/splits/`. This protocol uses channel 0, which is V5 for record
114. Records 122, 124, 205 and 223 form validation. Records 201 and 202 share a
subject across DS1/DS2. The official recipe has 45 maximum epochs, patience 15,
cosine learning-rate scheduling, weighted focal loss and FGSM training.

Run `python train_matched_recipe.py --help` to inspect the supported architecture
keys and options. The public snapshot stores the actual recipe and best epoch
for each run under `paper_results/v1/results/matched_recipe/`.

### MLII-by-name sensitivity

After the original split preparation, run the sensitivity preprocessing and
select its data directory explicitly.

```bash
python -m validation.preprocess_aami_mlii
python train_matched_recipe.py --arch full --seeds 42 123 456 --data-dir data/processed/splits_mlii --out-dir results/matched_recipe_mlii
python train_matched_recipe.py --arch normc --seeds 42 123 456 --data-dir data/processed/splits_mlii --out-dir results/matched_recipe_mlii
python train_matched_recipe.py --arch bilstm --seeds 42 123 456 --data-dir data/processed/splits_mlii --out-dir results/matched_recipe_mlii
```

This selects MLII by channel name and refits normalization on training data.
Compare the stored run summaries and conditional contrasts in the snapshot's
`results/tables/input_sensitivity.json`.

### Subject-grouped cross-validation

The controlled trainer consumes `data/processed/lopo/pool.npz` and
`records.json`, prepared by `validation/preprocess_lopo.py`. That preprocessing
module builds a mixed-source pool from MIT-BIH and INCART inputs. The controlled
trainer then selects MIT-BIH records and binds 201/202 into one subject group.
Inspect the module's input paths and the
[controlled protocol](docs/CONTROLLED_MIT_SUBJECT_CV_PROTOCOL.md) when preparing
the source recordings.

Once these pool files and original DS1 training arrays are available, inspect
the locked split before training.

```bash
python -m validation.run_controlled_mit_subject_cv --audit-only
python -m validation.run_controlled_mit_subject_cv --seeds 42 123
python -m validation.run_sgcv_arch --arch normc --seeds 42 123
```

The first training command produces the full-model runs. The second illustrates
one control. The five control keys in the published SG-CV snapshot are `normc`,
`normc_nodrop`, `bilstm`, `bilstm_nocw` and `bilstm_ce`. Review the train/validation/
test record membership and N/S/V/F/Q counts against the published manifests.
Timing and newly fitted weights will depend on the training environment.

## Data interpretation

Class order is N,S,V,F,Q. Macro F1 averages N/S/V/F while retaining Q predictions
as errors. The subject-grouped evaluation uses five fixed folds and two paired
training seeds. F generalisation remains weak in these runs. Record-bootstrap
intervals condition on fitted models, and fold summaries depend on the selected
partition. See the [snapshot dictionary](paper_results/v1/README.md) and
[evidence map](docs/REPOSITORY_MAP.md#evidence-map).

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named reviewer` | Run from the root containing `reviewer/`, using the current checkout |
| A Python dependency is missing | Use the same environment to install `requirements-reproduce.txt` and execute the commands |
| A checksum differs | Compare with a fresh checkout of the versioned snapshot before using the changed data |
| A step fails | Read its log and the failed status in `reviewer_report.json` |
| Training cannot find data or checkpoints | Follow the source-data preparation above and the paths in the corresponding trainer |
