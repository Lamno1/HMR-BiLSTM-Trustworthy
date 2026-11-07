# ResMemECG aggregate-results snapshot v1

This snapshot supports the controlled evaluation of recurrent ECG classifiers.
It contains aggregate metrics for 60 matched-harness runs, 9 MLII-by-name runs
and 60 subject-grouped runs. It also provides record/fold manifests, selected
transfer and selective-prediction diagnostics, and 729 numeric table entries.

The snapshot tag is `results-v1-2026-11-07`. Cite the tag and its resolved Git
commit when using these files. `manifest.json` identifies the model implementation
commit and SHA-256 digests. `SHA256SUMS` also covers the manifest. The snapshot
data license is CC BY 4.0, as described in `DATA_LICENSE.md`.

## Check from a fresh clone

The public table check needs Python, NumPy and scikit-learn.

```bash
python -m pip install numpy scikit-learn
python -m validation.check_bel_tables --public-snapshot paper_results/v1
```

The command checks the snapshot file digests and reproduces all 729 table values
from the published run metrics, confusion counts and archived bootstrap summaries.
Training, local ECG arrays and manuscript files are outside this check.
The original local manuscript check remains available as
`python -m validation.check_bel_tables` when the private working files exist.

## File layout and metric dictionary

- `results/matched_recipe/` has one JSON per architecture and training seed.
  The `recipe` field records the actual run settings. `test` contains fractional
  accuracy, F1, per-class precision/recall/F1/AUC and support counts.
- `results/matched_recipe_mlii/` has the nine sensitivity runs with MLII selected
  by channel name. Original channel0 uses V5 for record114. Refit training
  normalization is part of the input-protocol change.
- `results/controlled_mit_subject_cv/` has the ten full-model fold/seed summaries
  and five-class confusion matrices. `results/sgcv_arch/` has the other50runs.
  `test_metrics` stores macroF1 and per-class results.
- `results/tables/sgcv_fold_manifest.csv` lists fold, partition, record/group
  counts, N/S/V/F/Q support and record IDs. `train_after_n_control` gives support
  after the N cap. `sgcv_record_partition.csv` maps records into every fold.
- `sgcv_arch_runs.csv` gives core/fold/seed validation and test macroF1.
  `ranking_cells.csv` adds picks, ties, rank correlation and regret.
  `ranking_diagnostics.json` summarises the ten fixed fold/seed cells.
- `matched_diagnostics.json` supplies full-model confusion matrices and stored
  record-cluster bootstrap contrasts. `per_class_ds2.json` supplies the eight
  separate-recipe baseline confusion matrices. Current ResMemECG rows come from
  the matched-harness run summaries and confusion matrices.
- `mechanism_transfer.json` and `mechanism_lbbb*.json` supply transfer AUCs.
  Their score is the F logit minus the largest other-class logit, restricted to
  true F versus N-class or L-labelled beats. General class AUC instead uses
  one-versus-rest softmax probability.
- `input_sensitivity.json` contains paired MLII contrasts and bootstrap settings.
  `sgcv_variance_decomposition.json` describes this fixed partition.
- `results/p2_final/` contains seed-specific risk/coverage counts, the displayed
  GO assessment and F-referral diagnostics from the evaluated selective policy.
  Dataset is MIT-BIH or INCART, seed identifies a fitted model, and alpha is
  the evaluated grid value. Coverage/risk/precision are fractions. Auto-S
  coverage divides accepted correct-S decisions by all true-S beats. F leakage
  divides accepted true-F beats by all true-F beats. Fold GO summaries pool the
  two seed runs at each alpha.
- `provenance.json` records local input artifact names and their original hashes.
  `table_values.json` supplies unrounded numeric values in manuscript table order.

## Table numeric ordering

Rows and core ordering are fixed in `validation/check_bel_tables.py`.
Numeric arrays omit textual cells and dashes.

| Table key | Numeric fields in each row |
|---|---|
| `tab:baselines` | Accuracy, macroF1, F1-N, then precision/recall/F1 for S,V,F |
| `tab:literature` | Own three seeds only, S recall percent, S precision percent, S F1, then the same for V |
| `tab:matched` | MacroF1 mean, sample SD, F1-S, F1-V, F1-F, AUC-F |
| `tab:cellimpl` | Model rows have macroF1 mean/SD, optional three paired seed differences and two CI bounds, then F1-S/V/F. The last contrast-only row has three paired differences and two CI bounds |
| `tab:bootstrap` | Point, two CI bounds, fraction of positive resamples, two leave-one-record-out range bounds |
| `tab:transfer` | F/N AUC train/validation/test, optional F/L train, F/L test |
| `tab:sgcv-folds` | S/F support, S precision/recall/F1, F precision/recall/F1, AUC-S/F. Final row sums support and averages fold metrics |
| `tab:sgcv` | Validation mean, test mean/SD, optional difference and two CI bounds, F1-S/V, validation pick count, test-best count |
| `tab:mlii-sensitivity` | Seed, best epoch, macroF1, F1-S/V/F |
| `tab:ranking-val`, `tab:ranking-test` | Fold, seed, full, No-RMC, No-RMC no-dropout, LSTM, LSTM no-class-weights, LSTM cross-entropy |

## Interpretation and coverage

Class order is N,S,V,F,Q. Headline macroF1 averages N/S/V/F and retains Q
predictions as errors. DS1/DS2 shares a subject through records201/202.
Subject-grouped CV binds them together. Record208 is outer-test in fold0 and
inner-validation in folds1–4. Training N support is capped at37,208 per fold.

Record-bootstrap intervals condition on fitted models. Fold-bootstrap intervals
describe one fixed partition. Seeds share the same evaluation subjects. A narrow
conditional interval alone cannot resolve training variation. Original-input
no-dropout full–No-RMC mean is−0.002, whereas the MLII contrast is+0.036.
MLII no-dropout controls are unavailable.

This release supports table-level reproducibility and selected diagnostics.
Raw ECG, beat-level predictions and fitted model weights remain in their
respective source/local storage. Original MIT-BIH data can be obtained from
https://physionet.org/content/mitdb/1.0.0/.
External literature rows and other exploratory supplement analyses require their
own source checks. Manuscript sources and internal review documents remain in
the authors' working files.
