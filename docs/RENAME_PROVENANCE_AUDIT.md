# ResMemECG rename and numerical-artifact provenance audit

Date: 2026-09-09

## Naming strategy

`ResMemECG` is the journal-facing model name. Historical Python module names,
class names, checkpoint keys, experiment IDs, and committed artifact paths retain
their original HMR/RLSTM identifiers where changing them would break imports or
make past runs irreproducible. The README records this mapping explicitly.

## Working-tree separation

The current working tree contains three different change classes and they must
not be committed as one undifferentiated rename:

1. **Presentation rename:** prose, table labels, plot labels, and documentation
   changing HMR-BiLSTM to ResMemECG. These do not change scientific values.
2. **New scientific evidence:** `results/controlled_mit_subject_cv/`, its trainer,
   checker, report, and controlled-protocol documentation. These change the
   manuscript's primary generalisation estimate and require their own provenance.
3. **Recomputed/expanded legacy outputs:** `results/logs/baseline_results.json`
   contains changed logistic-regression values and newly added InceptionTime,
   TCN, and ECG-Transformer entries; several robustness/calibration tables and
   binary figures have also changed. These are not a pure model rename and must
   be reviewed against their generating scripts before being committed as
   evidence.

No historical checkpoint or prediction artifact was rewritten by the C1 audit.
The manuscript now cites C1 values only from
`results/controlled_mit_subject_cv/integrity_report.json`; the legacy RG-CV R2
and P2 values retain their original committed sources.

## Required commit discipline

- Commit C1 code, manifest, prediction/model artifacts, integrity report, and
  protocol documentation together or with an explicit SHA-256 manifest.
- Commit manuscript integration separately.
- Commit presentation-only renames separately from recomputed numerical tables.
- Do not describe `baseline_results.json` as rename-only until its changed and
  newly added model rows have been reproduced from the associated scripts.
