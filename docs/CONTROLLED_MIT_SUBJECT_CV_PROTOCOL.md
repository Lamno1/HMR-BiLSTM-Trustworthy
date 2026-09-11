# Controlled MIT-BIH subject-grouped cross-validation protocol

Protocol ID: `C1_MIT_SUBJECT_CV_V1`  
Status: completed, 2026-09-09

## Purpose

This experiment tests the flat ResMemECG model under a MIT-BIH-only,
subject-grouped protocol while matching the conventional training schedule and
number of model-training N beats. It was added after auditing the earlier
mixed-source record-grouped experiment. It must not be called leave-one-patient-
out: each of the five outer folds contains multiple held-out subject groups.

## Data and grouping

- Source: MIT-BIH only (`source == 0` in `data/processed/lopo/pool.npz`).
- Retained data: 44 non-paced records, forming 43 subject groups.
- Known repeated subject: records 201 and 202 are bound to one group.
- Split unit: complete subject group; no subject group crosses model-train,
  inner-validation, or outer-test within a fold.
- Five folds are snake-drafted by F count and then S count. Subject integrity
  takes priority over equal beat support.
- Outer-test F support: 376/367/19/17/23.
- Outer-test S support: 210/547/521/1,463/39.

The exact group membership and class counts are frozen in
`results/controlled_mit_subject_cv/protocol_manifest.json`.

## Model and controls

- Model: flat `hmr_bilstm.RLSTMClassifier`, two recurrent layers.
- Seeds: 42 and 123.
- Maximum epochs: 45; early-stopping patience: 15.
- Batch size: 128; learning rate: 0.001; minimum LR: 0.00001.
- Cosine schedule, focal loss (`gamma=1.5`), smoothness weight 0.003, FGSM
  adversarial training (`epsilon=0.02`, ratio 0.3).
- Model-training N count: exactly 37,208 in every fold, matching the official
  DS1 model-training count; all available S/V/F/Q training beats are retained.
- Normalisation is fitted on model-train beats only.
- Selection: best inner-validation macro-F1 over N/S/V/F.
- No outer-test fitting or tuning.
- Checkpoint/resume state is written every epoch.

## Reporting

The two seeds are paired optimisation repetitions on the same held-out groups,
not independent observations. Results first average seeds within each fold and
then report all five fold values. Beat pooling is not the primary summary.

The conventional DS1/DS2 result is a fixed-split reference rather than a paired
five-fold experiment. Their difference is therefore described as observed
protocol/subject-composition sensitivity, never as a causal estimate of the
effect of the split.

## Artifacts

- Trainer: `validation/run_controlled_mit_subject_cv.py`
- Results: `results/controlled_mit_subject_cv/`
- Integrity checker: `validation/check_controlled_mit_subject_cv.py`
- Independent audit: `results/controlled_mit_subject_cv/integrity_report.json`
- Interpretation: `results/controlled_mit_subject_cv/REPORT.md`
