# Legacy mixed-source record-grouped evaluation protocol (locked 2026-08-29)

> **Terminology/provenance notice (2026-09-09).** This protocol is RG-CV, not
> traditional leave-one-patient-out and not guaranteed subject-disjoint. MIT-BIH
> records 201/202 cross its folds, and no INCART subject map was available. It is
> retained unchanged for the historical R2, forensic, and P2 comparisons. The
> primary controlled MIT-only subject-grouped experiment is documented in
> `docs/CONTROLLED_MIT_SUBJECT_CV_PROTOCOL.md`. Do not transfer results between
> these protocols or relabel legacy artifacts as subject-grouped.

Primary evaluation for the rare classes F and S. Locked before any HMR-MultiPath
module is built, so results across rungs are comparable. Do not change without a
recorded reason here.

## Why

Fixed DS1/DS2: test-F is 93% patient 213, test-S is 75% patient 232 → F1-F/F1-S
are patient-composition-fragile point estimates. Record-level CV turns them into
a distribution over held-out records. Because record identity is not equivalent
to subject identity here, it does not by itself establish a distribution over
unseen patients.

## Pool

`validation/preprocess_lopo.py` → `data/processed/lopo/pool.npz`
- MIT-BIH: 44 records, lead MLII (`p_signal[:,0]`), official window (−90/+97 @ 125 Hz).
- INCART: 36 F/S-bearing records, lead II, 0.5 Hz Butterworth high-pass + R-peak
  re-center + per-record polarity fix + mean-N morphology filter (min-corr 0.4).
- 185 252 beats, 80 records. F from 34 records (17 MIT + 17 INCART), S from 62.
- **No normalization stored** — fit per fold (see below).
- When the V1 branch lands: pool must also carry V1 (`preprocess_lopo.py` update —
  MIT-BIH `p_signal[:,1]`, INCART lead `V1`).

## Fold construction (`train_lopo_cv.py::assign_folds`)

- Unit = record. No record's beats in both train and test of any fold.
- K = 5. Snake-draft records ordered by class-of-interest beat count descending
  (F-records first, then S-only, then rest) → the huge lumps (208, 213, 232) land
  in different folds; every fold gets big + small + INCART F/S records.
- Fold F-beat counts (fold-seed 0): 399 / 388 / 82 / 72 / 42.
  Fold S-beat counts: 388 / 391 / 576 / 860 / 2490.
  Heterogeneous by construction — the five values are a held-out-record
  distribution, not an i.i.d. patient sample.

## Per-fold pipeline

1. `outer_test` = the fold's records.
2. `inner_val` = ~15% of the remaining records (`carve_inner_val`), greedy-balanced
   on F/S/source, record-disjoint. **Used for model selection only — never sees
   outer_test.**
3. `train` = the rest. Train-N subsampled to `--train-n-cap` (default 35000) to
   keep per-epoch cost ≈ DS1; all S/V/F/Q kept. inner_val and outer_test kept full.
4. Normalization: per-source z-score, mean/std fit on **train-fold records only**,
   applied to train/inner_val/outer_test.
5. Recipe: official (`train_inter_patient` functions) — cosine LR, FGSM adv
   (deterministic clean/adv split, task-loss FGSM), Focal γ=1.5 + class weights
   from train-fold. Epochs reduced (`--epochs`, default 25).
6. Model selection: best `inner_val` macro-F1 over **{S,V,F}** (labels 1,2,3) —
   not N/S/V/F, so N/V stability does not dominate the pick.
7. **SWA**: average model weights over the tail epochs (from epoch `0.6·E`);
   recompute BatchNorm stats on train; the SWA model is the fold's final model.
   Also keep the best-epoch model for comparison. (This is the concrete fix for
   the epoch-lottery, failure mode B2.)
8. Score `outer_test`: F1 per class, reported for `all` / `mitbih` / `incart`.

## Reporting

Per rung: per-fold F1-F, F1-S, F1-macro; mean ± std over the 5 folds; the same
split by database (MIT-BIH vs INCART); F confusion. Report SWA and best-epoch
side by side.

## Sentinel screening fold (gate before full LOPO)

Full LOPO (5 folds × inner selection) is expensive. Before running it for a rung:

`python train_lopo_cv.py --mode sentinel --arch <arch> --epochs <E>`

Fixed held-out test = `mitdb/213` (F), `mitdb/232` (S), `incart/I18`,
`incart/I74`, `incart/I33`, `incart/I62` (largest cross-DB F/S records). 208 stays
in train. One training run.

**Promotion rule**: a rung earns a full 5-fold LOPO (2 seeds) only if, on
sentinel, it improves F1(213-F) and/or F1(232-S) **without** dropping F1-N or
F1-V versus the previous kept rung. Rungs that fail sentinel are logged as
rejected and not run further.

## Wall-time

Benchmark one fold before trusting any total estimate — sentinel mode prints
`[BENCHMARK] one fold = N min => full 5-fold ~ H h/seed`. Full LOPO with inner
selection is meaningfully slower than a DS1/DS2 run.

## What still gets a DS1/DS2 run

The final kept architecture was planned for the official DS1/DS2 split, 3 seeds,
mean ± std, for literature comparability. That D1 run was not completed. After
the 2026-09-09 audit, controlled SG-CV is the primary generalisation check and
this RG-CV run is retained for the R2/P2 diagnostic comparison.
