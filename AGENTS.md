# Agent entry point

This repo is the HMR-BiLSTM ECG paper + an active R&D effort to fix the rare
classes (F = fusion beats, S = supraventricular). N and V are already solved.

## Before doing rare-class (F/S) work, READ:

**[`docs/RARE_CLASS_PROGRESS.md`](docs/RARE_CLASS_PROGRESS.md)** — the lab notebook.

It has: the problem decomposed into tiers (all evidence measured), the full list of
~25 experiments already tried and rejected (§5 DEAD ENDS — do not repeat these),
the current target architecture, the running experiment, and how to run/read
everything. **Keep its CURRENT STEP block and EXPERIMENT LOG table up to date after
every run.**

## Hard constraints

- 4 GB GPU, 14 GB RAM — **one training job at a time**, ~11–15 min/epoch.
- The machine reboots intermittently — every trainer must checkpoint per epoch and
  resume (`train_hd_ladder.py` / `train_crossdb_multiseed.py` already do).
- Git Bash on Windows: use `ps -W` to check if a PID is alive, not `tasklist`.
- Never claim an experiment "works" from one seed — 2 seeds minimum, and success
  means *mean up AND seed variance down*, not a higher single-seed number.

## Repo conventions

- Training scripts import the official recipe (cosine LR, FGSM adv-training,
  epochs=45, patience=15) from `train_inter_patient.py` — do not re-implement it.
- Official checkpoint `results/checkpoints/inter_best_rlstm.pt` is `num_layers=2`;
  load `ck['model_state']`.
- Data lives in git-ignored `data/` — regenerate with `validation/preprocess_*.py`.
