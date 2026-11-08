# Reviewer workflow

Run from the repository root after installing `requirements-reproduce.txt`.

```bash
python -m reviewer
```

The workflow runs three CPU tasks in order.

| Step | Reads | Writes | Success criterion |
|---|---|---|---|
| `tables` | Snapshot manifest, aggregate run metrics and numeric tables | `logs/tables.txt` | 150 file checksums verified and 729 values reproduced |
| `ranking` | Six-core validation/test scores and paired-seed summaries | `ranking/`, `logs/ranking.txt` | Diagnostic JSON matches the snapshot exactly |
| `figures` | Confusion counts, transfer AUCs and validation/test scores | `figures/`, `logs/figures.txt` | Three figures, each exported as PDF, SVG and PNG |

Outputs default to `outputs/reviewer/`. `reviewer_report.json` records the code
commit, interpreter, platform, package versions and each step's result. A failed
step stops execution and retains its log and failure status.

## Run one task

```bash
python -m reviewer --step tables
python -m reviewer --step ranking
python -m reviewer --step figures
```

## Select inputs and outputs

```bash
python -m reviewer --snapshot paper_results/v1 --output outputs/reviewer-check
```

Generated outputs must be outside the versioned snapshot. The workflow uses
saved aggregate results and standard CPU packages. Training instructions and
the individual module commands are in [REPRODUCING.md](../REPRODUCING.md).
