"""Compute descriptive ranking and selection regret from saved metrics.

Run with ``python -m validation.audit_selection_rankings``. Outputs are written
to results/tables. The script reads summaries without training or inference.
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
CORES = ["full", "normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    source_tables = args.results_dir / "tables"
    if args.out_dir.resolve().is_relative_to((ROOT / "paper_results").resolve()):
        raise ValueError("Write derived files outside the versioned snapshot")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (source_tables / "sgcv_arch_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        (row["core"], int(row["fold"]), int(row["seed"])):
        (float(row["inner_val_macro_f1"]), float(row["outer_test_macro_f1"]))
        for row in rows
    }
    if len(rows) != 60 or len(scores) != 60:
        raise ValueError("Expected 60 distinct core/fold/seed runs")
    cells = []
    for fold in range(5):
        for seed in (42, 123):
            validation = np.array([scores[core, fold, seed][0] for core in CORES])
            test = np.array([scores[core, fold, seed][1] for core in CORES])
            selected = int(validation.argmax())
            best = int(test.argmax())
            cell = {
                "fold": fold, "seed": seed,
                "val_pick": CORES[selected], "test_best": CORES[best],
                "rho": float(spearmanr(validation, test).statistic),
                "selected_test": float(test[selected]), "best_test": float(test.max()),
                "validation_regret": float(test.max() - test[selected]),
                "uniform_random_expected_regret": float(test.max() - test.mean()),
                "fixed_normc_regret": float(test.max() - test[CORES.index("normc")]),
                "validation_tied_max": int((validation == validation.max()).sum()),
                "test_tied_max": int((test == test.max()).sum()),
            }
            cell.update({f"{core}_validation": float(validation[i]) for i, core in enumerate(CORES)})
            cell.update({f"{core}_test": float(test[i]) for i, core in enumerate(CORES)})
            cells.append(cell)
    result = {
        "n_cells": len(cells), "n_folds": 5, "n_training_seeds": 2,
        "mean_within_cell_rho": float(np.mean([cell["rho"] for cell in cells])),
        "within_cell_rho_range": [min(cell["rho"] for cell in cells), max(cell["rho"] for cell in cells)],
        "val_pick_test_best": sum(cell["val_pick"] == cell["test_best"] for cell in cells),
        "tie_rule": "first core in fixed listed order",
        "has_max_ties": any(cell["validation_tied_max"] > 1 or cell["test_tied_max"] > 1 for cell in cells),
        "mean_validation_regret": float(np.mean([cell["validation_regret"] for cell in cells])),
        "mean_uniform_random_expected_regret": float(np.mean([cell["uniform_random_expected_regret"] for cell in cells])),
        "mean_fixed_normc_regret": float(np.mean([cell["fixed_normc_regret"] for cell in cells])),
        "test_rank_cross_seed_rho_by_fold": {}, "paired_seed_differences": {},
    }
    for fold in range(5):
        first = [scores[core, fold, 42][1] for core in CORES]
        second = [scores[core, fold, 123][1] for core in CORES]
        result["test_rank_cross_seed_rho_by_fold"][str(fold)] = float(spearmanr(first, second).statistic)
    contrasts = [
        ("matched_recipe", "full", "normc"),
        ("matched_recipe", "full_nodrop", "normc_nodrop"),
        ("matched_recipe_mlii", "full", "normc"),
    ]
    for source, first, second in contrasts:
        differences = []
        for seed in (42, 123, 456):
            metrics = []
            for core in (first, second):
                path = args.results_dir / source / f"{core}_seed{seed}.json"
                metrics.append(json.loads(path.read_text())["test"]["f1_macro"])
            differences.append(metrics[0] - metrics[1])
        result["paired_seed_differences"][f"{source}/{first}-{second}"] = {
            "seeds": [42, 123, 456], "differences": differences,
            "mean": float(np.mean(differences)),
            "sample_sd": float(np.std(differences, ddof=1)),
        }
    result["limits"] = (
        "Descriptive on fixed cells/models. Does not resolve population rankings "
        "or estimate an interaction from interval inclusion."
    )
    (args.out_dir / "ranking_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out_dir / "ranking_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
