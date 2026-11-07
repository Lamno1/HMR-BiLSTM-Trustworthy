"""Export a versioned aggregate-results snapshot from verified local artifacts.

Run ``python -m validation.export_public_results --out-dir paper_results/v1``.
The output directory must be new or empty. Review it before committing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from sklearn.metrics import confusion_matrix

from validation import check_bel_tables as tables

ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAMES = ["Logistic regression", "Decision tree", "LSTM", "BiLSTM", "ResNet1D",
                  "InceptionTime", "TCN", "ECG-Transformer"]
RUN_FIELDS = {"arch", "seed", "fold", "protocol_id", "n_params", "best_epoch",
              "best_val_f1_macro", "epochs_run", "epochs_completed", "stopped_early",
              "recipe", "test", "test_metrics", "normalization", "train_counts_after_n_control",
              "val_counts", "test_counts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty snapshot directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=output.parent) as staging:
        file_count, numeric_count = export_snapshot(Path(staging))
        if output.exists():
            output.rmdir()  # Only an empty destination is accepted above.
        Path(staging).rename(output)
    print(f"Exported {file_count} files and {numeric_count} numeric table values to {output}")


def export_snapshot(output):
    """Build in a temporary sibling directory and return only a complete snapshot."""
    provenance = []

    def export(relative, content=None, destination=None):
        source = ROOT / relative
        raw = source.read_bytes()
        provenance.append({"source": relative, "source_sha256": hashlib.sha256(raw).hexdigest()})
        target = output / (destination or relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            target.write_bytes(raw.replace(b"\r\n", b"\n"))
        else:
            target.write_bytes((json.dumps(content, indent=2, allow_nan=False) + "\n").encode())

    for folder in ["matched_recipe", "matched_recipe_mlii"]:
        paths = sorted((ROOT / "results" / folder).glob("*_seed*.json"))
        expected = 60 if folder == "matched_recipe" else 9
        if len(paths) != expected:
            raise ValueError(f"Expected {expected} runs in {folder}, found {len(paths)}")
        for path in paths:
            source = json.loads(path.read_text())
            filtered = {key: value for key, value in source.items() if key in RUN_FIELDS}
            export(path.relative_to(ROOT).as_posix(), filtered)

    for fold in range(5):
        for seed in (42, 123):
            relative = f"results/controlled_mit_subject_cv/fold{fold}_seed{seed}_flat.json"
            source = json.loads((ROOT / relative).read_text())
            filtered = {key: value for key, value in source.items() if key in RUN_FIELDS}
            predictions = ROOT / relative.replace(".json", "_preds.npz")
            with np.load(predictions) as saved:
                filtered["confusion_NSVFQ"] = confusion_matrix(
                    saved["y"], saved["logits"].argmax(axis=1), labels=list(range(5))).tolist()
            provenance.append({"source": predictions.relative_to(ROOT).as_posix(),
                               "source_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()})
            export(relative, filtered)
            for core in ["normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]:
                relative = f"results/sgcv_arch/{core}/fold{fold}_seed{seed}.json"
                source = json.loads((ROOT / relative).read_text())
                export(relative, {key: value for key, value in source.items() if key in RUN_FIELDS})

    table_files = ["sgcv_fold_manifest.csv", "sgcv_record_partition.csv", "sgcv_arch_runs.csv",
                   "ranking_cells.csv", "ranking_diagnostics.json", "ds2_validation_blindness.json",
                   "sgcv_validation_blindness.json", "sgcv_variance_decomposition.json",
                   "mechanism_transfer.json", "mechanism_lbbb.json", "mechanism_lbbb_train.json"]
    for name in table_files:
        export("results/tables/" + name)
    source = json.loads((ROOT / "results/tables/matched_diagnostics.json").read_text())
    export("results/tables/matched_diagnostics.json", {
        "confusion_full": source["confusion_full"], "record_bootstrap": source["record_bootstrap"]})
    source = json.loads((ROOT / "results/tables/per_class_ds2.json").read_text())
    export("results/tables/per_class_ds2.json", {name: source[name] for name in BASELINE_NAMES})
    sensitivity = json.loads((ROOT / "results/tables/advisor_revision_audit.json").read_text())
    export("results/tables/advisor_revision_audit.json",
           {"mlii": sensitivity["mlii"], "mlii_bootstrap": sensitivity["mlii_bootstrap"]},
           destination="results/tables/input_sensitivity.json")
    for name in ["continuous_s_risk_coverage.csv", "go_criterion.csv", "f_referral_diagnostics.csv"]:
        export("results/p2_final/" + name)

    functions = {label: fn for label, fn, _ in tables.CHECKS}
    functions["tab:mlii-sensitivity"] = tables.expect_mlii
    functions["tab:ranking-val"] = lambda: tables.expect_ranking_scores("inner_val_macro_f1")
    functions["tab:ranking-test"] = lambda: tables.expect_ranking_scores("outer_test_macro_f1")
    values = {
        label: {"values": [[value.item() if isinstance(value, np.generic) else value
                            for value in row] for row in fn()]}
        for label, fn in functions.items()
    }
    numeric_count = sum(len(row) for entry in values.values() for row in entry["values"])
    (output / "table_values.json").write_bytes((json.dumps({
        "schema_version": 1, "numeric_count": numeric_count, "tables": values
    }, indent=2, allow_nan=False) + "\n").encode())

    # Keep provenance portable and inspect exported strings for local credentials/paths.
    (output / "provenance.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    for path in output.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if re.search(r"[A-Za-z]:[\\/]|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_", text):
                raise ValueError(f"Review a local path or credential-like value in {path.name}")

    # Snapshot-specific documentation is supplied as a reviewed template beside this exporter.
    template = ROOT / "validation/public_results_readme.md"
    (output / "README.md").write_bytes(template.read_bytes().replace(b"\r\n", b"\n"))
    license_template = ROOT / "validation/public_results_license.md"
    (output / "DATA_LICENSE.md").write_bytes(license_template.read_bytes().replace(b"\r\n", b"\n"))
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            files.append({"path": path.relative_to(output).as_posix(), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema_version": 1, "snapshot_version": "v1", "created_date": "2026-11-07",
        "model_code_commit": "f9d682c4438d88bff83ea23631ea886f182431e7",
        "export_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_license": "CC-BY-4.0", "files": files,
    }
    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    sums = files + [{"path": "manifest.json", "sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}]
    (output / "SHA256SUMS").write_bytes("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in sums).encode())
    return len(files), numeric_count


if __name__ == "__main__":
    main()
