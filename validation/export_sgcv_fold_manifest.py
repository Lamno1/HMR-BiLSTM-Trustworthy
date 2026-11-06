"""Export the SG-CV fold manifest: records and AAMI N/S/V/F/Q counts of every partition.

Reads the locked protocol manifest (results/controlled_mit_subject_cv/protocol_manifest.json)
and writes one row per fold x partition (model-train before the N control, inner validation,
outer test) with its subject groups, records and class counts, plus the partition of every
record in every fold. Also checks that each group is in exactly one outer-test fold and that
the three partitions of a fold are disjoint.

    python -m validation.export_sgcv_fold_manifest
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "results" / "controlled_mit_subject_cv" / "protocol_manifest.json"
OUT_CSV = ROOT / "results" / "tables" / "sgcv_fold_manifest.csv"
OUT_REC = ROOT / "results" / "tables" / "sgcv_record_partition.csv"
CLASSES = ["N", "S", "V", "F", "Q"]
PARTS = [("train", "train_groups", "train_counts_before_n_control"),
         ("inner_val", "val_groups", "val_counts"),
         ("outer_test", "test_groups", "test_counts")]


def main():
    m = json.loads(MAN.read_text(encoding="utf-8"))
    groups = m["group_records"]
    def recs(g):
        values = groups[str(g)] if isinstance(groups, dict) else groups[g]
        return [r.split("/")[-1] for r in values]

    rows, where = [], {}
    test_count = {}
    for f in m["folds"]:
        seen = set()
        for part, gkey, ckey in PARTS:
            gs = f[gkey]
            assert not seen & set(gs), f"fold {f['fold']}: overlapping partitions"
            seen |= set(gs)
            rr = sorted(r for g in gs for r in recs(g))
            for r in rr:
                where.setdefault(r, {})[f["fold"]] = part
            if part == "outer_test":
                for g in gs:
                    test_count[g] = test_count.get(g, 0) + 1
            rows.append({"fold": f["fold"], "partition": part, "n_groups": len(gs),
                         "n_records": len(rr), "records": " ".join(rr),
                         **dict(zip(CLASSES, f[ckey]))})
        if "n_target" in f:
            rows.append({"fold": f["fold"], "partition": "train_after_n_control",
                         "n_groups": len(f["train_groups"]), "n_records": "", "records": "",
                         **dict(zip(CLASSES, [f["n_target"], *f[PARTS[0][2]][1:]]))})
    n_groups = m["n_subject_groups"]
    assert sorted(test_count) == list(range(n_groups)) and set(test_count.values()) == {1}, \
        "a group is not in exactly one outer-test fold"

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "partition", "n_groups", "n_records", *CLASSES, "records"])
        w.writeheader()
        w.writerows(rows)
    folds = [f["fold"] for f in m["folds"]]
    with open(OUT_REC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["record", *[f"fold{k}" for k in folds]])
        for r in sorted(where):
            w.writerow([r, *[where[r].get(k, "") for k in folds]])
    for r in ("201", "202", "208", "213", "232"):
        print(r, [where[r][k] for k in folds])
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_REC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
