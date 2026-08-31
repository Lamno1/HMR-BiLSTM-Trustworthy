"""P0 artifact inspection — runs the smoke checklist against saved *_preds.npz.

Usage:
    python -m validation.inspect_p0_artifacts results/p0_smoke2_mp vrce
    python -m validation.inspect_p0_artifacts results/p0_smoke2_flat flat
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def check_one(out_dir: Path, tag: str) -> bool:
    a42 = out_dir / f"fold0_seed42_{tag}_preds.npz"
    a123 = out_dir / f"fold0_seed123_{tag}_preds.npz"
    ok = True

    for p in (a42, a123):
        if not p.exists():
            print(f"  MISSING {p.name}")
            return False

    d = load(a42)
    d2 = load(a123)

    n_te = d["te_logits"].shape[0]
    checks = [
        ("te_logits shape (n,5)", d["te_logits"].ndim == 2 and d["te_logits"].shape[1] == 5),
        ("va_logits present", "va_logits" in d and d["va_logits"].shape[1] == 5),
        ("tuning_logits present", "tuning_logits" in d),
        ("conformal_cal_logits present", "conformal_cal_logits" in d),
        ("te_logits dtype float32 (not fp16)", d["te_logits"].dtype == np.float32),
        ("no NaN in te_logits", not np.isnan(d["te_logits"].astype(np.float32)).any()),
        ("tuning == inner_val (alias)", np.allclose(d["tuning_logits"], d["va_logits"])),
        ("conformal_cal has no F (label 3)", 3 not in set(np.unique(d["conformal_cal_y"]).tolist())),
        ("conformal_cal has S (label 1)", 1 in set(np.unique(d["conformal_cal_y"]).tolist())),
    ]

    te_rec = set(np.unique(d["te_record"]).tolist())
    va_rec = set(np.unique(d["va_record"]).tolist())
    ca_rec = set(np.unique(d["conformal_cal_record"]).tolist())
    checks.append(("conformal_cal records disjoint from test", not (ca_rec & te_rec)))
    checks.append(("conformal_cal records disjoint from val", not (ca_rec & va_rec)))

    try:
        meta = json.loads(str(d["meta_json"]))
        checks.append(("meta_json parses", True))
        checks.append(("meta tuning_split == inner_val", meta.get("tuning_split") == "inner_val"))
    except Exception as e:  # noqa: BLE001
        checks.append((f"meta_json parses ({e})", False))

    checks.append(("seed42 te_logits != seed123 te_logits",
                   not np.allclose(d["te_logits"], d2["te_logits"])))

    for name, passed in checks:
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    ckpt = out_dir / "checkpoints"
    for seed in (42, 123):
        hits = list(ckpt.glob(f"*fold0_seed{seed}_state.pt")) if ckpt.exists() else []
        passed = len(hits) == 1
        print(f"  [{'OK ' if passed else 'FAIL'}] one checkpoint for seed {seed}: {[h.name for h in hits]}")
        ok = ok and passed

    print(f"  n_test={n_te}  conformal_cal_S_beats={int((d['conformal_cal_y'] == 1).sum())}")
    return ok


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    tag = sys.argv[2]
    print(f"== {out_dir}  tag={tag} ==")
    result = check_one(out_dir, tag)
    print("PASS" if result else "FAIL")
    sys.exit(0 if result else 1)
