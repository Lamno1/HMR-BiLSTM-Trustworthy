"""Post-run sanity gate for a LOPO result directory.

Flags abnormal runs so the plan driver stops instead of feeding garbage to P2.

Usage:
    python -m validation.check_lopo_health results/mp_lopo vrce 5 2
    python -m validation.check_lopo_health results/lopo_cv flat 5 2

Exit 0 = healthy, exit 1 = abnormal (driver should stop and report).
"""
import json
import sys
from pathlib import Path

import numpy as np

# N and V are "solid" for this project; a collapse below these floors on any
# fold means a training/pipeline bug, not a hard-class result.
N_FLOOR = 0.80
V_FLOOR = 0.60


def main() -> int:
    out_dir = Path(sys.argv[1])
    tag = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    files = sorted(out_dir.glob(f"fold*_seed*_{tag}.json"))
    expected = k * n_seeds
    problems = []
    print(f"== health: {out_dir} tag={tag} ==")
    print(f"  found {len(files)}/{expected} fold-seed result files")
    if len(files) < expected:
        problems.append(f"only {len(files)}/{expected} runs completed")

    by_fold = {}
    for p in files:
        r = json.loads(p.read_text())
        blk = r.get("best_epoch_model", {}).get("all")
        if not blk:
            problems.append(f"{p.name}: no best_epoch_model.all")
            continue
        for m in ("f1_macro", "N", "S", "V", "F"):
            v = blk.get(m)
            if v is None or (isinstance(v, float) and (v != v)):
                problems.append(f"{p.name}: {m} missing/NaN")
        if blk.get("N", 1) < N_FLOOR:
            problems.append(f"{p.name}: N={blk.get('N'):.3f} < {N_FLOOR} (bug?)")
        if blk.get("V", 1) < V_FLOOR:
            problems.append(f"{p.name}: V={blk.get('V'):.3f} < {V_FLOOR} (bug?)")
        fold = r.get("fold")
        by_fold.setdefault(fold, []).append((p.name, blk))
        print(f"  {p.name}: macro={blk['f1_macro']:.3f} "
              f"N={blk['N']:.3f} S={blk['S']:.3f} V={blk['V']:.3f} F={blk['F']:.3f}")

    # seed-independence guard: two seeds of the same fold must not be identical
    for fold, runs in by_fold.items():
        if len(runs) >= 2:
            (n1, a), (n2, b) = runs[0], runs[1]
            if all(abs(a[m] - b[m]) < 1e-9 for m in ("f1_macro", "N", "S", "V", "F")):
                problems.append(f"fold {fold}: seeds {n1} and {n2} are identical "
                                f"-> seed-resume bug regressed")

    if problems:
        print("\n  ABNORMAL:")
        for x in problems:
            print(f"   - {x}")
        return 1
    if by_fold:
        fs = [b["F"] for runs in by_fold.values() for _, b in runs]
        ss = [b["S"] for runs in by_fold.values() for _, b in runs]
        print(f"\n  OK. F: mean {np.mean(fs):.3f} std {np.std(fs):.3f} "
              f"min {np.min(fs):.3f} max {np.max(fs):.3f}")
        print(f"      S: mean {np.mean(ss):.3f} std {np.std(ss):.3f} "
              f"min {np.min(ss):.3f} max {np.max(ss):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
