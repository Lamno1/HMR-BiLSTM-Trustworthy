"""P1 gate - paired R2 (vrce) vs flat analysis on the LOPO fold results.

Primary analysis (not a pooled i.i.d. summary):
  1. average the two seeds within each fold -> 5 fold-level values
  2. paired R2 - flat delta per (fold, seed) AND per fold-mean
  3. per fold: F/S support and TP/FP/FN (from *_preds.npz te_y / te_pred)
  4. gate verdict for S; F is flagged referral-only unless the paired delta
     vs flat is clearly positive.

Run: python -u -m validation.analyze_p1_gate
Out: results/p1_gate/report.json  (+ printed summary)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

R2_DIR, FLAT_DIR = Path("results/mp_lopo"), Path("results/lopo_cv")
OUT = Path("results/p1_gate")
S, F = 1, 3


def load(result_dir: Path, tag: str) -> dict:
    out = {}
    for jp in result_dir.glob(f"fold*_seed*_{tag}.json"):
        n = jp.name
        fold = int(re.search(r"fold(\d)", n).group(1))
        seed = int(re.search(r"seed(\d+)", n).group(1))
        r = json.loads(jp.read_text())
        rec = {"mit": r["best_epoch_model"].get("mitbih"),
               "all": r["best_epoch_model"].get("all")}
        pp = jp.with_name(jp.name.replace(".json", "_preds.npz"))
        if pp.exists():
            d = np.load(pp, allow_pickle=True)
            rec["te_y"], rec["te_pred"] = d["te_y"], d["te_pred"]
        out[(fold, seed)] = rec
    return out


def confusion(te_y, te_pred, c) -> dict:
    return {"support": int((te_y == c).sum()),
            "tp": int(((te_pred == c) & (te_y == c)).sum()),
            "fp": int(((te_pred == c) & (te_y != c)).sum()),
            "fn": int(((te_pred != c) & (te_y == c)).sum())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r2, flat = load(R2_DIR, "vrce"), load(FLAT_DIR, "flat")
    folds = sorted({f for f, _ in r2})
    seeds = sorted({s for _, s in r2})
    rep = {"folds": folds, "seeds": seeds, "per_fold": [], "paired_delta": [], "gate": {}}
    fold_means = {"r2": {}, "flat": {}}

    for fl in folds:
        row = {"fold": fl, "confusion": {}}
        for tag, src in (("r2", r2), ("flat", flat)):
            vals = {}
            for b in ("mit", "all"):
                vals[b] = {}
                for name in ("F", "S"):
                    xs = [src[(fl, s)][b][name] for s in seeds
                          if (fl, s) in src and src[(fl, s)][b]]
                    vals[b][name] = {
                        "per_seed": [round(x, 4) for x in xs],
                        "mean": round(float(np.mean(xs)), 4) if xs else None,
                        "seed_spread": round(float(max(xs) - min(xs)), 4) if len(xs) > 1 else None}
            row[tag] = vals
            fold_means[tag][fl] = {b: {k: vals[b][k]["mean"] for k in ("F", "S")}
                                   for b in ("mit", "all")}
            for s in seeds:
                k = (fl, s)
                if k in src and "te_y" in src[k]:
                    row["confusion"][f"{tag}_seed{s}"] = {
                        "F": confusion(src[k]["te_y"], src[k]["te_pred"], F),
                        "S": confusion(src[k]["te_y"], src[k]["te_pred"], S)}
        rep["per_fold"].append(row)

    for fl in folds:
        for s in seeds:
            if (fl, s) in r2 and (fl, s) in flat and r2[(fl, s)]["mit"] and flat[(fl, s)]["mit"]:
                a, bb = r2[(fl, s)], flat[(fl, s)]
                rep["paired_delta"].append({
                    "fold": fl, "seed": s,
                    "dF_mit": round(a["mit"]["F"] - bb["mit"]["F"], 4),
                    "dS_mit": round(a["mit"]["S"] - bb["mit"]["S"], 4),
                    "dF_all": round(a["all"]["F"] - bb["all"]["F"], 4),
                    "dS_all": round(a["all"]["S"] - bb["all"]["S"], 4)})

    dS = [fold_means["r2"][fl]["mit"]["S"] - fold_means["flat"][fl]["mit"]["S"]
          for fl in folds if fold_means["flat"].get(fl, {}).get("mit", {}).get("S") is not None]
    dF = [fold_means["r2"][fl]["mit"]["F"] - fold_means["flat"][fl]["mit"]["F"]
          for fl in folds if fold_means["flat"].get(fl, {}).get("mit", {}).get("F") is not None]

    def sp(src):
        vs = [max(xs) - min(xs) for fl in folds
              for xs in [[src[(fl, s)]["mit"]["S"] for s in seeds
                          if (fl, s) in src and src[(fl, s)]["mit"]]] if len(xs) > 1]
        return round(float(np.mean(vs)), 4) if vs else None

    if dS:
        r2sp, flatsp = sp(r2), sp(flat)
        # necessary AND sufficient: mean gain not trivial, gain shows on >=3
        # folds (not one-fold-dominated, not merely 'not down'), and seed
        # spread not worse than flat.
        wins_s = bool(np.mean(dS) >= 0.02 and sum(x > 0.01 for x in dS) >= 3
                      and sum(x >= -1e-9 for x in dS) >= 4
                      and (r2sp is None or flatsp is None or r2sp <= flatsp + 1e-9))
        f_claim = bool(dF and np.mean(dF) > 0.03 and sum(x > 0.01 for x in dF) >= 3)
        rep["gate"] = {
            "n_flat_folds": len({f for f, _ in flat}), "n_folds": len(folds),
            "dS_fold_mean": round(float(np.mean(dS)), 4),
            "dS_folds_not_down": int(sum(x >= -1e-9 for x in dS)),
            "dF_fold_mean": round(float(np.mean(dF)), 4) if dF else None,
            "r2_S_seed_spread": r2sp, "flat_S_seed_spread": flatsp,
            "R2_wins_S": wins_s, "F_full_coverage_claim": f_claim,
            "verdict": ("flat incomplete - rerun when 10 flat folds exist"
                        if len({f for f, _ in flat}) < len(folds) else
                        ("R2 wins S. " if wins_s else "R2 does NOT clearly beat flat on S. ")
                        + ("F: full-coverage improvement supported."
                           if f_claim else "F: referral/frontier only."))}
    else:
        rep["gate"] = {"verdict": "flat baseline not started"}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2))
    print("=== P1 GATE ===")
    print(f"folds={folds} seeds={seeds}  flat folds: {len({f for f, _ in flat})}/{len(folds)}")
    for fl in folds:
        fs = fold_means["flat"].get(fl, {}).get("mit", {})
        print(f" fold{fl} mit  F: R2 {fold_means['r2'][fl]['mit']['F']}  flat {fs.get('F')}"
              f"   S: R2 {fold_means['r2'][fl]['mit']['S']}  flat {fs.get('S')}")
    print(json.dumps(rep["gate"], indent=2))
    print(f"[OK] {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
