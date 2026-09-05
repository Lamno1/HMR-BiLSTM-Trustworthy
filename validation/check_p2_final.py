"""Integrity and deterministic-reproduction checks for P2-final artifacts."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

REQUIRED = ["manifest.json","continuous_s_risk_coverage.csv","continuous_s_risk_coverage.json",
            "fixed_operating_points.csv","prediction_set_sizes.csv","patient_level_metrics.csv",
            "fold_level_metrics.csv","patient_cluster_uncertainty.json","f_referral_diagnostics.csv",
            "go_criterion.csv","r2_vs_flat_summary.json","fixed_operating_points.json",
            "fold_level_metrics.json","f_referral_diagnostics.json","P2_FINAL_REPORT.md"]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--compare",type=Path)
    a=ap.parse_args(); errors=[]
    for name in REQUIRED:
        if not (a.root/name).is_file(): errors.append(f"missing {name}")
    if errors: raise SystemExit("\n".join(errors))
    for name in [x for x in REQUIRED if x.endswith(".json")]:
        json.loads((a.root/name).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    with (a.root/"continuous_s_risk_coverage.csv").open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    # 2 models * 5 folds * 2 seeds * 2 datasets * 100 alpha, subject to both datasets present.
    keys={(r["dataset"],r["model"],r["fold"],r["seed"],r["alpha"]) for r in rows}
    if len(keys)!=len(rows): errors.append("duplicate curve keys")
    alphas=sorted({float(r["alpha"]) for r in rows})
    if len(alphas)!=100 or alphas[0]!=.005 or alphas[-1]!=.5: errors.append("alpha grid mismatch")
    if a.compare:
        for name in REQUIRED:
            if digest(a.root/name)!=digest(a.compare/name): errors.append(f"non-identical rerun: {name}")
    report={"status":"PASS" if not errors else "FAIL","errors":errors,"n_curve_rows":len(rows),
            "files":{n:digest(a.root/n) for n in REQUIRED}}
    (a.root/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); raise SystemExit(1 if errors else 0)
if __name__=="__main__": main()
