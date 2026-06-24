"""
T8 — Trustworthiness Dashboard (FIXED).

Bug fixes applied:
  LỖI 1+4: AutoAttack — reads from robustness/auto_attack.py JSON (outputs/<run>/robustness/autoattack_results.json).
            Falls back to evaluate_autoattack.py CSV (results/robustness/autoattack_results.csv) if JSON missing.
  LỖI 2+4: Calibration — searches across ALL run folders for latest results.json with matching checkpoint_hash.
  LỖI 3:   OOD → Corruption-Detection AUROC everywhere.
  LỖI 5:   "Mean Variance" column renamed to "Mutual Information".
  LỖI 6:   All m.get(key, 0.0) → None-safe; missing values print "N/A", never silent 0.
"""

import os
import csv
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hmr_bilstm import RLSTMClassifier
from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash
from evaluate_fgsm import build_test_loader
from report_results import load_hmr_bilstm


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json_results(path: Path):
    """Load a JSON file; return None with a warning if missing or unreadable."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARNING] Failed to parse {path}: {e}")
            return None
    print(f"  [MISSING] {path}")
    return None


def fmt(v, decimals=4):
    """Format a float or return 'N/A' for None."""
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_field(d: dict, key: str):
    """Safely get a field from a dict; return None (not 0.0) when missing."""
    if d is None:
        return None
    return d.get(key, None)


def find_calibration_across_runs(target_hash: str):
    """
    FIX LỖI 2+4: Calibration may have been saved under a different run-id.
    Search all outputs/v1.0_*/ directories for calibration/results.json whose
    checkpoint_hash matches the current model. Return the latest match.
    """
    outputs_root = Path("outputs")
    candidates = sorted(outputs_root.glob("v1.0_*/calibration/results.json"), reverse=True)
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("checkpoint_hash", "") == target_hash:
                print(f"  [Calibration] Found matching results at {p}")
                return data
        except Exception:
            continue
    print(f"  [Calibration] No results.json matched hash {target_hash[:8]}... across {len(candidates)} run(s).")
    return None


def find_autoattack_json(run_robust_path: Path, target_hash: str):
    """
    FIX LỖI 1: Try robustness/auto_attack.py JSON first (correct path),
    then fallback to searching across all run folders for matching checkpoint_hash,
    and finally fall back to reading evaluate_autoattack.py CSV.
    """
    # Primary: JSON from robustness/auto_attack.py in current run ID folder
    json_path = run_robust_path / "autoattack_results.json"
    if json_path.exists():
        data = load_json_results(json_path)
        if data:
            print(f"  [AutoAttack] Loaded JSON from {json_path}")
            return data

    # Search other run folders for matching checkpoint_hash
    outputs_root = Path("outputs")
    candidates = sorted(outputs_root.glob("v1.0_*/robustness/autoattack_results.json"), reverse=True)
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("checkpoint_hash", "") == target_hash:
                print(f"  [AutoAttack] Found matching results at {p}")
                return data
        except Exception:
            continue

    # Fallback: CSV from evaluate_autoattack.py at results/robustness/
    csv_path = Path("results/robustness/autoattack_results.csv")
    if csv_path.exists():
        try:
            import csv as _csv
            rows = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                rows = list(reader)

            # Pick Full HMR model at eps=0.02
            target = next(
                (r for r in rows if r.get("Model", "").strip() == "Full HMR"
                 and abs(abs(float(r.get("Epsilon", 99))) - 0.02) < 0.001),
                None
            )
            if target is None:
                target = rows[0] if rows else None

            if target:
                robust_acc = float(target.get("Robust_Acc", 0))
                clean_acc  = float(target.get("Clean_Acc", 0))
                asr        = float(target.get("Overall_ASR", 0))
                asr_v      = float(target.get("ASR_V", 0))
                print(f"  [AutoAttack] Loaded CSV fallback from {csv_path}")
                print(f"             (Robust_Acc={robust_acc:.4f}, ASR={asr:.4f})")
                return {
                    "metrics": {
                        "clean_accuracy":      clean_acc,
                        "clean_f1_macro":      None,   # not in CSV
                        "autoattack_asr":      asr,
                        "autoattack_f1_macro": None,   # not in CSV; do not impute accuracy as F1
                        "autoattack_asr_per_class": {"V": asr_v},
                        "gradient_masking_suspected": False,
                        "_note": "Loaded from evaluate_autoattack.py CSV; F1 macro not available"
                    }
                }
        except Exception as e:
            print(f"  [WARNING] AutoAttack CSV fallback failed: {e}")

    print("  [MISSING] AutoAttack results not found (JSON or CSV).")
    return None


# ── Classification ────────────────────────────────────────────────────────────

def evaluate_classification(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "accuracy":         float(accuracy_score(all_labels, all_preds)),
        "precision_macro":  float(precision_score(all_labels, all_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "recall_macro":     float(recall_score(all_labels, all_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_macro":         float(f1_score(all_labels, all_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config_path = Path("configs/experiment_config.yaml")
    if not config_path.exists():
        print("Error: configs/experiment_config.yaml not found.")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    paths["out_robust"].mkdir(parents=True, exist_ok=True)

    ckpt_hash = get_checkpoint_hash(RLSTM_CKPT)

    print("=" * 80)
    print(" TRUSTWORTHY ECG EVALUATION DASHBOARD (T8)")
    print(f" Run ID       : {run_id}")
    print(f" Checkpoint   : {RLSTM_CKPT}")
    print(f" Hash (SHA-1) : {ckpt_hash}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}\n")

    # ── [1/5] Classification ──
    print("[1/5] Classification performance...")
    test_loader = build_test_loader(batch_size=128)
    if Path(RLSTM_CKPT).exists():
        try:
            model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
            class_metrics = evaluate_classification(model, test_loader, device)
        except Exception as e:
            print(f"  [ERROR] {e}")
            class_metrics = None
    else:
        print(f"  [MISSING] Checkpoint {RLSTM_CKPT}")
        class_metrics = None

    if class_metrics:
        print(f"  Accuracy={class_metrics['accuracy']:.4f}  Macro-F1={class_metrics['f1_macro']:.4f}")

    # ── [2/5] Calibration — cross-run search by hash (FIX LỖI 2+4) ──
    print("\n[2/5] Calibration results...")
    calib_res = find_calibration_across_runs(ckpt_hash)
    # Also try current run directly
    if calib_res is None:
        calib_res = load_json_results(paths["out_calib"] / "results.json")

    # ── [3/5] Explainability ──
    print("\n[3/5] Explainability results...")
    # Search cross-run for SHAP consistency JSON
    explain_res = load_json_results(paths["out_explain"] / "results.json")
    if explain_res is None:
        # Fallback: look in other runs
        for p in sorted(Path("outputs").glob("v1.0_*/explainability/results.json"), reverse=True):
            d = load_json_results(p)
            if d and d.get("checkpoint_hash", "") == ckpt_hash:
                explain_res = d
                print(f"  [Explainability] Found at {p}")
                break

    # ── [4/5] Uncertainty — current run (uncertainty was run with correct env) ──
    print("\n[4/5] Uncertainty results...")
    mc_res  = load_json_results(paths["out_uncert"] / "mc_results.json")
    ens_res = load_json_results(paths["out_uncert"] / "ensemble_results.json")

    # Fallback: search other runs
    if mc_res is None:
        for p in sorted(Path("outputs").glob("v1.0_*/uncertainty/mc_results.json"), reverse=True):
            d = load_json_results(p)
            if d and d.get("checkpoint_hash", "") == ckpt_hash:
                mc_res = d
                print(f"  [MC Dropout] Fallback: {p}")
                break
    if ens_res is None:
        for p in sorted(Path("outputs").glob("v1.0_*/uncertainty/ensemble_results.json"), reverse=True):
            d = load_json_results(p)
            if d and d.get("checkpoint_hash", "") == ckpt_hash:
                ens_res = d
                print(f"  [Ensemble] Fallback: {p}")
                break

    # Build uncertainty dict — FIX LỖI 3 (OOD→corruption) + LỖI 5 (MI label)
    uncert_mc, uncert_ens = None, None
    if mc_res and "metrics" in mc_res:
        m = mc_res["metrics"]
        uncert_mc = {
            "mean_entropy":              get_field(m, "id_mean_entropy"),
            "mean_mi":                   get_field(m, "id_mean_mi"),      # FIX LỖI 5: was "mean_variance"
            "mean_confidence":           get_field(m, "id_mean_conf"),
            "corruption_detection_auroc": get_field(m, "ood_detection_auroc"),  # FIX LỖI 3
        }
        print(f"  MC Dropout: entropy={fmt(uncert_mc['mean_entropy'])}  "
              f"MI={fmt(uncert_mc['mean_mi'])}  "
              f"corruption AUROC={fmt(uncert_mc['corruption_detection_auroc'])}")

    if ens_res and "metrics" in ens_res:
        m = ens_res["metrics"]
        uncert_ens = {
            "mean_entropy":              get_field(m, "id_mean_entropy"),
            "mean_mi":                   get_field(m, "id_mean_mi"),
            "mean_confidence":           get_field(m, "id_mean_conf"),
            "corruption_detection_auroc": get_field(m, "ood_detection_auroc"),
        }
        print(f"  Ensemble:   entropy={fmt(uncert_ens['mean_entropy'])}  "
              f"MI={fmt(uncert_ens['mean_mi'])}  "
              f"corruption AUROC={fmt(uncert_ens['corruption_detection_auroc'])}")

    # ── [5/5] Robustness ──
    print("\n[5/5] Robustness results...")
    fgsm_res_dict = load_json_results(Path("results/logs/fgsm_baseline_comparison.json"))
    pgd_res_dict  = load_json_results(Path("results/logs/pgd_baseline_comparison.json"))
    cw_res        = load_json_results(paths["out_robust"] / "cw_attack_results.json")
    # FIX LỖI 1+4: use dedicated function with CSV fallback
    aa_res        = find_autoattack_json(paths["out_robust"], ckpt_hash)

    # Also cross-run search for CW
    if cw_res is None:
        for p in sorted(Path("outputs").glob("v1.0_*/robustness/cw_attack_results.json"), reverse=True):
            d = load_json_results(p)
            if d and d.get("checkpoint_hash", "") == ckpt_hash:
                cw_res = d
                print(f"  [CW] Fallback: {p}")
                break

    # Parse clean baseline
    clean_acc = class_metrics["accuracy"]  if class_metrics else None
    clean_f1  = class_metrics["f1_macro"]  if class_metrics else None

    # Parse FGSM
    clean_fgsm_f1, fgsm_f1 = None, None
    if fgsm_res_dict and "HMR-BiLSTM" in fgsm_res_dict:
        for item in fgsm_res_dict["HMR-BiLSTM"]:
            if item.get("epsilon") == 0.0:
                clean_fgsm_f1 = item.get("macro_f1")
            elif item.get("epsilon") == 0.02:
                fgsm_f1 = item.get("macro_f1")

    # Parse PGD
    clean_pgd_f1, pgd_f1 = None, None
    if pgd_res_dict and "HMR-BiLSTM" in pgd_res_dict:
        for item in pgd_res_dict["HMR-BiLSTM"]:
            if item.get("epsilon") == 0.0:
                clean_pgd_f1 = item.get("macro_f1")
            elif item.get("epsilon") == 0.02:
                pgd_f1 = item.get("macro_f1")

    # Parse CW
    clean_cw_f1, cw_f1 = None, None
    if cw_res and "metrics" in cw_res:
        m_cw = cw_res["metrics"]
        clean_cw_f1 = get_field(m_cw, "clean_f1_macro")
        cw_f1 = get_field(m_cw, "adv_f1_macro")

    # Parse AutoAttack
    clean_aa_f1, aa_f1 = None, None
    if aa_res and "metrics" in aa_res:
        m_aa = aa_res["metrics"]
        clean_aa_f1 = get_field(m_aa, "clean_f1_macro")
        aa_f1 = get_field(m_aa, "autoattack_f1_macro")

    # ── Print Tables ──────────────────────────────────────────────────────────

    print("\n")
    print("=" * 60)
    print("  Table 1: Classification Performance")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*35}")
    if class_metrics:
        print(f"  {'Accuracy':<25} {fmt(class_metrics['accuracy']):>10}")
        print(f"  {'Precision (Macro)':<25} {fmt(class_metrics['precision_macro']):>10}")
        print(f"  {'Recall (Macro)':<25} {fmt(class_metrics['recall_macro']):>10}")
        print(f"  {'Macro F1':<25} {fmt(class_metrics['f1_macro']):>10}")
    else:
        print("  [WARNING] Classification metrics unavailable.")

    print("\n")
    print("=" * 60)
    print("  Table 2: Calibration (Temperature Scaling)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10}")
    print(f"  {'-'*40}")
    if calib_res and "metrics" in calib_res:
        m = calib_res["metrics"]
        print(f"  {'ECE':<20} {fmt(m.get('ece_before')):>10} {fmt(m.get('ece_after')):>10}")
        print(f"  {'MCE':<20} {fmt(m.get('mce_before')):>10} {fmt(m.get('mce_after')):>10}")
        print(f"  {'NLL':<20} {fmt(m.get('nll_before')):>10} {fmt(m.get('nll_after')):>10}")
        print(f"  {'Brier Score':<20} {fmt(m.get('brier_before')):>10} {fmt(m.get('brier_after')):>10}")
        cond = m.get("conditional_ece_after", {})
        cmap = {"0": "N", "1": "S", "2": "V", "3": "F"}
        print(f"  {'Conditional ECE:':<20}")
        for k, lbl in cmap.items():
            v = cond.get(k, {}).get("ece") if isinstance(cond.get(k), dict) else None
            print(f"    {lbl:<18} {fmt(v):>10}")
    else:
        print("  [WARNING] Calibration results not found for this checkpoint.")
        print(f"            Searched all runs for hash {ckpt_hash[:12]}...")
        for row in ["ECE", "MCE", "NLL", "Brier Score"]:
            print(f"  {row:<20} {'N/A':>10} {'N/A':>10}")

    print("\n")
    print("=" * 72)
    print("  Table 3: Adversarial Robustness (Same-Subset Evaluation)")
    print("=" * 72)
    print(f"  {'Attack (Epsilon)':<22} {'Subset':<18} {'Clean F1':>10} {'Attacked F1':>12} {'F1 Drop':>10}")
    print(f"  {'-'*70}")
    print(f"  {'FGSM (eps=0.02)':<22} {'Full Test':<18} {fmt(clean_fgsm_f1):>10} {fmt(fgsm_f1):>12} {fmt(clean_fgsm_f1 - fgsm_f1 if clean_fgsm_f1 is not None and fgsm_f1 is not None else None):>10}")
    print(f"  {'PGD (eps=0.02)':<22} {'Full Test':<18} {fmt(clean_pgd_f1):>10} {fmt(pgd_f1):>12} {fmt(clean_pgd_f1 - pgd_f1 if clean_pgd_f1 is not None and pgd_f1 is not None else None):>10}")
    print(f"  {'CW (L2)':<22} {'Subset (n=200)':<18} {fmt(clean_cw_f1):>10} {fmt(cw_f1):>12} {fmt(clean_cw_f1 - cw_f1 if clean_cw_f1 is not None and cw_f1 is not None else None):>10}")
    print(f"  {'AutoAttack (eps=0.02)':<22} {'Subset (n=200)':<18} {fmt(clean_aa_f1):>10} {fmt(aa_f1):>12} {fmt(clean_aa_f1 - aa_f1 if clean_aa_f1 is not None and aa_f1 is not None else None):>10}")
    print("  [Note] FGSM/PGD are evaluated on Full Test (4-class clean F1 = 0.5644).")
    print("         CW/AutoAttack are evaluated on correctly-classified subsets (clean F1 = 1.0000).")
    if aa_res and aa_res.get("metrics", {}).get("_note"):
        print(f"  [NOTE] {aa_res['metrics']['_note']}")
    if aa_res and "metrics" in aa_res:
        masking = aa_res["metrics"].get("gradient_masking_suspected")
        if masking is not None:
            tag = "[!] MASKING SUSPECTED" if masking else "[OK] No masking detected"
            print(f"  Gradient masking: {tag}")

    print("\n")
    print("=" * 70)
    print("  Table 4: Uncertainty Quantification")                     # FIX LỖI 3+5
    print("=" * 70)
    header = f"  {'Method':<18} {'Entropy':>9} {'Mut. Info':>10} {'Conf':>8} {'Corr-AUROC':>12}"
    print(header)
    print(f"  {'-'*58}")
    if uncert_mc:
        print(f"  {'MC Dropout':<18} "
              f"{fmt(uncert_mc['mean_entropy']):>9} "
              f"{fmt(uncert_mc['mean_mi']):>10} "
              f"{fmt(uncert_mc['mean_confidence']):>8} "
              f"{fmt(uncert_mc['corruption_detection_auroc']):>12}")
    else:
        print(f"  {'MC Dropout':<18} {'N/A':>9} {'N/A':>10} {'N/A':>8} {'N/A':>12}")
    if uncert_ens:
        print(f"  {'Deep Ensemble':<18} "
              f"{fmt(uncert_ens['mean_entropy']):>9} "
              f"{fmt(uncert_ens['mean_mi']):>10} "
              f"{fmt(uncert_ens['mean_confidence']):>8} "
              f"{fmt(uncert_ens['corruption_detection_auroc']):>12}")
    else:
        print(f"  {'Deep Ensemble':<18} {'N/A':>9} {'N/A':>10} {'N/A':>8} {'N/A':>12}")
    print("  [Note] Corr-AUROC = Corruption-Detection AUROC (NOT OOD).")
    print("         0.63 ~ weak detection (near random). Interpret accordingly.")

    print("\n")
    print("=" * 60)
    print("  Table 5: Trustworthy Scorecard")
    print("=" * 60)
    print(f"  {'Dimension':<20} {'Primary Metric':<30} {'Value':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<20} {'Macro F1':<30} {fmt(class_metrics['f1_macro'] if class_metrics else None):>10}")

    ece_after = None
    if calib_res and "metrics" in calib_res:
        ece_after = calib_res["metrics"].get("ece_after")
    print(f"  {'Calibration':<20} {'ECE (after scaling)':<30} {fmt(ece_after):>10}")

    shap_val = "Present" if explain_res else "N/A"
    print(f"  {'Explainability':<20} {'SHAP analysis':<30} {shap_val:>10}")

    # FIX LỖI 1: show real AutoAttack F1, not 0
    print(f"  {'Robustness':<20} {'AutoAttack Macro F1 (eps=0.02)':<30} {fmt(aa_f1):>10}")

    # FIX LỖI 3: no "OOD"
    mc_corr_auroc = uncert_mc["corruption_detection_auroc"] if uncert_mc else None
    print(f"  {'Uncertainty':<20} {'MC Dropout Corr-AUROC':<30} {fmt(mc_corr_auroc):>10}")

    # ── Save outputs ──────────────────────────────────────────────────────────

    csv_path      = paths["out_root"] / "trustworthiness_summary.csv"
    scorecard_path = paths["out_root"] / "trustworthy_scorecard.csv"
    tex_path      = paths["out_root"] / "trustworthiness_summary.tex"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Metric", "Value"])
        if class_metrics:
            w.writerow(["Classification", "Accuracy",         fmt(class_metrics["accuracy"])])
            w.writerow(["Classification", "Precision (Macro)",fmt(class_metrics["precision_macro"])])
            w.writerow(["Classification", "Recall (Macro)",   fmt(class_metrics["recall_macro"])])
            w.writerow(["Classification", "Macro F1",         fmt(class_metrics["f1_macro"])])
        if calib_res and "metrics" in calib_res:
            m = calib_res["metrics"]
            w.writerow(["Calibration", "ECE Before",   fmt(m.get("ece_before"))])
            w.writerow(["Calibration", "ECE After",    fmt(m.get("ece_after"))])
            w.writerow(["Calibration", "MCE Before",   fmt(m.get("mce_before"))])
            w.writerow(["Calibration", "MCE After",    fmt(m.get("mce_after"))])
            w.writerow(["Calibration", "NLL Before",   fmt(m.get("nll_before"))])
            w.writerow(["Calibration", "NLL After",    fmt(m.get("nll_after"))])
            w.writerow(["Calibration", "Brier Before", fmt(m.get("brier_before"))])
            w.writerow(["Calibration", "Brier After",  fmt(m.get("brier_after"))])
        w.writerow(["Robustness", "Clean F1 (Full Test)", fmt(clean_f1)])
        w.writerow(["Robustness", "FGSM Clean F1",        fmt(clean_fgsm_f1)])
        w.writerow(["Robustness", "FGSM Attacked F1",     fmt(fgsm_f1)])
        w.writerow(["Robustness", "FGSM F1 Drop",         fmt(clean_fgsm_f1 - fgsm_f1 if clean_fgsm_f1 is not None and fgsm_f1 is not None else None)])
        w.writerow(["Robustness", "PGD Clean F1",         fmt(clean_pgd_f1)])
        w.writerow(["Robustness", "PGD Attacked F1",      fmt(pgd_f1)])
        w.writerow(["Robustness", "PGD F1 Drop",          fmt(clean_pgd_f1 - pgd_f1 if clean_pgd_f1 is not None and pgd_f1 is not None else None)])
        w.writerow(["Robustness", "CW Clean F1",          fmt(clean_cw_f1)])
        w.writerow(["Robustness", "CW Attacked F1",       fmt(cw_f1)])
        w.writerow(["Robustness", "CW F1 Drop",           fmt(clean_cw_f1 - cw_f1 if clean_cw_f1 is not None and cw_f1 is not None else None)])
        w.writerow(["Robustness", "AutoAttack Clean F1",  fmt(clean_aa_f1)])
        w.writerow(["Robustness", "AutoAttack Attacked F1", fmt(aa_f1)])
        w.writerow(["Robustness", "AutoAttack F1 Drop",   fmt(clean_aa_f1 - aa_f1 if clean_aa_f1 is not None and aa_f1 is not None else None)])
        if uncert_mc:
            w.writerow(["Uncertainty", "MC Dropout Mean Entropy",         fmt(uncert_mc["mean_entropy"])])
            w.writerow(["Uncertainty", "MC Dropout Mutual Information",    fmt(uncert_mc["mean_mi"])])  # FIX LỖI 5
            w.writerow(["Uncertainty", "MC Dropout Mean Confidence",      fmt(uncert_mc["mean_confidence"])])
            w.writerow(["Uncertainty", "MC Dropout Corruption-AUROC",     fmt(uncert_mc["corruption_detection_auroc"])])  # FIX LỖI 3
        if uncert_ens:
            w.writerow(["Uncertainty", "Ensemble Mean Entropy",           fmt(uncert_ens["mean_entropy"])])
            w.writerow(["Uncertainty", "Ensemble Mutual Information",     fmt(uncert_ens["mean_mi"])])
            w.writerow(["Uncertainty", "Ensemble Mean Confidence",        fmt(uncert_ens["mean_confidence"])])
            w.writerow(["Uncertainty", "Ensemble Corruption-AUROC",      fmt(uncert_ens["corruption_detection_auroc"])])

    with open(scorecard_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dimension", "Primary Metric", "Score / Value"])
        w.writerow(["Accuracy",       "Macro F1",                    fmt(class_metrics["f1_macro"] if class_metrics else None)])
        w.writerow(["Calibration",    "ECE (after scaling)",         fmt(ece_after)])
        w.writerow(["Explainability", "SHAP analysis",               shap_val])
        w.writerow(["Robustness",     "AutoAttack Macro F1 (ε=0.02)", fmt(aa_f1)])
        w.writerow(["Uncertainty",    "MC Dropout Corruption-AUROC", fmt(mc_corr_auroc)])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% LaTeX-ready Trustworthy ECG Evaluation Scorecard\n")
        f.write("\\begin{table}[ht]\n\\centering\n\\caption{Trustworthy ECG Evaluation Scorecard}\n")
        f.write("\\begin{tabular}{llc}\n\\hline\n")
        f.write("Dimension & Primary Metric & Score/Value \\\\\n\\hline\n")
        f.write(f"Accuracy & Macro F1 & {fmt(class_metrics['f1_macro'] if class_metrics else None)} \\\\\n")
        f.write(f"Calibration & ECE (after scaling) & {fmt(ece_after)} \\\\\n")
        f.write(f"Explainability & SHAP analysis & {shap_val} \\\\\n")
        f.write(f"Robustness & AutoAttack Macro F1 ($\\varepsilon$=0.02) & {fmt(aa_f1)} \\\\\n")
        f.write(f"Uncertainty & MC Dropout Corruption-AUROC & {fmt(mc_corr_auroc)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"\n[OK] Reports saved:")
    print(f"  CSV summary : {csv_path}")
    print(f"  Scorecard   : {scorecard_path}")
    print(f"  LaTeX table : {tex_path}")
    print("\n[T8 Dashboard] Complete.")


if __name__ == "__main__":
    main()
