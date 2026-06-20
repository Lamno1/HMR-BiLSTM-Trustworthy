#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gather_final.py — Copy all verified artefacts into outputs/v1.0_FINAL/
COPY ONLY — no training, no evaluation, no deletion of source folders.
"""
import shutil
from pathlib import Path

FINAL = Path("outputs/v1.0_FINAL")

SOURCE_MAP = {
    # [destination_subfolder]: [(src_path, optional_new_filename), ...]
    "explainability": [
        ("outputs/v1.0_20260616_061207/explainability/label_noise_candidates.csv",  None),
        ("outputs/v1.0_20260616_061207/explainability/confusable_samples_all.csv",  None),
        ("outputs/v1.0_20260616_061207/explainability/shap_importance_ranking.csv", None),
        ("outputs/v1.0_20260616_061207/explainability/tracin_top_harmful.json",     None),
        ("outputs/v1.0_20260616_061207/explainability/results.json",                "explainability_results.json"),
    ],
    "explainability/tracin_waveforms": "GLOB:outputs/v1.0_20260616_061207/explainability/tracin_waveforms/*.png",

    "calibration": [
        ("outputs/v1.0_20260611_180710/calibration/temperature.json", None),
        ("outputs/v1.0_20260611_180710/calibration/results.json",     "calibration_results.json"),
    ],

    "figures": [
        # SHAP
        ("outputs/v1.0_20260616_061207/explainability/shap_summary_plot.png",    None),
        ("outputs/v1.0_20260616_061207/explainability/shap_class_F.png",         None),
        ("outputs/v1.0_20260616_061207/explainability/shap_class_N.png",         None),
        ("outputs/v1.0_20260616_061207/explainability/shap_class_S.png",         None),
        ("outputs/v1.0_20260616_061207/explainability/shap_class_V.png",         None),
        ("outputs/v1.0_20260616_061207/explainability/shap_misclassified_F.png", None),
        ("outputs/v1.0_20260616_061207/explainability/shap_misclassified_N.png", None),
        ("outputs/v1.0_20260616_061207/explainability/shap_misclassified_S.png", None),
        ("outputs/v1.0_20260616_061207/explainability/shap_misclassified_V.png", None),
        # Calibration reliability
        ("outputs/v1.0_20260611_180710/calibration/reliability_after.png",        None),
        ("outputs/v1.0_20260611_180710/calibration/reliability_before.png",       None),
        ("outputs/v1.0_20260611_180710/calibration/reliability_before_after.png", None),
        ("outputs/v1.0_20260611_180710/calibration/reliability_per_class.png",    None),
        # Uncertainty
        ("outputs/v1.0_20260616_061207/uncertainty/mc_entropy_distribution.png",        None),
        ("outputs/v1.0_20260616_061207/uncertainty/mc_confidence_calibration.png",      None),
        ("outputs/v1.0_20260616_061207/uncertainty/ensemble_disagreement.png",          None),
        ("outputs/v1.0_20260616_061207/uncertainty/ensemble_entropy_distribution.png",  None),
        ("outputs/v1.0_20260616_061207/uncertainty/ensemble_confidence_calibration.png",None),
        ("outputs/v1.0_20260616_061207/uncertainty/corruption_degradation.png",         None),
        # Robustness
        ("outputs/v1.0_20260616_061207/robustness/autoattack_comparison.png", None),
        ("outputs/v1.0_20260616_061207/robustness/cw_asr_by_class.png",       None),
        ("outputs/v1.0_20260616_061207/robustness/cw_perturbation_norms.png", None),
        # Core results/figures
        ("results/figures/ablation_clean_vs_adv.png",              None),
        ("results/figures/ablation_f1_drop.png",                   None),
        ("results/figures/fgsm_baseline_f1_vs_epsilon.png",        None),
        ("results/figures/fgsm_comparison_macro_f1.png",           None),
        ("results/figures/fgsm_ecg_example_epsilon_0.020.png",     None),
        ("results/figures/fgsm_fused.png",                         None),
        ("results/figures/fgsm_per_class_recall_eps0.02.png",      None),
        ("results/figures/fgsm_per_class_recall_eps0.05.png",      None),
        ("results/figures/pgd_baseline_f1_vs_epsilon.png",         None),
        ("results/figures/pgd_per_class_recall_eps0.02.png",       None),
        ("results/figures/pgd_vs_fgsm_comparison.png",             None),
        ("results/figures/confusion_matrix.png",                   None),
        ("results/figures/roc_curve.png",                          None),
        ("results/figures/reliability_diagram.png",                None),
        ("results/figures/robustness_noise_all.png",               None),
        ("results/figures/robustness_summary.png",                 None),
        ("results/figures/gate_trajectories.png",                  None),
        ("results/figures/trustworthy_ai_summary.png",             None),
        ("results/figures/comparison_bars.png",                    None),
        ("results/figures/final_results_table.png",                None),
        ("results/figures/case_fusion.png",                        None),
        ("results/figures/case_normal.png",                        None),
        ("results/figures/case_supraventricular.png",              None),
        ("results/figures/case_ventricular.png",                   None),
        ("results/figures/case_comparison_N_vs_F.png",             None),
        ("results/figures/case_comparison_N_vs_S.png",             None),
        ("results/figures/case_comparison_N_vs_V.png",             None),
    ],

    "logs": [
        ("results/logs/baseline_results.json",            None),
        ("results/logs/fgsm_baseline_comparison.json",   None),
        ("results/logs/fgsm_comparison_results.json",    None),
        ("results/logs/pgd_baseline_comparison.json",    None),
        ("outputs/v1.0_20260616_061207/uncertainty/corruption_sweep_results.json", None),
    ],

    "tables": [
        ("results/tables/ablation_table_final.csv",        None),
        ("results/tables/ablation_table_final.tex",        None),
        ("results/tables/baseline_full_comparison.csv",    None),
        ("results/tables/baseline_full_comparison.tex",    None),
        ("results/tables/fgsm_baseline_comparison.csv",    None),
        ("results/tables/fgsm_baseline_summary.csv",       None),
        ("results/tables/pgd_baseline_comparison.csv",     None),
        ("results/tables/final_results.csv",               None),
        ("results/tables/final_results.tex",               None),
        ("results/tables/ablation_robustness.csv",         None),
        ("results/tables/calibration_results.csv",         None),
        ("results/tables/table2_fgsm_robustness.csv",      None),
        ("results/tables/table2_fgsm_robustness.tex",      None),
        ("results/tables/table4_pgd_robustness.csv",       None),
        ("results/tables/table4_pgd_robustness.tex",       None),
        ("results/tables/table5_consolidated.csv",         None),
        ("results/tables/table5_consolidated.tex",         None),
        ("outputs/v1.0_20260611_180710/calibration/reliability_after.csv",  None),
        ("outputs/v1.0_20260611_180710/calibration/reliability_before.csv", None),
    ],
}


def copy_file(src_str, dst_dir, new_name=None):
    src = Path(src_str)
    if not src.exists():
        print(f"  [SKIP — NOT FOUND] {src}")
        return False
    dst = dst_dir / (new_name if new_name else src.name)
    shutil.copy2(src, dst)
    print(f"  [OK] {src}  ->  {dst}")
    return True


def main():
    print("=" * 70)
    print("  gather_final.py — COPY ONLY (no train, no delete)")
    print("=" * 70)

    copied = 0
    skipped = 0

    for subfolder, items in SOURCE_MAP.items():
        dst_dir = FINAL / subfolder
        dst_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(items, str) and items.startswith("GLOB:"):
            # Handle glob patterns
            pattern = items[5:]  # strip "GLOB:"
            parent = Path(pattern).parent
            glob_pat = Path(pattern).name
            for src in sorted(parent.glob(glob_pat)):
                ok = copy_file(str(src), dst_dir)
                if ok: copied += 1
                else: skipped += 1
        else:
            print(f"\n[{subfolder}]")
            for src_str, new_name in items:
                ok = copy_file(src_str, dst_dir, new_name)
                if ok: copied += 1
                else: skipped += 1

    print(f"\n{'=' * 70}")
    print(f"  Copied: {copied}  |  Skipped (not found): {skipped}")
    print(f"{'=' * 70}")

    # Print final tree
    print(f"\n=== CÂY THU MUC {FINAL} ===\n")
    for p in sorted(FINAL.rglob("*")):
        depth = len(p.relative_to(FINAL).parts)
        indent = "  " * (depth - 1)
        if p.is_dir():
            print(f"{indent}[{p.name}/]")
        else:
            size_kb = p.stat().st_size // 1024
            print(f"{indent}{p.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
