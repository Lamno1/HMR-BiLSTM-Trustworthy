#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_reproducible_pipeline.py
----------------------------
Master orchestration script to execute the entire HMR-BiLSTM pipeline sequentially.
Provides a --dry-run option for syntax validation and command inspection without execution.
"""

import sys
import subprocess
import argparse
import os

PIPELINE_STEPS = [
    {
        "step": 0,
        "name": "Preprocess Data (intra-patient splits)",
        "command": [sys.executable, "preprocess.py"],
    },
    {
        "step": 1,
        "name": "Preprocess Data (inter-patient AAMI splits)",
        "command": [sys.executable, "validation/preprocess_aami.py"],
    },
    {
        "step": 2,
        "name": "Train Baseline Models (LSTM, BiLSTM, ResNet1D)",
        "command": [sys.executable, "run_baselines.py"],
    },
    {
        "step": 3,
        "name": "Train HMR-BiLSTM (Main Model — inter-patient)",
        "command": [sys.executable, "train_inter_patient.py"],
    },
    {
        "step": 4,
        "name": "Run Ablation Study",
        "command": [sys.executable, "run_ablation.py"],
    },
    {
        "step": 5,
        "name": "Generate Core Figures",
        "command": [sys.executable, "report_results.py"],
    },
    {
        "step": 6,
        "name": "FGSM Adversarial Robustness Evaluation",
        "command": [sys.executable, "compare_fgsm_baselines.py"],
    },
    {
        "step": 7,
        "name": "PGD Adversarial Robustness Evaluation",
        "command": [sys.executable, "evaluate_pgd.py"],
    },
    {
        "step": 8,
        "name": "AutoAttack Robustness Evaluation",
        "command": [sys.executable, "evaluate_autoattack.py"],
    },
    {
        "step": 9,
        "name": "AutoAttack Robustness Comparison (LSTM/BiLSTM/HMR-BiLSTM)",
        "command": [sys.executable, "evaluate_autoattack_baselines.py"],
    },
    {
        "step": 10,
        "name": "Gaussian Noise Robustness Evaluation",
        "command": [sys.executable, "evaluate_robustness_all.py"],
    },
    {
        "step": 11,
        "name": "Calibration Analysis",
        "command": [sys.executable, "evaluate_calibration.py"],
    },
    {
        "step": 12,
        "name": "Generate Final Tables and Figures",
        "command": [sys.executable, "generate_results_tables.py"],
    },
    {
        "step": 13,
        "name": "Execute Trustworthiness Evaluation Dashboard (T8)",
        "command": [sys.executable, "evaluate_trustworthiness.py"],
    }
]

def main():
    parser = argparse.ArgumentParser(description="Execute HMR-BiLSTM complete reproducible pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    print("=" * 80)
    print(" HMR-BiLSTM: COMPLETE PIPELINE EXECUTION MASTER SCRIPT")
    if args.dry_run:
        print(" [DRY-RUN MODE] Commands will be listed without execution.")
    print("=" * 80)

    # Set environment variable to tag run ID if not dry-run
    if not args.dry_run and "TRUSTWORTHY_RUN_ID" not in os.environ:
        os.environ["TRUSTWORTHY_RUN_ID"] = "v1.0_REPRODUCED"

    for step in PIPELINE_STEPS:
        print(f"\n[STEP {step['step']}] {step['name']}")
        cmd_str = " ".join(step['command'])
        print(f"Command: {cmd_str}")
        
        if args.dry_run:
            continue
            
        try:
            print("Executing...")
            res = subprocess.run(step['command'], check=True)
            print(f"[OK] Step {step['step']} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Step {step['step']} failed with exit code {e.returncode}.", file=sys.stderr)
            sys.exit(e.returncode)
        except Exception as e:
            print(f"\n[ERROR] Step {step['step']} failed to run: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n" + "=" * 80)
    if args.dry_run:
        print(" [OK] Dry-run completed. All step definitions checked.")
    else:
        print(" [OK] ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
        print(" Results saved in results/ and outputs/v1.0_REPRODUCED/")
    print("=" * 80)

if __name__ == "__main__":
    main()
