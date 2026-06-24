@echo off
REM ============================================================
REM HMR-BiLSTM: Complete Pipeline Execution
REM ============================================================

set PY=python

echo.
echo ============================================================
echo  STEP 0: Preprocess Data (intra-patient splits)
echo ============================================================
%PY% preprocess.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 1: Preprocess Data (inter-patient AAMI splits)
echo ============================================================
%PY% validation\preprocess_aami.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 2: Train Baseline Models (LSTM, BiLSTM, ResNet1D)
echo ============================================================
%PY% run_baselines.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 3: Train HMR-BiLSTM (Main Model - inter-patient)
echo ============================================================
%PY% train_inter_patient.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 4: Run Ablation Study
echo ============================================================
%PY% run_ablation.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 4: Generate Core Figures
echo ============================================================
%PY% report_results.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 5: FGSM Adversarial Robustness Evaluation
echo ============================================================
%PY% compare_fgsm_baselines.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 6: PGD Adversarial Robustness Evaluation
echo ============================================================
%PY% evaluate_pgd.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 7: Ablation Variants Robustness
echo ============================================================
%PY% evaluate_ablation_robustness.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 8: Gaussian Noise Robustness Evaluation
echo ============================================================
%PY% evaluate_robustness_all.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 9: Calibration Analysis
echo ============================================================
%PY% evaluate_calibration.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 10: Combine Ablation Tables
echo ============================================================
%PY% combine_ablation_tables.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 11: Generate Final Tables + Figures
echo ============================================================
%PY% generate_results_tables.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 12: Export Final Results
echo ============================================================
%PY% plot_and_export.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 13: AutoAttack Robustness Evaluation (T7)
echo ============================================================
%PY% evaluate_autoattack.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  STEP 14: Execute Trustworthiness Dashboard (T8)
echo ============================================================
%PY% evaluate_trustworthiness.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  ✓ ALL STEPS COMPLETED SUCCESSFULLY!
echo  Results saved to: results/
echo  Tables:  results/tables/
echo  Figures: results/figures/
echo  Logs:    results/logs/
echo ============================================================
goto end

:error
echo.
echo [ERROR] Pipeline failed at a step above. Check output.
exit /b 1

:end

