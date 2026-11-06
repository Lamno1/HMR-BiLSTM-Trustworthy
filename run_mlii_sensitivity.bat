@echo off
setlocal
cd /d "%~dp0"
REM Sensitivity check for record 114 (channel 0 = V5): retrain the three main cores on DS1/DS2
REM with MLII selected by name. Same recipe and seeds as results/matched_recipe.
set LOG=results\matched_recipe_mlii\mlii.log
if not exist results\matched_recipe_mlii mkdir results\matched_recipe_mlii
python -m validation.preprocess_aami_mlii >> %LOG% 2>&1
if errorlevel 1 exit /b 1
for %%A in (full normc bilstm) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 --data-dir data/processed/splits_mlii --out-dir results/matched_recipe_mlii >> %LOG% 2>&1
    if errorlevel 1 exit /b 1
)
