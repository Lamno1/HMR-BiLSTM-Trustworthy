@echo off
REM Multi-seed matched-recipe ablations (seeds 42 123 456), most informative first.
REM Sequential and resumable: stop with Ctrl+C at any time and rerun this file;
REM finished runs are skipped and an interrupted run resumes from its last epoch.
cd /d %~dp0
set LOG=results\matched_recipe\ablation.log

REM 1) Recipe ablations on the nn.LSTM control (cuDNN, a few minutes per run)
for %%A in (bilstm_noadv bilstm_ce bilstm_nocw) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)

REM 2) Recipe ablations on ResMemECG (several hours per seed)
for %%A in (full_noadv full_ce) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)

REM 3) Architecture ablations on ResMemECG
for %%A in (nohybrid meanpool) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)

REM 4) Lowest priority: remaining recipe term and RMC sub-components
for %%A in (full_nocw nointer full_nosmooth) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)
echo All ablations finished. >> %LOG%
