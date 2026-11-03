@echo off
REM Which implementation detail separates the No-RMC cell from nn.LSTM? (seeds 42 123 456)
REM Each variant removes one difference from the No-RMC cell; normc_lstmlike removes both the
REM in-cell dropout and the custom initialisation and should approach bilstm (0.473).
REM Ordered by priority: results of a finished variant are usable even if later ones are not run.
REM Sequential and resumable: stop with Ctrl+C at any time and rerun this file;
REM finished runs are skipped and an interrupted run resumes from its last epoch.
cd /d %~dp0
set LOG=results\matched_recipe\cell_impl.log

for %%A in (normc_nodrop normc_torchinit normc_lstmlike normc_nofb) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)
echo Cell-implementation runs finished. >> %LOG%
