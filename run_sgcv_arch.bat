@echo off
REM Matched recurrent cores under the controlled subject-grouped CV (5 folds x seeds 42/123).
REM Fast nn.LSTM configs first (~1 h each), then the custom-cell configs (~15 h each).
REM Resumable: re-run this file after a reboot; finished folds are skipped.
cd /d %~dp0
if not exist results\sgcv_arch mkdir results\sgcv_arch
for %%A in (bilstm bilstm_nocw bilstm_ce normc_nodrop normc) do (
    python -u -m validation.run_sgcv_arch --arch %%A >> results\sgcv_arch\%%A.log 2>&1
)
echo SG-CV arch runs finished.
