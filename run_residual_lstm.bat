@echo off
REM Stacked residual BiLSTM baseline under the matched harness (seeds 42 123 456),
REM with the official focal recipe and with cross-entropy.
REM Sequential and resumable: stop with Ctrl+C at any time and rerun this file;
REM finished runs are skipped and an interrupted run resumes from its last epoch.
cd /d %~dp0
set LOG=results\matched_recipe\residual_lstm.log

for %%A in (reslstm reslstm_ce) do (
    python -u train_matched_recipe.py --arch %%A --seeds 42 123 456 >> %LOG% 2>&1
)
echo Residual LSTM runs finished. >> %LOG%
