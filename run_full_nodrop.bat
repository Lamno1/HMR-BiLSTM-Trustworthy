@echo off
REM ResMemECG without the in-cell dropout (seeds 42 123 456): the fair counterpart of
REM normc_nodrop, which reached 0.573 against 0.559 for ResMemECG with the dropout.
REM Sequential and resumable: stop with Ctrl+C at any time and rerun this file.
cd /d %~dp0
set LOG=results\matched_recipe\full_nodrop.log

python -u train_matched_recipe.py --arch full_nodrop --seeds 42 123 456 >> %LOG% 2>&1
echo full_nodrop runs finished. >> %LOG%
