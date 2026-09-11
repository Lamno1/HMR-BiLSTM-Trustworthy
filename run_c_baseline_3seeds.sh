#!/bin/bash
cd /d/HMR-BiLSTM-main
python -u dsta_c_baseline_multiseed.py --seed 42 --out-suffix seed42 > results/logs/dsta_c_baseline_seed42.log 2> results/logs/dsta_c_baseline_seed42.err.log
echo "seed42 xong exit=$?" >> results/logs/dsta_c_baseline_chain.log
python -u dsta_c_baseline_multiseed.py --seed 123 --out-suffix seed123 > results/logs/dsta_c_baseline_seed123.log 2> results/logs/dsta_c_baseline_seed123.err.log
echo "seed123 xong exit=$?" >> results/logs/dsta_c_baseline_chain.log
python -u dsta_c_baseline_multiseed.py --seed 7 --out-suffix seed7 > results/logs/dsta_c_baseline_seed7.log 2> results/logs/dsta_c_baseline_seed7.err.log
echo "seed7 xong exit=$?" >> results/logs/dsta_c_baseline_chain.log
echo "CA 3 SEED HOAN TAT" >> results/logs/dsta_c_baseline_chain.log
