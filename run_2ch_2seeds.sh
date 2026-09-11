#!/bin/bash
cd /d/HMR-BiLSTM-main
python -u dsta_c_rr_template_2ch_train.py --seed 123 --out-suffix seed123 > results/logs/dsta_c_rr_template_2ch_seed123.log 2> results/logs/dsta_c_rr_template_2ch_seed123.err.log
echo "seed123 xong exit=$?" >> results/logs/dsta_c_rr_template_2ch_chain.log
python -u dsta_c_rr_template_2ch_train.py --seed 7 --out-suffix seed7 > results/logs/dsta_c_rr_template_2ch_seed7.log 2> results/logs/dsta_c_rr_template_2ch_seed7.err.log
echo "seed7 xong exit=$?" >> results/logs/dsta_c_rr_template_2ch_chain.log
echo "CA 2 SEED HOAN TAT" >> results/logs/dsta_c_rr_template_2ch_chain.log
