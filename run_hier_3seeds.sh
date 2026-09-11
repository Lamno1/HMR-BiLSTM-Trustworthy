#!/bin/bash
cd /d/HMR-BiLSTM-main
python -u dsta_hier_n_first_train.py --seed 42 --out-suffix seed42 > results/logs/dsta_hier_n_first_seed42.log 2> results/logs/dsta_hier_n_first_seed42.err.log
echo "seed42 xong exit=$?" >> results/logs/dsta_hier_n_first_chain.log
python -u dsta_hier_n_first_train.py --seed 123 --out-suffix seed123 > results/logs/dsta_hier_n_first_seed123.log 2> results/logs/dsta_hier_n_first_seed123.err.log
echo "seed123 xong exit=$?" >> results/logs/dsta_hier_n_first_chain.log
python -u dsta_hier_n_first_train.py --seed 456 --out-suffix seed456 > results/logs/dsta_hier_n_first_seed456.log 2> results/logs/dsta_hier_n_first_seed456.err.log
echo "seed456 xong exit=$?" >> results/logs/dsta_hier_n_first_chain.log
echo "CA 3 SEED HOAN TAT" >> results/logs/dsta_hier_n_first_chain.log
