#!/bin/bash

python3 train_mutagenicity.py \
  --epochs 10 \
  --batch_size 128 \
  --queryset_size 407 \
  --max_queries 407 \
  --max_queries_test 20 \
  --threshold 0.85 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling random \
  --seed 0 \
  --name mutagenicity_testing \
  --mode offline \
  --save_dir ./saved/debugging/ \
  --data_dir ./data/Mutagenicity \
  --query_dir ./data/rdkit_queryset.csv