#!/bin/bash

python3 train_mutagenicity_cv.py \
  --epochs 10 \
  --batch_size 128 \
  --queryset_size 407 \
  --max_queries 407 \
  --max_queries_test 20 \
  --threshold 0.85 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --seed 0 \
  --name cv_testing \
  --mode offline \
  --save_dir ./saved/debugging/ \
  --data_dir ./data/Mutagenicity \
  --query_dir ./data/rdkit_queryset.csv
  #--ckpt_fold_idx -1 \  # Omit if not using a checkpoint
  #--ckpt_epoch 1 \  # Omit if not using a checkpoint
  # --ckpt_sampling_method \ # biased, random, or omit for default=None
  # --ckpt_path None \  # Add a path to the saved checkpoint if needed