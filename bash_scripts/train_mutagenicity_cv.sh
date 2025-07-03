#!/bin/bash

# ckpt loading
# ckpt_path="./saved/testing_cv/fold_idx=4/biased/4q43gde9/ckpt/epoch10.ckpt"
# ckpt_fold_idx=4
# ckpt_epoch=10
# ckpt_sampling_method=biased

# With ckpt_path
# python3 main_mutagenicity_cv.py --epochs=$epochs --max_queries=$max_queries --max_queries_test=$max_queries_test --threshold=$threshold --seed=$seed --name=$name --mode=$mode --save_dir=$save_dir --ckpt_path=$ckpt_path --ckpt_fold_idx=$ckpt_fold_idx --ckpt_epoch=$ckpt_epoch --ckpt_sampling_method=$ckpt_sampling_method

python3 train_mutagenicity_cv.py \
  --epochs 100 \
  --batch_size 128 \
  --queryset_size 407 \
  --max_queries 407 \
  --max_queries_test 20 \
  --threshold 0.85 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --seed 0 \
  --name 500ep_cv_updated_qryset \
  --mode online \
  --save_dir ./saved/ \
  --data_dir ./data/Mutagenicity \
  --query_dir ./data/rdkit_queryset.csv \
  #--ckpt_fold_idx -1 \  # Omit if not using a checkpoint
  #--ckpt_epoch 1 \  # Omit if not using a checkpoint
  # --ckpt_sampling_method \ # biased, random, or omit for default=None
  # --ckpt_path \  # Add a path to the saved checkpoint if needed