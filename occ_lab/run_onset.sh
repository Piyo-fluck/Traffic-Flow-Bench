#!/bin/bash
# Single-horizon (T+30) onset models: 4 CNN seeds fine-tuned from the production checkpoints.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
M=work/models
CUDA_VISIBLE_DEVICES=0 $PY occ_lab/train_onset.py --seed 0 --init $M/t2_cnn_lov.pt    > $L/onset30_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY occ_lab/train_onset.py --seed 1 --init $M/t2_cnn_lov_s1.pt > $L/onset30_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY occ_lab/train_onset.py --seed 2 --init $M/t2_cnn_lov_s2.pt > $L/onset30_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY occ_lab/train_onset.py --seed 3                          > $L/onset30_s3.log 2>&1 &
wait
echo done
