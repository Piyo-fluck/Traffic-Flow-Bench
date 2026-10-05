#!/bin/bash
# Occupancy as extra input channels for the two neural ongoing families.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
OCC_ARCH=cnn CUDA_VISIBLE_DEVICES=0 $PY occ_lab/train_occ.py --arch cnn --seed 0 > $L/cnn_s0.log 2>&1 &
OCC_ARCH=cnn CUDA_VISIBLE_DEVICES=1 $PY occ_lab/train_occ.py --arch cnn --seed 1 > $L/cnn_s1.log 2>&1 &
OCC_ARCH=tfm CUDA_VISIBLE_DEVICES=2 $PY occ_lab/train_occ.py --arch tfm --seed 0 > $L/tfm_s0.log 2>&1 &
OCC_ARCH=tfm CUDA_VISIBLE_DEVICES=3 $PY occ_lab/train_occ.py --arch tfm --seed 1 > $L/tfm_s1.log 2>&1 &
wait
echo done
