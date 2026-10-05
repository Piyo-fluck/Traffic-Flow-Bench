#!/bin/bash
# Two more seeds per architecture, so occ_cnn can be compared with cnn3 at equal ensemble size.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
OCC_ARCH=cnn CUDA_VISIBLE_DEVICES=0 $PY occ_lab/train_occ.py --arch cnn --seed 2 > $L/cnn_s2.log 2>&1 &
OCC_ARCH=cnn CUDA_VISIBLE_DEVICES=1 $PY occ_lab/train_occ.py --arch cnn --seed 3 > $L/cnn_s3.log 2>&1 &
OCC_ARCH=tfm CUDA_VISIBLE_DEVICES=2 $PY occ_lab/train_occ.py --arch tfm --seed 2 > $L/tfm_s2.log 2>&1 &
OCC_ARCH=tfm CUDA_VISIBLE_DEVICES=3 $PY occ_lab/train_occ.py --arch tfm --seed 3 > $L/tfm_s3.log 2>&1 &
wait
echo done
