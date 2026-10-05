#!/bin/bash
# Single-horizon onset models WITH the reconstruction head, so idea 9b can adapt them afterwards.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
M=work/models
export T2CNN_REC=0.3
CUDA_VISIBLE_DEVICES=0 $PY occ_lab/train_onset.py --seed 20 --init $M/t2_cnn_lov_rec_s20.pt > $L/onset30rec_s20.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY occ_lab/train_onset.py --seed 21 --init $M/t2_cnn_lov_rec_s21.pt > $L/onset30rec_s21.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY occ_lab/train_onset.py --seed 22 --init $M/t2_cnn_lov_rec_s22.pt > $L/onset30rec_s22.log 2>&1 &
wait
echo done
