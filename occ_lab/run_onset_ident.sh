#!/bin/bash
# Structured identity for the onset STID: Laplacian PE (STEMLP) and neighbourhood identity A.E (ST-MLP).
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY occ_lab/train_onset_stid.py --seed $s --ident lappe > $L/onset_stid_lappe_s$s.log 2>&1 &
done
wait
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY occ_lab/train_onset_stid.py --seed $s --ident ab > $L/onset_stid_ab_s$s.log 2>&1 &
done
wait
echo done
