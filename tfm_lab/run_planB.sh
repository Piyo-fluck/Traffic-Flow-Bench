#!/bin/bash
# Plan B driver on Ada: train 8 rec-head transformers (2 per GPU), adapt/evaluate each, then report.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=tfm_lab/logs; mkdir -p $L
echo "[$(date +%T)] training"
for s in 3 4 5 6 7 8 9 10; do
  g=$(( (s - 3) % 4 ))
  CUDA_VISIBLE_DEVICES=$g $PY tfm_lab/tfm_rec.py --seed $s > $L/train_s$s.log 2>&1 &
done
wait
echo "[$(date +%T)] adaptation / evaluation"
for s in 3 4 5 6 7 8 9 10; do
  g=$(( (s - 3) % 4 ))
  CUDA_VISIBLE_DEVICES=$g $PY tfm_lab/eval_tfm_adapt.py --seed $s > $L/eval_s$s.log 2>&1 &
done
wait
echo "[$(date +%T)] report"
$PY tfm_lab/eval_tfm_adapt.py --report > $L/report.txt 2>&1
echo "[$(date +%T)] done"
