#!/bin/bash
# Onset STID seeds 4-7, so the member is an 8-seed average.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs
for s in 4 5 6 7; do
  g=$((s-4))
  CUDA_VISIBLE_DEVICES=$g $PY occ_lab/train_onset_stid.py --seed $s > $L/onset_stid_s$s.log 2>&1 &
done
wait
echo done
