#!/bin/bash
# Onset STID (T+30 only): four seeds.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY occ_lab/train_onset_stid.py --seed $s > $L/onset_stid_s$s.log 2>&1 &
done
wait
echo done
