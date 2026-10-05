#!/bin/bash
# STID with occupancy: four seeds, one per GPU.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s $PY occ_lab/train_stid.py --seed $s > $L/stid_s$s.log 2>&1 &
done
wait
echo done
