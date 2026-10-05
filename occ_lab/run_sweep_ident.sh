#!/bin/bash
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
for t in stid30 stid_lappe stid_ab; do
  echo "##### $t"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python occ_lab/onset_weight_sweep.py --tag $t 2>&1 | grep -A 6 "beat production on both"
done
