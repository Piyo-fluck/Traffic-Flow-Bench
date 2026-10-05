#!/bin/bash
# Spatial selective scan (Mamba-style) as the fourth ongoing family.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=dcrnn_lab/logs; mkdir -p $L
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s $PY dcrnn_lab/spatial_ssm.py --seed $s --steps 30000 > $L/ssm_s$s.log 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 $PY dcrnn_lab/eval_dcrnn.py --arch ssm --ckpt dcrnn_lab/models/ssm_s0.pt > $L/eval_ssm.log 2>&1
echo done
