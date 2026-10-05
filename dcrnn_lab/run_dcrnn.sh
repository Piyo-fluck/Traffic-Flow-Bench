#!/bin/bash
# Fourth ongoing family: DCRNN-style diffusion-GRU encoder + autoregressive decoder.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=dcrnn_lab/logs; mkdir -p $L
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s $PY dcrnn_lab/dcrnn.py --seed $s --steps 30000 > $L/train_s$s.log 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 $PY dcrnn_lab/eval_dcrnn.py --ckpt dcrnn_lab/models/dcrnn_s0.pt > $L/eval_s0.log 2>&1
echo done
