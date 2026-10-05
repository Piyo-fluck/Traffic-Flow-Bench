#!/bin/bash
# E re-check: inner SGD steps large enough to actually move the weights (note 72: alpha 1e-3 was inert).
cd /data1/nagao/Kaggle/TrafficFlowBench/tta_lab
PY=../.venv/bin/python
L=runs/E2_logs; mkdir -p $L
echo "[$(date +%T)] stage 1: meta training (alpha 3e-2, 1e-1) + no-meta controls with the same inner loop"
CUDA_VISIBLE_DEVICES=0 $PY tta/run_E_meta.py --name E2_meta_a3e2 --steps 3000 --inner 5 --alpha 3e-2 --order 1 --aug 0.1 > $L/meta_a3e2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY tta/run_E_meta.py --name E2_meta_a1e1 --steps 3000 --inner 5 --alpha 1e-1 --order 1 --aug 0.1 > $L/meta_a1e1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY tta/run_adapt.py --name E2_ctrl_sgd3e2 --opt sgd --lr 3e-2 --steps 5 --reset-daily --no-report > $L/ctrl_sgd3e2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY tta/run_adapt.py --name E2_ctrl_sgd1e1 --opt sgd --lr 1e-1 --steps 5 --reset-daily --no-report > $L/ctrl_sgd1e1.log 2>&1 &
wait
echo "[$(date +%T)] stage 2: evaluate the meta checkpoints with and without adaptation"
CUDA_VISIBLE_DEVICES=0 $PY tta/run_adapt.py --name E2_eval_a3e2 --ckpt runs/E2_meta_a3e2/meta.pt --opt sgd --lr 3e-2 --steps 5 --reset-daily --no-report > $L/eval_a3e2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY tta/run_adapt.py --name E2_eval_a1e1 --ckpt runs/E2_meta_a1e1/meta.pt --opt sgd --lr 1e-1 --steps 5 --reset-daily --no-report > $L/eval_a1e1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY tta/run_adapt.py --name E2_noad_a3e2 --ckpt runs/E2_meta_a3e2/meta.pt --no-adapt --no-report > $L/noad_a3e2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY tta/run_adapt.py --name E2_noad_a1e1 --ckpt runs/E2_meta_a1e1/meta.pt --no-adapt --no-report > $L/noad_a1e1.log 2>&1 &
wait
echo "[$(date +%T)] stage 3: report"
$PY -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['verify_00_rep9b','verify_00_noadapt','E2_ctrl_sgd3e2','E2_ctrl_sgd1e1','E2_eval_a3e2','E2_noad_a3e2','E2_eval_a1e1','E2_noad_a1e1'], file='runs/E2_logs/report.txt')" > $L/report.log 2>&1
echo "[$(date +%T)] done"
