#!/bin/bash
# Cross-panel adaptation pool for 9b (allowed since the 2026-09-20 clarification).
cd /data1/nagao/Kaggle/TrafficFlowBench/tta_lab || exit 1
PY=../.venv/bin/python
L=runs/cross_logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY tta/run_cross_adapt.py --name cross50 --own-share 0.5 > $L/cross50.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY tta/run_cross_adapt.py --name cross_all --own-share 0.125 > $L/cross_all.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY tta/run_cross_adapt.py --name cross_own --own-share 1.0 > $L/cross_own.log 2>&1 &
wait
$PY -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['verify_00_rep9b','cross_own','cross50','cross_all'])" > $L/report.txt 2>&1
echo done
