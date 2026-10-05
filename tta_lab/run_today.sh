#!/bin/bash
# Plan 2: add same-day (earlier today) windows to the 9b adaptation, now that data at or before T is allowed.
cd /data1/nagao/Kaggle/TrafficFlowBench/tta_lab || exit 1
PY=../.venv/bin/python
L=runs/today_logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY tta/run_adapt.py --name today10 --today-steps 10 --no-report > $L/today10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY tta/run_adapt.py --name today30 --today-steps 30 --no-report > $L/today30.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY tta/run_adapt.py --name today30_only --today-steps 30 --steps 0 --no-report > $L/today30_only.log 2>&1 &
wait
$PY -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['verify_00_rep9b','today10','today30','today30_only'])" > $L/report.txt 2>&1
echo done
