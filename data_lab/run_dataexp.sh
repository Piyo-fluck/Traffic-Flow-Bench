#!/bin/bash
# More ongoing training windows: the new model and the matched control, then 9b adaptation + report.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=data_lab/logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY data_lab/train_more_og.py --name og_step2_nocap --og-step 2 --max-og 100000 > $L/train_more.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY data_lab/train_more_og.py --name ctrl_prod --og-step 6 --max-og 3000 > $L/train_ctrl.log 2>&1 &
wait
cd tta_lab || exit 1
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tta/run_adapt.py --name og_more_9b --ckpt ../data_lab/out/og_step2_nocap.pt --no-report > ../$L/eval_more.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tta/run_adapt.py --name og_ctrl_9b --ckpt ../data_lab/out/ctrl_prod.pt --no-report > ../$L/eval_ctrl.log 2>&1 &
wait
../.venv/bin/python -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['verify_00_rep9b','og_ctrl_9b','og_more_9b'])" > ../$L/report.txt 2>&1
echo done
