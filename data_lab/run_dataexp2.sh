#!/bin/bash
# More ongoing windows, with the onset share restored to the production 9.8% by repeating the onset windows.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=data_lab/logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY data_lab/train_more_og.py --name og_step2_share --og-step 2 --max-og 100000 --onset-share 0.098 > $L/train_share.log 2>&1
cd tta_lab || exit 1
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tta/run_adapt.py --name og_share_9b --ckpt ../data_lab/out/og_step2_share.pt --no-report > ../$L/eval_share.log 2>&1
../.venv/bin/python -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['og_ctrl_9b','og_more_9b','og_share_9b'])" > ../$L/report2.txt 2>&1
echo done
