#!/bin/bash
# Two more seeds of the more-ongoing recipe, then the ongoing slot as a 3-model average (unadapted, as production).
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=data_lab/logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY data_lab/train_more_og.py --name og_share_s21 --og-step 2 --max-og 100000 --onset-share 0.098 --seed 21 > $L/train_s21.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY data_lab/train_more_og.py --name og_share_s22 --og-step 2 --max-og 100000 --onset-share 0.098 --seed 22 > $L/train_s22.log 2>&1 &
wait
cd tta_lab || exit 1
for s in og_step2_share og_share_s21 og_share_s22; do
  g=0; case $s in og_share_s21) g=1;; og_share_s22) g=2;; esac
  CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python tta/run_adapt.py --name ${s}_noad --ckpt ../data_lab/out/$s.pt --no-adapt --kinds og --no-report > ../$L/eval_${s}_noad.log 2>&1 &
done
wait
echo done
