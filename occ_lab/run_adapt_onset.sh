#!/bin/bash
# Idea 9b adaptation of the single-horizon onset models, on the official-like windows.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs
export T2CNN_REC=0.3
while ! grep -q done occ_lab/logs/driver_onset_rec.log 2>/dev/null; do sleep 20; done
for s in 20 21 22; do
  g=$((s-20))
  CUDA_VISIBLE_DEVICES=$g $PY occ_lab/adapt_onset.py --ckpt occ_lab/models/onset30_cnn_s$s.pt --tag onset30_s$s > $L/adapt_onset_s$s.log 2>&1 &
done
# the production rec checkpoints adapted the same way, as the reference
CUDA_VISIBLE_DEVICES=3 $PY occ_lab/adapt_onset.py --ckpt work/models/t2_cnn_lov_rec_s20.pt --tag base_s20 > $L/adapt_onset_base.log 2>&1 &
wait
echo done
