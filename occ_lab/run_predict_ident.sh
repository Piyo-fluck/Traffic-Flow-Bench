#!/bin/bash
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
for v in lappe ab; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python occ_lab/predict_onset_stid.py \
    --ckpt occ_lab/models/onset_stid_${v}_s0.pt occ_lab/models/onset_stid_${v}_s1.pt \
           occ_lab/models/onset_stid_${v}_s2.pt occ_lab/models/onset_stid_${v}_s3.pt \
    --tag stid_${v}
done
echo done
