#!/bin/bash
# Coverage environments for the residual corrector: one process per environment.
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
PY=.venv/bin/python
L=occ_lab/logs; mkdir -p $L
CUDA_VISIBLE_DEVICES=0 $PY occ_lab/cache_cov.py --env obs  > $L/cov_obs.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY occ_lab/cache_cov.py --env c50  > $L/cov_c50.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY occ_lab/cache_cov.py --env c60  > $L/cov_c60.log  2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY occ_lab/cache_cov.py --env c70  > $L/cov_c70.log  2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY occ_lab/cache_cov.py --env full > $L/cov_full.log 2>&1 &
wait
echo done
