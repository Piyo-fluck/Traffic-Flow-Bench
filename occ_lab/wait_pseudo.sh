#!/bin/bash
cd /data1/nagao/Kaggle/TrafficFlowBench || exit 1
while pgrep -f "[p]seudo_adapt.py" > /dev/null; do sleep 30; done
grep -v Warn occ_lab/logs/pseudo_adapt.log | tail -n 16
