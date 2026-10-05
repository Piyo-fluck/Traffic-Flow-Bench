#!/bin/bash
# Candidate U: candidate T with the onset CNN adapted from THREE reconstruction-head seeds instead of one.
#
# T replaced the three-seed CNN of M by a single adapted model, so part of its onset variance is just the seed.
# This waits for the two extra reconstruction-head models on the Ada machine, adapts all three causally on the past
# days of each scored month, averages their probabilities, and rebuilds the candidate. Fire and forget.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
ADA=nagao@10.80.238.106
REM=/data1/nagao/Kaggle/TrafficFlowBench
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the two reconstruction-head models on Ada"
for i in $(seq 1 90); do
  n=$(ssh -o ConnectTimeout=10 $ADA "ls $REM/work/models/t2_cnn_lov_rec_s21.pt $REM/work/models/t2_cnn_lov_rec_s22.pt 2>/dev/null | wc -l" 2>/dev/null)
  if [ "${n:-0}" -ge 2 ]; then break; fi
  sleep 30
done
log "models present: ${n:-0} of 2"
scp $ADA:$REM/work/models/t2_cnn_lov_rec_s21.pt work/models_remote/ || exit 1
scp $ADA:$REM/work/models/t2_cnn_lov_rec_s22.pt work/models_remote/ || exit 1

for s in 20 21 22; do
  log "adapting seed $s on the past days of each scored month"
  T2CNN_REC=0.3 T2CNN_LOVASZ=0.5 OMP_NUM_THREADS=4 $PY scripts/t2_adapt_official.py \
    --ckpt work/models_remote/t2_cnn_lov_rec_s${s}.pt --out-dir work/sub_t2adapt_s${s} > work/adapt_s${s}.log 2>&1 || exit 1
done

log "averaging the three adapted probability sets"
$PY - <<'PYEOF'
import numpy as np
from pathlib import Path
out = Path("work/sub_t2adapt3"); out.mkdir(parents=True, exist_ok=True)
dirs = [Path(f"work/sub_t2adapt_s{s}") for s in (20, 21, 22)]
for f in sorted(dirs[0].glob("prob_*.npz")):
    zs = [np.load(d / f.name) for d in dirs]
    ids = zs[0]["window_id"]
    assert all((z["window_id"] == ids).all() for z in zs), f.name
    np.savez(out / f.name, prob=np.mean([z["prob"] for z in zs], 0).astype(np.float32),
             is_onset=zs[0]["is_onset"], window_id=ids, t0=zs[0]["t0"])
print("averaged", len(list(out.glob("prob_*.npz"))), "files")
PYEOF

log "rebuilding the queue files"
M=work/models_remote
OMP_NUM_THREADS=4 $PY scripts/t2_cnn_blend_predict.py \
  --cnn $M/t2_cnn_lov.pt $M/t2_cnn_lov_s1.pt $M/t2_cnn_lov_s2.pt --cnn-arch cnn \
  --og-nets $M/t2_cnn_lov.pt $M/t2_cnn_lov_s1.pt $M/t2_cnn_lov_s2.pt --og-arch cnn \
  --og-nets2 $M/t2_cnn_lov_tfm_s3.pt $M/t2_cnn_lov_tfm_s4.pt $M/t2_cnn_lov_tfm_s5.pt $M/t2_cnn_lov_tfm_s6.pt \
             $M/t2_cnn_lov_tfm_s7.pt $M/t2_cnn_lov_tfm_s8.pt $M/t2_cnn_lov_tfm_s9.pt $M/t2_cnn_lov_tfm_s10.pt \
  --og-arch2 tfm --w-og2 0.60 \
  --onset-probs-dir work/sub_t2robust_bag5 --onset-cnn-probs-dir work/sub_t2adapt3 \
  --ongoing-probs-dir work/sub_t2gs --w-on 0.7 --w-og 0.20 --tau 0.50 \
  --out-dir work/sub_t2U > work/build_U.log 2>&1 || exit 1

log "assembling the upload"
mkdir -p work/sub_U_ttt_t2adapt3_t4ref
OMP_NUM_THREADS=4 $PY scripts/swap_task_rows.py --base work/sub_R_ttt_t2ens_t4ref/submission.csv --task queue \
  --files work/sub_t2U/queue_validation.csv work/sub_t2U/queue_private.csv \
  --out work/sub_U_ttt_t2adapt3_t4ref/submission.csv >> work/build_U.log 2>&1 || exit 1

log "verifying against candidate T"
$PY - <<'PYEOF'
import pandas as pd
A = "work/sub_T_ttt_t2adapt_t4ref/submission.csv"; B = "work/sub_U_ttt_t2adapt3_t4ref/submission.csv"
rows, diff = 0, {}
for ca, cb in zip(pd.read_csv(A, chunksize=500_000), pd.read_csv(B, chunksize=500_000)):
    rows += len(ca)
    val = [c for c in ca.columns if c not in ("submission_id", "task")]
    d = (ca[val].fillna(-999) != cb[val].fillna(-999)).any(axis=1)
    for t, n in ca.loc[d, "task"].value_counts().items():
        diff[t] = diff.get(t, 0) + int(n)
print("U vs T: rows", rows, "differing by task", diff)
PYEOF
log "candidate U ready: work/sub_U_ttt_t2adapt3_t4ref/submission.csv"
