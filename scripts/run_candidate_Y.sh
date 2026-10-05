#!/bin/bash
# Candidate Y: T with Task 1 test-time training extended from 400 to 1200 steps (note 32 appendix: +0.0003 total).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
OUT=work/sub_m3ttt1200
export OMP_NUM_THREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }
for s in validation private; do
  log "TTT 1200 steps: $s"
  TTT_STEPS=1200 $PY scripts/t1_ttt_train.py --split $s --out-dir $OUT || exit 1
done
for s in validation private; do
  for p in D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S D12_I405_N D12_I405_S; do
    $PY scripts/t1_predict_ckpt.py --ckpt "$OUT/ttt_${s}_0.pt:64" "$OUT/ttt_${s}_1.pt:64" "$OUT/ttt_${s}_2.pt:128" \
      --month-kappa 2 --month-kappa-q 10 --out-dir $OUT --panels $p --splits $s --npz-only || exit 1
  done
  log "inference done: $s"
done
$PY scripts/t1_npz_to_state.py --pred-dir $OUT || exit 1
mkdir -p work/sub_Y_ttt1200_t2adapt_t4ref
$PY scripts/swap_task_rows.py --base work/sub_T_ttt_t2adapt_t4ref/submission.csv --task state \
  --files $OUT/state_validation.csv $OUT/state_private.csv --out work/sub_Y_ttt1200_t2adapt_t4ref/submission.csv || exit 1
$PY - <<'PYEOF'
import pandas as pd
A = "work/sub_T_ttt_t2adapt_t4ref/submission.csv"; B = "work/sub_Y_ttt1200_t2adapt_t4ref/submission.csv"
rows, diff = 0, {}
for ca, cb in zip(pd.read_csv(A, chunksize=500_000), pd.read_csv(B, chunksize=500_000)):
    rows += len(ca)
    assert (ca.submission_id.values == cb.submission_id.values).all()
    val = [c for c in ca.columns if c not in ("submission_id", "task")]
    d = (ca[val].fillna(-999) != cb[val].fillna(-999)).any(axis=1)
    for t, n in ca.loc[d, "task"].value_counts().items():
        diff[t] = diff.get(t, 0) + int(n)
print("Y vs T: rows", rows, "differing by task", diff)
PYEOF
log "candidate Y ready"
