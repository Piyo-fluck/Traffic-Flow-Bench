#!/bin/bash
# Task 1 of candidate AS (LB 0.88405; Task 1 unchanged through the current best BB 0.88697), end to end.
# Run from this folder's root, which must also contain:
#   data/kaggle_public/          the competition release (corridors/, task1/, task2/, task4/, submission_key.csv, ...)
#   trafficflowbench-public/     the organizers' repository (only scripts/assemble_submission.py needs it)
# GPU: one CUDA GPU is enough; set GPU=<id>. See README_task1_AS.md for what each step does.
set -e
GPU=${GPU:-0}
PY=${PY:-python}
export M3_WIDTH=192 M3_RF=all1          # every step below must see these two (architecture of the checkpoints)
mkdir -p work/models work/logs work/cache work/sub_sslttt work/sub_sslens5

# 0. base tables (weekday x time-of-day x station means over all train days) -> work/cache/<panel>_t1tables.npz
#    (the first call of tfb.data.load also builds work/cache/<panel>_<split>.npz from the parquet release)
$PY scripts/t1_tables_cache.py 2>&1 | tee work/logs/t1_tables.log

# 1. five members: stage 1 self-supervised (mask rate U(0.10, 0.70)) 25 epochs + stage 2 real task 25 epochs
#    -> work/models/t1_ssl_wide_all.pt (seed 0), t1_ssl_wide_s{1..4}_all.pt
for s in 0 1 2 3 4; do
  tag=ssl_wide; [ "$s" != 0 ] && tag=ssl_wide_s$s
  CUDA_VISIBLE_DEVICES=$GPU M3_SEED=$s $PY -u analysis/t1_ssl.py --mode wide --pre-epochs 25 --ft-epochs 25 \
    --ft-lr 0.0007 --tag $tag > work/logs/$tag.log 2>&1
  echo "member $s done ($tag)"
done

# 2. test-time training: each member adapted separately on each scored split, 1200 steps
#    -> work/sub_sslttt/ttt_<split>_<i>.pt (i = member index in TTT_CKPTS order)
C="work/models/t1_ssl_wide_all.pt:192"
for s in 1 2 3 4; do C="$C,work/models/t1_ssl_wide_s${s}_all.pt:192"; done
for split in validation private; do
  CUDA_VISIBLE_DEVICES=$GPU TTT_STEPS=1200 TTT_CKPTS="$C" $PY -u scripts/t1_ttt_train.py --split $split \
    --out-dir work/sub_sslttt > work/logs/ttt_$split.log 2>&1
  echo "TTT $split done"
done

# 3. inference: mean of the five adapted members, month-deviation base (kappa 2 speed, 10 flow)
for split in validation private; do
  CK=""
  for i in 0 1 2 3 4; do CK="$CK work/sub_sslttt/ttt_${split}_${i}.pt:192"; done
  CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/t1_predict_ckpt.py --ckpt $CK --month-kappa 2 --month-kappa-q 10 \
    --out-dir work/sub_sslens5 --splits $split --npz-only > work/logs/pred_$split.log 2>&1
  echo "prediction $split done"
done

# 4. state CSVs (checked against the published templates) -> work/sub_sslens5/state_{validation,private}.csv
$PY scripts/t1_npz_to_state.py --pred-dir work/sub_sslens5 | tee work/logs/npz_to_state.log

# 5. put Task 1 into an existing full submission (all other rows kept byte for byte):
#    $PY scripts/swap_task_rows.py --base <existing submission.csv> --task state \
#        --files work/sub_sslens5/state_validation.csv work/sub_sslens5/state_private.csv --out work/submission_t1AS.csv
#    or build a submission from scratch with scripts/assemble_submission.py (needs queue and odme files too).
echo "Task 1 AS done: work/sub_sslens5/state_*.csv"
