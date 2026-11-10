#!/bin/bash
# Task 2 ONGOING of the current best submission (BB 0.88697; its ongoing is candidate AU's), end to end.
# Run from this folder's root, which must also contain:
#   data/kaggle_public/          the competition release (corridors/, task2/, submission_key.csv, ...)
#   trafficflowbench-public/     the organizers' repository (scripts/assemble_submission.py uses its merge script)
# Usage:  STAGE=all GPU=0 bash run_task2_ongoing.sh      (STAGE = train | corrector | official | build | all)
# See README_task2_ongoing.md. Every stage writes its outputs where the next one reads them.
set -e
STAGE=${STAGE:-all}
GPU=${GPU:-0}
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
mkdir -p work/models work/logs occ_lab/models occ_lab/logs

if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
  # 1a. cnn3: T2Net width 64, BCE + 0.5 Lovasz, 30k steps, seeds 0-2 -> work/models/t2_cnn_lov{,_s1,_s2}.pt
  for s in 0 1 2; do
    T2CNN_LOVASZ=0.5 T2CNN_STEPS=30000 T2CNN_SEED=$s $PY analysis/t2_cnn.py > work/logs/cnn_lov_s$s.log 2>&1
  done
  # 1b. tfm8: T2Tfm, same loss and steps, seeds 3-10 -> work/models/t2_cnn_lov_tfm_s{3..10}.pt
  for s in 3 4 5 6 7 8 9 10; do
    T2CNN_ARCH=tfm T2CNN_LOVASZ=0.5 T2CNN_STEPS=30000 T2CNN_SEED=$s $PY analysis/t2_cnn.py > work/logs/tfm_s$s.log 2>&1
  done
  # 1c. p_PS backbones: T2Net width 192 with the rec head, 30k SSL steps (masked reconstruction) then 30k supervised,
  #     seeds 20-23 -> work/models/t2_cnn_lov_w192_rec_ssl30k_s{20..23}.pt
  for s in 20 21 22 23; do
    T2CNN_REC=0.3 T2CNN_LOVASZ=0.5 T2CNN_STEPS=30000 T2CNN_WIDTH=192 T2CNN_SSL=30000 T2CNN_SEED=$s \
      $PY analysis/t2_cnn.py > work/logs/ssl_w192_s$s.log 2>&1
  done
  # 1d. occupancy members (corrector features only): occ_cnn / occ_tfm seeds 0-3 -> occ_lab/models/occ_{cnn,tfm}_s*.pt
  #     (the first run also writes occ_lab/models/occ_profiles.npz)
  for a in cnn tfm; do for s in 0 1 2 3; do
    OCC_ARCH=$a $PY occ_lab/train_occ.py --arch $a --seed $s > occ_lab/logs/${a}_s$s.log 2>&1
  done; done
  # 1e. STID seeds 0-3 (corrector features only) -> occ_lab/models/stid_s*.pt
  for s in 0 1 2 3; do $PY occ_lab/train_stid.py --seed $s > occ_lab/logs/stid_s$s.log 2>&1; done
  # 1f. gs1: LightGBM ongoing model with global network-state features + per-panel PCA -> work/models_gstate/
  $PY analysis/t2_gstate.py > work/logs/gstate.log 2>&1
  echo "stage train done"
fi

if [ "$STAGE" = corrector ] || [ "$STAGE" = all ]; then
  # 2. residual corrector: out-of-fold member predictions on the train tail (2031-01/02, never fitted by the members),
  #    then LightGBM on the residual -> occ_lab/models/residual_lgb.txt
  $PY occ_lab/cache_oof.py > work/logs/cache_oof.log 2>&1
  $PY occ_lab/fit_corrector.py --out occ_lab/models/residual_lgb.txt > work/logs/fit_corrector.log 2>&1
  echo "stage corrector done"
fi

if [ "$STAGE" = official ] || [ "$STAGE" = all ]; then
  # 3a. p_PS: each w192 SSL backbone adapted on the scored month (causal, supervised by released labels + rec)
  for s in 20 21 22 23; do
    T2CNN_REC=0.3 T2CNN_LOVASZ=0.5 T2CNN_WIDTH=192 $PY scripts/t2_pseudo_adapt_official.py \
      --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s$s.pt --out-dir work/sub_t2pseudo_ssl_s$s > work/logs/official_ps_s$s.log 2>&1
  done
  $PY scripts/mean_prob_dirs.py --out work/sub_t2pseudo_ssl4 work/sub_t2pseudo_ssl_s{20,21,22,23}
  # 3b. the 8 transformers adapted the same way, supervised loss only (no rec head)
  for s in 3 4 5 6 7 8 9 10; do
    env -u T2CNN_REC T2CNN_LOVASZ=0.5 $PY scripts/t2_pseudo_adapt_official.py --arch tfm \
      --ckpt work/models/t2_cnn_lov_tfm_s$s.pt --out-dir work/sub_t2tfmps_s$s > work/logs/official_tfm_s$s.log 2>&1
  done
  $PY scripts/mean_prob_dirs.py --out work/sub_t2tfmps8 work/sub_t2tfmps_s{3,4,5,6,7,8,9,10}
  # 3c. gs1 on the official windows
  $PY scripts/t2_gstate_probs.py --out-dir work/sub_t2gs > work/logs/gstate_probs.log 2>&1
  echo "stage official done"
fi

if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  # 4. blend + corrector -> work/sub_t2ongoing/ongoing_{validation,private}.csv (ongoing windows only)
  M=work/models; O=occ_lab/models
  $PY scripts/t2_residual_predict.py --out-dir work/sub_t2ongoing --eta 0.75 --gate 0.10 \
    --cnn $M/t2_cnn_lov.pt $M/t2_cnn_lov_s1.pt $M/t2_cnn_lov_s2.pt \
    --tfm $M/t2_cnn_lov_tfm_s3.pt $M/t2_cnn_lov_tfm_s4.pt $M/t2_cnn_lov_tfm_s5.pt $M/t2_cnn_lov_tfm_s6.pt \
          $M/t2_cnn_lov_tfm_s7.pt $M/t2_cnn_lov_tfm_s8.pt $M/t2_cnn_lov_tfm_s9.pt $M/t2_cnn_lov_tfm_s10.pt \
    --occ-cnn $O/occ_cnn_s0.pt $O/occ_cnn_s1.pt $O/occ_cnn_s2.pt $O/occ_cnn_s3.pt \
    --occ-tfm $O/occ_tfm_s0.pt $O/occ_tfm_s1.pt $O/occ_tfm_s2.pt $O/occ_tfm_s3.pt \
    --stid $O/stid_s0.pt $O/stid_s1.pt $O/stid_s2.pt $O/stid_s3.pt \
    --profiles $O/occ_profiles.npz --gs-dir work/sub_t2gs \
    --pseudo-dir work/sub_t2pseudo_ssl4 --w-pseudo 0.60 --tfm-dir work/sub_t2tfmps8 --tfm-feat frozen \
    > work/logs/residual_predict.log 2>&1
  # 5. put the ongoing rows into an existing full submission (its onset rows and other tasks kept byte for byte):
  #    $PY scripts/t2_patch_ongoing.py --base <submission.csv> \
  #        --ongoing work/sub_t2ongoing/ongoing_validation.csv work/sub_t2ongoing/ongoing_private.csv --out-dir work/sub_t2ongoing_q
  #    $PY scripts/swap_task_rows.py --base <submission.csv> --task queue \
  #        --files work/sub_t2ongoing_q/queue_validation.csv work/sub_t2ongoing_q/queue_private.csv --out <new submission.csv>
  echo "stage build done: work/sub_t2ongoing/ongoing_*.csv"
fi
