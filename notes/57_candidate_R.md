# 57 — Candidate R: M with the note-56 ongoing ensemble (built and verified, not submitted)

Date: 2026-09-16 · Files: `work/sub_t2R/queue_{validation,private}.csv`, `work/sub_R_ttt_t2ens_t4ref/submission.csv`
· Logs: `work/t2R_predict.log`, `work/sub_R_swap.log`

R = M with **only the ongoing half of Task 2 replaced** by the ensemble selected in note 56. Everything else
(Task 1 with test-time training, Task 3, Task 4, and M's onset rule) is untouched.

| part | M | **R** |
|---|---|---|
| T2 onset | 0.7·LightGBM(H) + 0.3·CNN×3, expected-IoU ratio rule | **unchanged** |
| T2 ongoing | 0.5·LightGBM(I) + 0.5·CNN×3, τ 0.60 | **0.25·gs1 + 0.50·transformer×4 (s3–s6) + 0.25·CNN×3, τ 0.45** |

Build (the predict script gained `--og-nets2 / --og-arch2 / --w-og2`; the old two-term behaviour is the default):

```bash
.venv/Scripts/python.exe scripts/t2_cnn_blend_predict.py \
  --cnn work/models_remote/t2_cnn_lov.pt work/models_remote/t2_cnn_lov_s1.pt work/models_remote/t2_cnn_lov_s2.pt \
  --og-nets work/models_remote/t2_cnn_lov.pt work/models_remote/t2_cnn_lov_s1.pt work/models_remote/t2_cnn_lov_s2.pt --og-arch cnn \
  --og-nets2 work/models_remote/t2_cnn_lov_tfm_s3.pt work/models_remote/t2_cnn_lov_tfm_s4.pt \
             work/models_remote/t2_cnn_lov_tfm_s5.pt work/models_remote/t2_cnn_lov_tfm_s6.pt --og-arch2 tfm --w-og2 0.50 \
  --onset-probs-dir work/sub_t2robust_bag5 --ongoing-probs-dir work/sub_t2gs \
  --w-on 0.7 --w-og 0.25 --tau 0.45 --out-dir work/sub_t2R
.venv/Scripts/python.exe scripts/swap_task_rows.py --base work/sub_M_ttt_t2cnnblend_t4ref/submission.csv \
  --task queue --files work/sub_t2R/queue_validation.csv work/sub_t2R/queue_private.csv \
  --out work/sub_R_ttt_t2ens_t4ref/submission.csv
```

## Verification (all three checks pass)

1. **Only ongoing windows changed.** Differing windows by condition: validation 32, private 34, **all
   `queue_ongoing`** (each split holds 40 onset + 40 ongoing windows). No onset window moved, which confirms the
   predict-script changes did not touch the onset path.
2. **Row keys aligned**: 87,000 rows per split, `window_id`/`timestamp`/`link_id` identical to M's queue files;
   415 → 435 (val) and 280 (priv) differing predictions.
3. **The assembled upload differs from M only in queue rows**: 6,980,503 rows compared chunk by chunk,
   **715 differing rows, all `task == queue`** (435 + 280, exactly the queue-file diff). `speed_kmh`, `flow_vph`
   and `path_flow` are bit-identical. Share queued 0.0604 → 0.0618.

## Expected score

The ongoing family mean (the quantity S_queue uses) rises 0.8057 → 0.8327 on validation and 0.7928 → 0.8216 on
private (note 56). Δtotal = 0.15 × Δongoing:

| | Δongoing (family) | Δtotal |
|---|---|---|
| validation (= the public LB month) | +0.0270 | **+0.0041** |
| private | +0.0288 | +0.0043 |

**Expected public LB ≈ 0.87505 + 0.004 ≈ 0.879.** The recipe's out-of-sample credential is note 56's cross-split
test: fitted on private, applied to validation, +0.0294 ongoing.

## Not included (available to stack later)

| item | Δtotal | why not in R |
|---|---|---|
| onset train-template shape prior (note 51) | +0.0015 | needs a change to the onset decision path; CI crosses 0 in one split |
| T1 test-time training at 1200 steps (note 32 appendix) | +0.0003 over M's 400 steps | needs a TTT rerun per month plus T1 inference and a state-row swap |
| shockwave features (note 54) | 0 | redundant inside this ensemble (note 54 appendix) |
| onset ensemble (note 56) | 0 | overfits on 169/162 windows |
