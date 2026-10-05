# 32 — Task 1: test-time training on the scored month gives a small, consistent gain (+0.001 total)

Date: 2026-09-15 · Script: `analysis/t1_ttt.py` · Log: `work/t1_ttt_validation.log`

Each of the 3 production CNNs was fine-tuned self-supervised on the validation month's visible eligible cells:
10–50% of cells hidden per sample, 400 steps, lr 2e-4, ens3 + month base. T1 is offline, so this is allowed.
The evaluation is the in-month pseudo-eval with the same seed and panel order as t1_pseudo_phys (20% of visible
cells hidden). Those cells are also removed from the TTT input.

| validation | S_state~ | S_LWR~ |
|---|---|---|
| ens3 + month base | 0.9494 | 0.7702 |
| + TTT | **0.9513 (+0.0019)** | **0.7735 (+0.0033)** |

- S_state~ improves on 10/10 panels. S_LWR~ improves on 8/10, and the two losses are ≤ 0.001.
- Total ≈ +0.001. It is small but consistent, and more steps might add a little.
- Adopting it needs production work: TTT on validation and private with every visible cell, then T1 inference
  for 10 panels, then a swap of the state rows. It is not in candidate K.

## More steps (2026-09-16, the +0.01 stacking route of note 55; log `work/t1_ttt_validation_1200.log`)

`TTT_STEPS` and `TTT_LR` are environment overrides, so the same script measures longer runs (6 min on the 4090).

| validation, in-month pseudo eval | S_state~ | S_LWR~ | total |
|---|---|---|---|
| ens3 + month base | 0.9494 | 0.7702 | — |
| + TTT 400 steps (production, candidate L) | 0.9513 (+0.0019) | 0.7735 (+0.0033) | ≈ +0.0010 |
| **+ TTT 1200 steps** | **0.9518 (+0.0024)** | **0.7745 (+0.0042)** | **≈ +0.0013** |

- S_state~ improves on **10/10 panels** at 1200 steps, and S_LWR~ on 10/10 as well (it was 8/10 at 400).
- The biggest movers are D12_I5_S (+0.0061 state, +0.0092 LWR) and D12_I5_N (+0.0023, +0.0101).
- Tripling the steps buys only +0.0003 total over the production setting, i.e. the curve is flattening. Worth
  taking, since the cost is one 6-minute run per month, but it is not the +0.002…+0.003 that note 55 hoped for.

**3600 steps (log `work/t1_ttt_validation_3600.log`, 18 min): saturated.** S_state~ 0.9494 → 0.9521 (+0.0027) and
S_LWR~ 0.7702 → 0.7739 (+0.0037), i.e. **total ≈ +0.0013, the same as 1200 steps** (state marginally better, LWR
marginally worse). Tripling the compute again buys nothing.

**Conclusion: 1200 steps is the operating point, and the whole test-time-training line is worth +0.0003 over the
400-step version already inside candidates L/M/R.** Adopting it needs a TTT rerun per month, T1 inference for 10
panels and a state-row swap; that is not worth +0.0003, so it stays unused.

## Production (candidate L = K + TTT T1; user decision 2026-09-15)

- `scripts/t1_ttt_train.py --split <s> --out-dir work/sub_m3ttt` uses the same recipe with every visible cell and
  the inputs of `t1_predict_ckpt.py` (month base κ 2 / 10). It writes `ttt_<split>_<i>.pt`.
- `scripts/run_submission_L.ps1`:
  1. TTT per month.
  2. Per-panel inference with each month's own checkpoints (retries against the Windows native crash).
  3. `t1_npz_to_state.py`.
  4. `swap_task_rows.py --task state` into K, giving `work/sub_L_ttt_t2K_t4ref/submission.csv`.
  5. A check that only speed/flow differ from K.
