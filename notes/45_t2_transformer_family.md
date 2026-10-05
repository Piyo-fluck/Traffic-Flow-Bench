# 45 — Task 2: a third model family (GRU + panel-wide transformer) beats the CNN on ongoing in every condition

Date: 2026-09-15 · Code: `analysis/t2_cnn.py` with `T2CNN_ARCH=tfm`, trained on the Ada machine (GPUs 0–2, seeds
0/1/2, 30k steps, about 17 min each). Remote logs were copied to `work/remote_logs/`, caches to
`work/cache/t2cnn_t2_cnn_lov_tfm*`, and models to `work/models_remote/`. Evaluation log: `work/t2_tfm_eval.log`.

## Model

- **Time:** a GRU over each link's 12-slot history turns it into one token.
- **Space:** a 4-layer transformer runs across all links of the panel. Each head has a learned bias per signed
  link distance (clipped at ±32), so upstream and downstream stay distinct. Width 64, 150k parameters.
- **Training:** whole panels (one panel per batch), not the CNN's 64-link crops.
- **Unchanged from the CNN:** the inputs (12 channels), labels, BCE + 0.5·Lovász, early stopping, and the cache format.

Single models, on their own (CNN lov seeds in brackets):
- ongoing: 0.78–0.81 (CNN 0.76–0.78);
- onset: 0.69–0.81 (CNN 0.62–0.74).

## Blends

| Δ vs LightGBM (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| ongoing 0.5·I + 0.5·CNN×3, τ 0.60 (M) | +0.035 | +0.034 | +0.035 | +0.036 |
| **ongoing 0.5·I + 0.5·TFM×3, τ 0.60** | **+0.045 (7/1)** | **+0.043 (8/0)** | **+0.046 (8/0)** | **+0.041 (8/0)** |
| ongoing 0.5·I + 0.5·(CNN×3 + TFM×3) | +0.042 | +0.043 | +0.036 | +0.041 |
| onset 0.7·H + 0.3·CNN×3 (M) | +0.009 | +0.014 | +0.023 | +0.020 |
| onset 0.7·H + 0.3·TFM×3 | −0.012 | −0.022 | +0.035 | −0.013 |
| onset 0.85·H + 0.15·TFM×3 | +0.017 | +0.007 | +0.012 | +0.010 |

In N's ongoing blend (global-state LightGBM g1 + neural; Δ vs M's ongoing; `t2_gs_blend.py`):

| | mean of 4 | min of 4 |
|---|---|---|
| g1 w 0.5 + CNN×3, τ 0.55 (candidate N) | +0.0126 | +0.0098 |
| **g1 w 0.5 + TFM×3, τ 0.55** | **+0.0188** | **+0.0170** |
| g1 w 0.4 + TFM×3, τ 0.50 | +0.0210 | +0.0157 |
| g1 w 0.4 + CNN×3 + TFM×3, τ 0.50 | +0.0205 | +0.0138 |

- **For ongoing, the transformer is the better partner in every condition.** It sees the whole panel, while the
  CNN sees only crops, which matches the global-state finding of note 43.
- Onset stays with the CNN: the transformer's onset blend flips sign between conditions.
- **Width 128** (GPU 3, seed 0, tag `t2_cnn_lov_tfm_w128`; log `work/t2_tfm_w128_eval.log`):
  - Alone in M's ongoing blend it gives +0.036…+0.040, the level of a single width-64 seed; width adds nothing.
  - As a 4th member it helps like any extra seed: ongoing +0.044…+0.046 in M's blend. In N's blend (g1 w 0.5, τ
    0.55) it gives +0.0205 (min +0.0182), against +0.0188 (min +0.0170) with 3 members.
  - More transformer seeds are worth about +0.002 ongoing.
- **Candidate O:** onset as M; ongoing = 0.5·g1 + 0.5·TFM×3 at τ 0.55. These settings were fixed for N before the
  transformer existed, and they give the highest minimum. Expected ongoing ≈ +0.019 vs M, which is ≈ +0.003 total
  (≈ 0.878).
- **Candidate O built and verified (not submitted):** `work/sub_O_ttt_t2gstfm_t4ref/submission.csv` (367 MB).
  - Built with `t2_cnn_blend_predict.py --cnn <CNN×3> --og-nets <TFM×3> --og-arch tfm --ongoing-probs-dir
    work/sub_t2gs --w-on 0.7 --w-og 0.5 --tau 0.55`, then swapped into L. Log `work/t2_blend_O.log`.
  - Only `queue_pred` differs: 630 cells vs M and 445 vs N. No NaN; every other column is identical.
- Seeds 3–6 of the transformer are training on Ada (4 GPUs), for a 7-member ensemble (log `work/t2_tfm7_eval.log`).
