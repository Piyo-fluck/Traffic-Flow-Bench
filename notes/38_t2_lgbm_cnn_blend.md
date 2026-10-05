# 38 — Task 2: blending LightGBM with a 3-seed Lovász CNN improves both onset and ongoing in every condition

Date: 2026-09-15 · Scripts: `analysis/t2_cnn.py` (Ada machine), `analysis/t2_blend_eval.py`,
production `scripts/t2_cnn_blend_predict.py`

CNN setup: T2Net, width 64, no prior channels; BCE + 0.5 × Lovász hinge; 30k steps; seeds 0, 1, 2
(`t2_cnn_lov*.pt`). The probabilities are cached per window. The evaluation uses the blank-aware official-like
windows against K's LightGBM (onset H with the expected-IoU ratio rule; ongoing I with τ 0.65).

| Δ IoU vs LightGBM (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| **ongoing: 0.5·LGBM + 0.5·CNN×3, τ 0.60** | **+0.035 (7/1)** | **+0.034 (8/0)** | **+0.035 (7/1)** | **+0.036 (8/0)** |
| ongoing: 0.5/0.5, τ 0.65 | +0.024 | +0.024 | — | +0.024 |
| **onset: 0.7·LGBM + 0.3·CNN×3, expected-IoU ratio** | **+0.009 (3/1)** | **+0.014 (6/1)** | **+0.023 (3/0)** | **+0.020 (5/1)** |
| onset: 0.85/0.15 | +0.008 | +0.010 | +0.012 | +0.013 |
| (1 seed only) ongoing 0.6/0.4, τ 0.60 | +0.028 | +0.031 | +0.024 | +0.031 |

- **Averaging 3 seeds helps**: ongoing goes from +0.024…+0.031 with 1 seed to +0.034…+0.036. Adding the plain CNN
  as a 4th model changes nothing.
- The weights and τ are chosen on validation. The same settings are also the best on private, a separate run.
- The CNN alone is weaker on onset, but at 30% weight it improves onset everywhere. The earlier 50/50 blend
  hurt (note 36).
- **Estimate**: ongoing about +0.035 and onset about +0.017, so S_queue about +0.026, i.e. **about +0.008 total
  over L (≈ 0.877)**.
- **LB: M = 0.87505** (ref 56245044, 04:16 UTC 2026-09-15). That is +0.0061 over L (0.86893); the estimate was ≈ 0.877.
  M differs from L only in 685 T2 cells.
- Production: `t2_cnn_blend_predict.py --cnn t2_cnn_lov{,_s1,_s2}.pt --w-on 0.7 --w-og 0.5 --tau 0.6`, with onset
  probs from `work/sub_t2robust_bag5` and ongoing probs from `work/sub_t2robust_bag5_stress`. It writes
  `work/sub_t2M`, which becomes candidate M = L with these T2 rows.
