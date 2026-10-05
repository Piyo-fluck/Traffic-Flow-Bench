# 43 — Task 2: ongoing τ 0.50 beats 0.60 in both runs; onset recalibration fails; global-state features (pending)

Date: 2026-09-15 · Scripts: `analysis/t2_decision_cal.py` (caches only; log `work/t2_decision_cal.log`),
`analysis/t2_gstate.py` (log `work/t2_gstate.log`)

The CNN probability caches were regenerated locally in eval-only mode from `work/models_remote/t2_cnn_lov*.pt`
(log `work/t2_cnn_evalonly.log`). The LightGBM baselines match note 41 exactly (onset 0.793 / 0.765 / 0.824 / 0.798).
**Interpreter:** `py` can no longer import pandas, so everything runs with `.venv\Scripts\python.exe`.

## Thresholds and calibration on M's blend (note 42, items 6–7)

Each variant is fitted on one run and scored on the other, in both directions.

| Δ IoU vs M (panels better/worse) | fit val → priv d1–14 | priv all | fit priv → val d1–14 | val all |
|---|---|---|---|---|
| ongoing: one τ per horizon (≈ 0.50 except T+5) | +0.003 (7/1) | +0.006 (7/1) | +0.012 (7/1) | +0.011 (7/1) |
| ongoing: one τ per family (0.45–0.55) | +0.004 (7/1) | +0.008 (8/0) | +0.012 (6/2) | +0.012 (7/1) |
| ongoing: logit-space blend, τ 0.55 | +0.002 | +0.007 (8/0) | +0.011 | +0.012 |
| onset: CNN recalibrated sigmoid(a·logit + b), with w and c | −0.010 | −0.004 | −0.001 | −0.007 |
| onset: the same, logit-space blend | −0.023 | −0.009 | −0.001 | −0.004 |

Scan of one global τ on the ongoing blend, 0.5·LGBM + 0.5·CNN×3 (family mean):

| τ | 0.45 | **0.50** | 0.55 | 0.60 (M) | 0.65 |
|---|---|---|---|---|---|
| val d1–14 / all | 0.816 / 0.814 | **0.820 / 0.817** | 0.815 / 0.814 | 0.809 / 0.806 | 0.798 / 0.796 |
| priv d1–14 / all | 0.814 / 0.794 | **0.825 / 0.800** | 0.823 / 0.798 | 0.823 / 0.793 | 0.810 / 0.782 |

- **τ 0.50 is the best single value in both runs and both day sets.** Against M's 0.60 it gains +0.011 on val and
  +0.002 to +0.007 on priv, so ongoing improves by about +0.008 and the total by about +0.001.
- It needs no retraining. The per-horizon and per-family fits add nothing beyond this single τ.
- Onset recalibration does not transfer. The ratio rule on the raw 0.7/0.3 blend stays.

## Global network-state features (note 42, item 5; Traffic4cast 2022 2nd place)

Twelve panel-wide features computed from each window's own history (queued share, mean v/vf and q/cap with their
trends, the number of queue heads, and 5 PCs of the v/vf profile; PCA fitted on train truth). Paired LightGBM runs:
onset in H's configuration, ongoing in I's. Log `work/t2_gstate.log` (1565 s).

| IoU base → +global (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| **ongoing** | 0.773 → 0.795 **(+0.022; 7/1)** | 0.771 → 0.794 **(+0.022; 7/1)** | 0.799 → 0.821 **(+0.022; 8/0)** | 0.758 → 0.778 **(+0.020; 8/0)** |
| onset | 0.793 → 0.797 (+0.005; 2/1) | 0.765 → 0.769 (+0.004; 3/2) | 0.824 → 0.825 (+0.000; 1/2) | 0.798 → 0.776 (−0.022; 1/5) |

- The global features take 10% of the gain in both models.
- **Ongoing gains +0.020…+0.022 in all 4 conditions**, the largest single LightGBM gain since the stress features.
  This matches the T4c finding that the panel-wide state is a strong, shift-robust signal. The CNN sees only 64-link
  crops, so it has no such view either.
- Onset is mixed (priv all −0.022), so it is not adopted for onset.
- The re-run (log `work/t2_gstate_v2.log`) reproduces these numbers exactly. It also saves the PCA and the
  per-window probabilities (`work/models_gstate/`, `work/cache/t2gs_*`).

### In the blend with the CNN (`analysis/t2_gs_blend.py`, log `work/t2_gs_blend.log`)

| ongoing Δ vs M (0.5·I + 0.5·CNN, τ 0.60) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| **0.5·LGBM+global + 0.5·CNN, τ 0.55** | **+0.014 (6/2)** | **+0.015 (8/0)** | **+0.010 (6/1)** | **+0.012 (7/1)** |
| 0.4·LGBM+global + 0.6·CNN, τ 0.50 | +0.015 (7/1) | +0.016 (8/0) | +0.009 (5/3) | +0.012 (7/1) |
| M at τ 0.50 | +0.011 | +0.012 | +0.003 | +0.007 |
| control: 0.5·re-trained base + 0.5·CNN, τ 0.55 | +0.006 | +0.006 | +0.000 | +0.004 |

- About half of the LightGBM-alone gain survives the blend. Against the paired control, the global features add
  about +0.009 inside the blend.
- **Chosen: w 0.5, τ 0.55.** It has the highest minimum over the 4 conditions (+0.010). Ongoing gains about
  +0.013, which is about +0.002 total over M.
- Production probabilities look healthy. Against I's on the official windows, the correlation is 0.93–0.997 per
  panel and the means are nearly equal.
- **Note:** the official release has only **5 ongoing windows per panel and split**, against about 370 official-like
  pseudo windows. The LB's ongoing half is therefore much noisier than our estimate.
- **Candidate N built and verified (not submitted):** `work/sub_N_ttt_t2gs_t4ref/submission.csv` (367 MB).
  - N = L + T2 from `work/sub_t2N`: onset as in M (0.7·H + 0.3·CNN×3, ratio rule); ongoing 0.5·LGBM+global +
    0.5·CNN×3 at τ 0.55.
  - Against M only `queue_pred` differs: 347 cells (161 val, 186 priv), in ongoing windows only. No NaN, and every
    other column is identical.
  - Estimate ≈ M + 0.002 ≈ 0.877. The LB's ongoing half has only 5 windows per panel, so the realised change is noisy.
- Production: `scripts/t2_gstate_probs.py` → `work/sub_t2gs`.

## CNN v2: winners' training tricks (note 42, items 1, 2 and 4), seed 0, 30k steps, trained locally

Setup: `T2CNN_LOVASZ=0.5`, plus
- A: `T2CNN_AUX=0.5` (future v/vf Huber head) and `T2CNN_EMA=0.999`;
- B: A plus `T2CNN_EMBED=8` (link identity, 50% embedding dropout).

Logs: `work/t2_cnn_v2A.log`, `work/t2_cnn_v2B.log`. Blends from `analysis/t2_blend_eval.py` (vs LightGBM I/H).

| Δ vs LightGBM, blend (ongoing 0.5/0.5 τ 0.60 · onset 0.7/0.3) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| single lov s0 (production seed): ongoing · onset | +0.024 · +0.005 | +0.028 · −0.004 | +0.022 · +0.033 | +0.029 · +0.001 |
| single v2 A: ongoing · onset | +0.029 · +0.005 | +0.032 · −0.001 | +0.020 · +0.010 | +0.026 · −0.008 |
| single v2 B: ongoing · onset | +0.030 · −0.002 | +0.030 · +0.001 | +0.020 · +0.026 | +0.028 · +0.010 |
| 3 production seeds (M) | +0.035 · +0.009 | +0.034 · +0.014 | +0.035 · +0.023 | +0.036 · +0.020 |
| 3 seeds + v2 B | +0.035 · +0.009 | +0.036 · +0.018 | +0.030 · +0.023 | +0.037 · +0.020 |

- Neither v2 moves beyond seed-to-seed noise: ±0.005 against the 3-seed ensemble, in both directions. Multi-task
  heads, EMA and link embeddings are **not adopted**.
- Run C: the CNN plus the 12 global-state channels (`T2CNN_GLOBAL=1`), log `work/t2_cnn_v2C.log`.
  - Alone, ongoing is 0.015–0.017 below LightGBM, where lov s0 is about level with it.
  - In M's blend it gives +0.022…+0.027, slightly below lov s0.
  - Added to N's blend (`t2_gs_blend.py` with 4 CNNs), ongoing falls from +0.0126 to +0.0106.
  - The global view belongs in LightGBM; in the CNN it only duplicates that signal. **Not adopted.**
- **Conclusion: the CNN is at its seed-noise level.** Of the winners' tricks, only the global state (in LightGBM)
  transferred. Candidate N (≈ M + 0.002) holds everything useful found in this round. Then run `t2_cnn_blend_predict.py` with
  `--ongoing-probs-dir work/sub_t2gs --w-og 0.5 --tau 0.55`.
