# 79 - Plan B': calibrating the blend on the scored month's own past days does not pay

Date: 2026-09-20. Script: `analysis/t2_blend_calibrate.py`. Motivated by the literature (OneNet-style online
ensembling, black-box error correction), which prefers calibrating or re-weighting over fine-tuning under drift.

Setup: the blend constants of candidate T (onset 0.7 LightGBM + 0.3 adapted CNN with ratio c = 1.0; ongoing
0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 at tau 0.50) were chosen on the train run. Here, for each official-like window on
day d, the constants are re-chosen on every window of the same split on days < d, pooled over panels (expanding
window, causal). No new inference: the cached probabilities already cover every day.

| split / condition | fixed (candidate T) | causal past-day calibration | oracle (whole month, illegal) |
|---|---|---|---|
| validation onset | 0.7862 | **-0.0014** | +0.0173 |
| validation ongoing | 0.8357 | -0.0004 | +0.0002 |
| private onset | 0.8227 | +0.0024 | +0.0239 |
| private ongoing | 0.8233 | **-0.0046** | +0.0000 |

- **Onset**: the oracle is worth +0.017 / +0.024 family, but the causal choice captures none of it. The picked
  constants jump from day to day ((0.6, 0.8) 7 days, (0.7, 1.0) 7 days, (0.6, 1.1) 6 days on validation): with
  about 8 onset windows per day, the past-day estimate is far too noisy for a surface this flat.
- **Ongoing**: the oracle is worth +0.0002 / +0.0000, i.e. the production constants are already optimal for the
  month; the causal search only adds noise (-0.0046 on private, window CI [-0.0048, -0.0019]).
- So the run-specific part of these constants is either tiny (ongoing) or unlearnable from a month of past days
  (onset). Plan B' is negative.
