# 140 - Decision-layer methods on AU (time-lagged ensemble, label shift, online recalibration, per-window selection)

`occ_lab/decision_c_eval.py` (all causal within the run; labels only from earlier days' lab windows);
lagged forecasts from `occ_lab/adapt_variants.py --method lagged` (the four p_PS seeds, forecasts issued at T-5 / T-10).

| ongoing family IoU vs AU (val / priv) | all windows | days 1-5 |
|---|---|---|
| prior shift, supervised (earlier days' queued share / AU's mean) | -0.0000 / -0.0000 | -0.0005 / +0.0001 |
| prior shift, EM (unsupervised) | -0.048 / -0.029 | -0.011 / -0.020 |
| online Platt a, b_h (ridge 1e3 / 1e4) | +0.0002 / -0.0002, 0 / 0 | -0.0003 / +0.0002 |
| per-window member selection, beta 5 / 20 | +0.0007 / +0.0005, **+0.0010 / +0.0010** | +0.0004 / +0.0004, -0.0001 / +0.0007 |
| time-lagged ensemble in the p_PS slot, w 0.2 / 0.4 | -0.0010 / -0.0007, -0.0012 / -0.0019 | +0.0008 / -0.0017, +0.0007 / -0.0039 |

- The lagged forecasts are far weaker than the current one (single p_PS s20: 0.850 -> 0.718 issued 5 min earlier, 0.583
  10 min earlier): the last 5-minute slot carries most of the information, so a lagged ensemble only dilutes.
- AU is already calibrated for the run (supervised prior and Platt corrections stay at 0); the unsupervised EM prior
  breaks it (the label-shift assumption p(x|y) fixed does not hold).
- Per-window selection by competence on similar earlier windows is the only consistent positive, +0.0010 on all
  windows of both months, +0.0003 on days 1-5 (about +0.0001 total).
