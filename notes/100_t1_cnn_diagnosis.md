# 100. What the Task 1 CNN succeeds at, and what it fails at

Date 2026-09-24. Code `analysis/t1_cnn_diagnosis.py`, `analysis/t1_residual_scale_bias.py`. Measured on the saved
production predictions over the local test block (train days 2031-01/02, the real official mask, 10 panels,
7,089,289 targets).

## Succeeds: skill over every reference, and optimal calibration

Flow RMSE, per lane:

| group | cells | CNN | profile | time interpolation | CNN gain over the best reference |
|---|---|---|---|---|---|
| all | 7,089,289 | **29.67** | 100.82 | 52.99 | -23.33 |
| free flow | 6,779,974 | 28.87 | 96.33 | 51.18 | -22.32 |
| queued | 296,037 | 42.66 | 157.27 | 76.39 | -33.73 |
| peak hours | 1,770,611 | 35.04 | 120.47 | 58.21 | -23.17 |
| night | 2,069,657 | 13.83 | 54.46 | 38.21 | -24.38 |

Regressing the predicted deviation from the profile on the true deviation:

| group | flow slope | flow corr | speed slope | speed corr |
|---|---|---|---|---|
| all | 0.910 | 0.956 | 0.986 | 0.994 |
| free flow | 0.906 | 0.954 | 0.976 | 0.989 |
| queued | 0.925 | 0.963 | 0.996 | 0.999 |
| night | 0.931 | 0.967 | 0.810 | 0.898 |

The slope looks like shrinkage towards the profile, but **it is the MSE-optimal amount**: corr^2 = 0.914 against
a predicted/true variance ratio of 0.906, and the same identity holds in every group including night speed
(0.806 against 0.814). A predictor whose variance ratio equals its squared correlation has a residual orthogonal
to its prediction - there is nothing to gain from rescaling, recalibrating or de-biasing. R^2 on the
profile-deviation is 0.914 for flow and 0.988 for speed.

## Fails: the level-proportional high-frequency component

Residual correlation between neighbouring target cells: **+0.043** at t+1, +0.025 at t+2, **+0.046** on the
adjacent link (about 2.1M pairs each). Essentially independent.

| flow level | cells | flow RMSE | relative |
|---|---|---|---|
| 0-400 | 1,421,241 | 7.66 | 3.8% |
| 400-800 | 1,267,576 | 18.23 | 3.0% |
| 800-1200 | 2,348,269 | 29.78 | 3.0% |
| 1200-1800 | 1,927,813 | 40.05 | 2.7% |
| 1800+ | 124,390 | 67.14 | 3.4% |

| local volatility | cells | flow RMSE |
|---|---|---|
| 0-20 | 2,808,893 | 18.62 |
| 20-40 | 1,456,782 | 25.94 |
| 40-80 | 1,322,943 | 35.85 |
| 80-160 | 480,411 | 53.56 |
| 160+ | 67,053 | 74.00 |

**The CNN captures the smooth profile-relative structure and misses the fluctuation, whose size is about 3% of
the flow level and which neighbouring cells do not share.**

Checked and ruled out along the way: occupancy is blanked at Task 1 targets too (finite fraction 0.000 there
against 1.000 at non-targets), so the model is not throwing away a visible density measurement.

## Multiplicative noise or multiplicative structure?

Linear calibration only says E[r q_hat] = 0; it cannot see E[r | q_hat = 800]. A LightGBM fitted to the residual
from scoring-time features only (q_hat, base_q, v_hat, base_v, observed-layer volatility, tod, link, lanes,
observed rate, occupancy of the day), out of fold by day in three folds:

| flow bin | cells | RMSE | corrected | delta | mean residual |
|---|---|---|---|---|---|
| 0-400 | 1,420,573 | 7.63 | 7.63 | -0.008 | +0.201 |
| 400-800 | 1,262,314 | 18.01 | 18.00 | -0.013 | +0.157 |
| 800-1200 | 2,350,080 | 29.71 | 29.64 | -0.075 | +0.154 |
| 1200-1800 | 1,934,553 | 40.24 | 40.04 | -0.201 | -0.164 |
| 1800+ | 121,769 | 66.59 | 66.35 | -0.240 | +0.022 |
| **overall** | 7,089,289 | **29.668** | **29.557** | **-0.111** | |

A scale-dependent mean bias exists and sits exactly where it was predicted to - the high-flow bins - but it is
worth 0.111 veh/h/lane overall, 0.4% of the error, or about +0.00003 on the total score. The corrector's own
output has sd 1.99 against a residual sd of 29.7. The user's decision threshold for taking the ratio
parameterisation seriously was a move to about 29.2; the measurement is 29.56.

So the level dependence is **mostly variance, not mean**: Var(q | X) grows with the level, E[q | X] is nearly
right. What remains for a ratio parameterisation is the optimisation / capacity argument, which this test cannot
measure.

## The ratio parameterisation: tested and closed

Code: `analysis/t1_m3_cnn.py` env flags `M3_SCALE` (default 0, so production is untouched) and `M3_SCALECH`;
comparison `analysis/t1_param_compare.py`. Backup of the original file at `work/t1_m3_cnn.py.bak_scale`.

    A   the production additive target (q - base) / lanes / 100                     not retrained, evaluated from
                                                                                     its saved checkpoint
    B   z = (q - base) / lanes / s(base),  s = max(base per lane, 400)
        q_hat = base + s * lanes * z, and the loss multiplied back by s / 100 so the objective is numerically
        identical to A - only the gradient geometry and the capacity allocation change
    C   B plus explicit scale channels, log1p(base per lane) and log1p(observed flow per lane)

Same seed, same 25 epochs, same 10 panels, same everything else.

| group | A | B | C | B - A | C - A |
|---|---|---|---|---|---|
| all | **29.668** | 30.486 | 30.432 | +0.818 | +0.764 |
| 2031-01 | 29.831 | 30.672 | 30.622 | +0.841 | +0.791 |
| 2031-02 | 29.494 | 30.287 | 30.230 | +0.794 | +0.736 |
| 0-400 | 7.661 | 8.171 | 8.171 | +0.510 | +0.510 |
| 400-800 | 18.227 | 18.894 | 18.893 | +0.667 | +0.666 |
| 800-1200 | 29.783 | 30.550 | 30.471 | +0.767 | +0.688 |
| 1200-1800 | 40.046 | 41.106 | 41.019 | +1.060 | +0.972 |
| 1800+ | 67.135 | 68.993 | 69.120 | +1.857 | +1.985 |

S_state: A 0.9523, B 0.9504, C 0.9507. Blending with A is worse from alpha 0.2 upwards (29.691 / 29.683).

**Every stop criterion fails, and the loss is largest exactly where the change was supposed to help** - the
high-flow bins. The likely mechanism is resolution rather than gradient size: at high flow s is large, so the
head must emit a very small z to the same absolute accuracy, while at low flow s is pinned at b0 = 400 against
A's 100, making the target four times smaller there (+0.51). Branch closed.
