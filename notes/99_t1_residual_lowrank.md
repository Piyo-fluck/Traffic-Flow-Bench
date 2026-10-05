# 99. Residual-LATC: is the Task 1 CNN residual low-rank over (day x TOD x link)? The ceiling is zero.

Date 2026-09-24. Code `analysis/t1_residual_lowrank.py`. The user's proposal after the ImputeFormer result: keep
the CNN and model its residual with a low-rank (day, time-of-day, link) tensor, in the spirit of LATC / MLATC.
Note 98 had only ruled out MEAN corrections per day and per day x hour; a factor that is positive on some links
and negative on others averages to zero and would survive that test.

## The measurement

The residual tensor `R = (q_true - q_CNN) / lanes` on the local test block (2031-01/02, the real official mask),
defined on target cells only - at observed cells the CNN copies its input, so the residual there is meaningless.
CP factors fitted by alternating ridge least squares on a **random half of the target cells** and scored on the
other half. That is far more information than a scored month provides (there, only 5% probes exist, note 98), so
it is a ceiling, not a method.

| model | flow RMSE, free flow | change |
|---|---|---|
| no correction | 28.85 | - |
| per-day mean | 28.85 | -0.003 |
| per-TOD mean | 28.86 | +0.005 |
| per-link mean | 28.85 | -0.000 |
| rank 1, strong shrinkage | 28.85 | **+0.000** |
| rank 2, strong shrinkage | 28.85 | +0.000 |
| rank 1, weak shrinkage | 29.32 | +0.466 |
| rank 2, weak shrinkage | 29.96 | +1.109 |
| rank 4, weak shrinkage | 31.79 | +2.933 |

With weak shrinkage the fit does find in-sample structure - 28.895 -> 28.478, 2.9% of the variance with ~700
parameters against 440k observations, well above chance - **but none of it transfers to held-out cells of the
same tensor**; the held-out error gets worse. With strong shrinkage the factors collapse to zero and the
correction does nothing. Across the whole regularisation range the best achievable held-out effect is 0.000.

## Reading

There is no (day x TOD x link) systematic component left in the residual, even when half the answers are handed
to the estimator. Together with notes 96 and 98 - no dependence on mask geometry, nothing reachable from corridor
conservation, no day-level bias - the Task 1 residual behaves as cell-specific noise, not as a missing model
term. Branch closed without building the model.
