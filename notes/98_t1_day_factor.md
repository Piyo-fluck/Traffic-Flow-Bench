# 98. Is there a day-specific demand bias left in the Task 1 CNN residual? No.

Date 2026-09-24. Code `analysis/t1_day_factor.py`. The user's proposal after note 97: keep the CNN, and ask
whether its residual still carries a state that belongs to THE DAY - "today runs 3% heavy, 6% heavy in the
evening" - which a fixed forward pass cannot re-estimate. Task 1 is offline, so the day's own observed cells may
be used to estimate it.

## Protocol (probes, so the targets are never touched)

The CNN copies observed cells, so its residual there is meaningless. Instead, for each test day (2031-01/02):
hide a small share of the visible eligible cells, predict them, and take `e = observed - predicted` on those
probes only. From `e`, estimate a correction - a scalar per day, or per day x hour - and apply it to the REAL
targets. The probe rate is kept at 5% so the input stays close to the real one; at the full regime rate the
input's missingness doubles and the probe residual stops representing the targets (measured: it did, the
correction then hurt by 0.06).

## Result (8 panels, 617,313 probes)

Measured day-level flow bias: **mean +0.069, mean |bias| 0.689 veh/h/lane**, against a free-flow flow RMSE of
29.13 - 2.4% of the error scale.

| correction | speed RMSE | flow RMSE | flow RMSE, free flow | change |
|---|---|---|---|---|
| CNN | 1.268 | 29.95 | **29.13** | - |
| + daily scalar | 1.268 | 29.96 | 29.14 | +0.007 |
| + daily x hour (24 bins) | 1.271 | 30.05 | 29.22 | +0.094 |
| + daily scalar, flow only | 1.268 | 29.96 | 29.14 | +0.007 |

The agreed stop criterion was 0.3 veh/h/lane; the measurement is +0.007, i.e. nothing, and the hourly version is
worse because the estimate's noise exceeds the effect. **Hypothesis dropped**, and no rank-K factor was built.

The CNN already receives the month-corrected profile as input channels and sees the whole day's observed map, so
a level shift of the day is something it can read off its own input.

## What the three negative results together say about Task 1

Errors do not separate by mask geometry (note 96), are not reachable from corridor conservation (note 96), and
carry no day-level bias (here). 90% of the mass is free-flow cells at 29 veh/h/lane. **The residual is
cell-specific and high-frequency, not a systematic bias** - closer in character to measurement noise than to a
missing model term, which also puts a question mark on note 01's flow noise floor of 18.
