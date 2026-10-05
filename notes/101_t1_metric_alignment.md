# 101. Training against the official score: the mismatch is real, the fix is not

Date 2026-09-24. Code `analysis/t1_metric_alignment.py` (diagnostic), `analysis/t1_m3_cnn.py` env flags
`M3_LOSSW` and `M3_GROUPW`, comparison `analysis/t1_metric_compare.py`. The user's proposal: every branch so far
has optimised a pooled RMSE, while the official score is a macro average - 5 families, 2 panels each, 3 regimes
each - so small panels and small regimes count as much as large ones.

## The diagnostic: two real mismatches

S_state's loss is `sum_g w_g [0.0216 RMSE_v,g + 0.000767 RMSE_q,g]` with **w_g = 1/30 for every (panel, regime)
group**, since every family has exactly two panels and three regimes. Training weights a group by its cell count.

**Group weighting.** The per-cell value the score implies varies by a factor of 70:

| panel | regime | cells | RMSE v | RMSE q | share of the score | share of the training MSE | implied per-cell weight |
|---|---|---|---|---|---|---|---|
| D12_I405_S | R1 | 19,268 | 0.956 | 35.17 | 3.32% | 0.29% | **6.84** |
| D12_I405_N | R1 | 62,720 | 0.712 | 20.71 | 2.18% | 0.37% | 2.82 |
| D12_I5_N | R3 | 510,717 | 1.008 | 24.33 | 2.82% | 4.64% | 0.24 |
| D7_I405_N | R3 | 811,975 | 1.580 | 35.14 | 4.26% | 16.31% | 0.10 |
| D7_I210_E | R3 | 981,072 | 1.437 | 30.07 | 3.78% | **15.14%** | **0.09** |

**Channel weighting.** S_state is LINEAR in RMSE, so a unit of squared error is worth `w_g a / (2 RMSE_g)`: the
better a channel gets, the more each further unit is worth. The metric-optimal split today is **0.873 / 0.127**
against `LOSS_W = (0.75, 0.25)`. And the origin is exact: **at a speed RMSE of 3.0 km/h the optimum is 0.750** -
the value the comment in the code cites. LOSS_W was right when it was set; the model's own improvement on speed
(3.0 -> 1.31) moved the optimum and the constant stayed behind. Speed now carries 54.3% of the official loss.

## Four training runs, all worse

Same seed, epochs, inputs, masks, architecture; only the loss changed.

| run | loss | S_state | speed RMSE | flow RMSE | 2031-01 | 2031-02 |
|---|---|---|---|---|---|---|
| **A** (production) | 0.75 / 0.25 | **0.9523** | **1.307** | **29.67** | 0.9518 | 0.9528 |
| F | 0.60 / 0.40 | 0.9513 | 1.347 | 29.74 | 0.9510 | 0.9516 |
| D2 | 0.873 / 0.127, loss scale matched to A | 0.9505 | 1.333 | 31.23 | 0.9502 | 0.9510 |
| D | 0.873 / 0.127 | 0.9503 | 1.337 | 31.28 | 0.9498 | 0.9509 |
| E | per (panel, regime) w_g / (n_g RMSE_g), per channel | 0.9502 | 1.360 | 31.22 | 0.9498 | 0.9507 |

- **More weight on speed makes speed worse** (1.307 -> 1.333), not just flow. That is not a trade-off, so the
  first suspicion was a confound: cutting the flow term shrinks the loss by 26% at these RMSEs, which lowers the
  effective learning rate under a fixed OneCycle schedule. D2 restores the scale and lands in the same place
  (0.9505 against 0.9503), so the confound is not the explanation.
- **More weight on flow also makes speed worse** (F: 1.347) while flow barely moves (29.74). Both directions from
  (0.75, 0.25) degrade speed, so the production weights sit at an empirical optimum that the metric's first-order
  calculus does not predict.

Caveat: D/E/D2/F are single seeds and A is the production checkpoint, so the 0.0010-0.0021 gaps are within seed
noise. The safe statement is that no variant beat A, not that each is exactly that much worse.

## Reading

The mismatch between the training objective and the macro score is real and measurable, but **the loss weights
are not a free dial**: the two channels share a trunk, and neither direction buys what the marginal calculus
says it should. None of the user's stop criteria (both months above A, +0.0003 on the 59 days, not one panel,
surviving a blend) is met. The training side of Task 1 is closed.
