# 114. The learning curve has not saturated: data is the binding constraint, and most of what it buys is S_LWR

Date 2026-09-24. Code `analysis/t1_train_test_gap.py`, `t1_capacity_probe.py`, `M3_NDAYS` in `t1_m3_cnn.py`.

Every note from 96 to 113 measured the model and concluded the residual cannot be reduced. All of them measured
the test side. Three things were never measured: how well the model fits the days it trained on, whether the
architecture can fit at all, and whether test error still improves with more days.

## 1. Train and test are nearly equal

Same checkpoint, same cells, 183 training days against 59 held-out days:

| days | S_state | RMSE v | RMSE q | sd(dk/k) |
|---|---|---|---|---|
| **trained on** | 0.95499 | 1.0903 | 120.413 | **0.02984** |
| held out | 0.95090 | 1.2661 | 124.686 | 0.03191 |

The model leaves 94% of the density error on data it saw 25 times. It is not memorising - but this alone cannot
separate underfitting from having reached the Bayes error.

## 2. The architecture can fit

One panel, 8 days, no extra hiding, no early stopping (`t1_capacity_probe.py`):

| epoch | RMSE v | RMSE q | sd(dk/k) |
|---|---|---|---|
| 60 | 0.845 | 83.1 | 0.02623 |
| 180 | 0.433 | 35.9 | 0.01514 |
| 280 | **0.340** | **30.1** | **0.01273** (still falling) |

Against production's 1.09 / 120.4 / 0.0298 on its own training days. Representation is not the wall.

## 3. The learning curve is still in its linear regime

Same optimisation budget (25 epochs), only the number of fit days per panel changes:

| train days | S_state | S_FD | S_LWR | 0.35 s + 0.05 f + 0.10 l |
|---|---|---|---|---|
| 20 | 0.94782 | 0.98728 | 0.61055 | 0.44216 |
| 45 | 0.94962 | 0.98739 | 0.62526 | 0.44426 |
| 90 | 0.95019 | 0.98744 | 0.63078 | 0.44502 |
| 183 | 0.95091 | 0.98749 | 0.63641 | 0.44583 |

Per doubling: 45->90 gives S_LWR **+0.0055**, 90->183 gives **+0.0056**. **The last doubling pays exactly as
much as the one before it** - log-linear, no sign of flattening.

And the split matters: from 90 to 183, S_state gains 0.0007 (0.00025 of the total) while S_LWR gains 0.0056
(0.00056). **More than half of what extra data buys goes to S_LWR**, which is the score we cannot otherwise move.

## Reading

Notes 96-113 are not wrong, but their scope was narrower than it read: **the residual resists every change to the
model, and responds to data.** Nine directions of model-side work returned nothing; the first data-side
measurement returns a clean log-linear curve.

We have 183 fit days and cannot get more. What we can get is more (input, target) pairs from them: `MASK_SPEC` is
a deterministic hash, the train month carries the full truth, and 76.4% of cells are eligible while only
20/30/50% of them are targets under any one mask. Today's augmentation (`hide_extra`) hides a random share of
visible eligible cells at **rate 0 half the time and U(0, 0.3) otherwise - about 7.5% on average** - and those
extra cells are removed from the input too, so the input gets thinner than it is at test time, which is why the
rate is capped.

The obvious next step is to resample the WHOLE mask from the truth at the correct rate instead of adding extra
hiding on top of the real one: every epoch sees a different target set with exactly the test-time input density.
A version biased towards isolated single-slot targets would aim it at S_LWR directly, since 70% of that
residual sits on one-target transitions (note 105).

Caveat: the curve above is over distinct DAYS. Resampled masks reuse the same traffic, so the gain per effective
doubling will be smaller than the curve implies. It is still the only lever that has moved anything today.
