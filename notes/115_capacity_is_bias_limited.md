# 115. Width was dismissed on one run: it is the largest S_LWR lever found, and it is bias-limited

Date 2026-09-25. Ada, 4x RTX 6000 Ada. Logs `work/logs/*.log` on the remote, summary `work/ada_summary.py`.

## The seed noise floor, measured at last

Nine independent seeds at the production width 64, 25 epochs, 10 panels, local test block:

| | min | max | range | sd |
|---|---|---|---|---|
| S_state | 0.95062 | 0.95135 | 0.00073 | **0.00021** |
| S_LWR | 0.63041 | 0.63962 | 0.00921 | **0.00263** |
| total (0.35s+0.05f+0.10l) | 0.44522 | 0.44626 | 0.00104 | **0.00032** |

Every single-run comparison in notes 105, 112 and 113 sits inside 0.5 sd of this. The nine runs were launched as
an ensemble expansion, which the user correctly called a hedge off the thread; **their real value was this noise
floor**, and it should have been measured before any of those sweeps.

## Width

| width | S_state | S_FD | S_LWR | total | vs w64 |
|---|---|---|---|---|---|
| 64 (mean of 9 seeds) | 0.95101 | 0.98747 | 0.63635 | 0.44586 | - |
| 128 | 0.95239 | 0.98757 | 0.64765 | 0.44748 | +0.00162 |
| **192** | **0.95263** | 0.98759 | **0.65135** | **0.44793** | **+0.00207** |

In sd units, w192 over w64: S_state **+7.7 sd**, S_LWR **+5.7 sd**, total **+6.5 sd**. Nothing else moved S_LWR
outside 0.5 sd all session. **Note 07's "width 128 alone = no gain" is wrong**; w128 alone is +4.3 sd on S_LWR.

A single w192 beats the three-member production ensemble (S_state 0.9523, S_LWR 0.6486) on both terms.

## Why: marginal accuracy, not correlation

`analysis/t3_qv_error_correlation.py` on the saved predictions:

| run | rho(dv, dq) | sd dv/v | sd dq/q | sd dk/k |
|---|---|---|---|---|
| w64 | +0.019 | 0.0171 | 0.0286 | 0.0319 |
| w128 | +0.008 | 0.0154 | 0.0277 | 0.0309 |
| w192 | +0.007 | 0.0155 | 0.0275 | **0.0306** |

**Capacity does not move rho** - it falls slightly. Note 113's degree of freedom resists the explicit loss AND
raw capacity. What capacity buys is the marginals, speed most (-9.4%).

And a clean confirmation: sd(dk/k) falls 4.1% from w64 to w192, and E_LWR falls 0.3636 -> 0.3486, **also 4.1%**.
`E_LWR` is proportional to `|e|`, which every extrapolation in notes 111-113 assumed without checking.

## Bias-limited, and data quantity nearly exhausted

`analysis/t1_train_test_gap.py` on each checkpoint, sd(dk/k):

| width | trained on (183 d) | held out (59 d) | gap |
|---|---|---|---|
| 64 | 0.02991 | 0.03186 | -0.00195 |
| 128 | 0.02905 | 0.03087 | -0.00182 |
| 192 | 0.02877 | 0.03063 | -0.00186 |

**Train and test fall in lockstep and the gap is constant.** That is approximation-error (bias) limited, not
variance limited: extra capacity lowers what the model class can represent, and does not buy overfitting.

It also settles note 114's reading. The learning curve over days was real, but at 183 days the gap has nearly
closed, so **data quantity is close to its asymptote**; what remained was approximation error, which is what
width just took. Width is decelerating too (train -2.9% then -1.0%, for 4.0x then 2.25x the parameters).

## The bound

If test reached the current training level (0.0288), E_LWR would fall another 6% and S_LWR reach about 0.672,
i.e. **+0.002 on the total**. Capacity and data together are worth roughly **+0.002 to +0.004 and no more**.

S_LWR's pool runs to 0.945 (note 112). **The remaining 0.29 is outside what capacity or more of the same data
can reach**, which is the user's position: the information is not there. The correct statement after today is
that there was an unclaimed +0.003 in approximation error, now found, and the body of S_LWR still needs
information of a different kind.
