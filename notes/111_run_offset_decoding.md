# 111. Target-run offset decoding: a +0.0168 ceiling, a +0.00007 floor, and a leak I had to correct

Date 2026-09-24. Code `analysis/t3_run_offset_oracle.py`, `t3_run_offset_feasible.py`, `t3_run_offset_eb.py`.

## The idea

The user's decision unit for S_LWR is not the cell but the **run of consecutive target slots**. Adding one
constant c to the density over a run leaves every internal target-target difference untouched, so the 846,229 of
residual that note 105 puts on both-target transitions is frozen by construction, and only the run's two
visible<->target boundaries move - the 1,979,135 that carries 70% of the residual. Density goes back onto (q, v)
by the minimum-move projection under the metric's own curvature.

This is Minimum Bayes Risk decoding: the predictor is untouched and only the action changes. It is the right
frame, because the CNN is MSE-optimal (note 103) and S_state wants exactly that, while S_LWR is an L1 functional
of error *differences* and does not.

## The ceiling: +0.01678

Giving every run the offset an oracle knowing its two boundary errors would choose (10 panels, 4.9M runs, local
test block, the real metric end to end):

| | S_state | S_FD | S_LWR | 0.35 s + 0.05 f + 0.10 l |
|---|---|---|---|---|
| production | 0.9523 | 0.9875 | 0.6486 | 0.44752 |
| **oracle offsets** | 0.9520 | 0.9863 | **0.8180** | **0.46431 (+0.01678)** |

Mean \|c\| is 0.6 veh/km against a mean density of 40 - a 1.5% move - and **S_state is untouched (-0.0003)**.
Larger than note 105's constant-error oracle (+0.0125), because this optimises the trade directly rather than
imposing a shape.

## The floor: +0.00007

The boundary has an **observable** proxy: the cell just outside a run is visible, hence the truth, so
`ehat_L = k_A[a] - k_true[a-1]` is computable and equals the true boundary error plus one slot of the truth's own
change. Substituting `alpha * ehat` for the unknown error and calibrating alpha out of fold, evaluated on the
other month:

| decoder | 2031-01 | 2031-02 |
|---|---|---|
| production | +0.00000 | +0.00000 |
| one global alpha (0.20) | **-0.00033** | **-0.00034** |
| alpha per (regime, run length, both-boundaries) | **+0.00007** | **+0.00006** |

The empirical-Bayes calibration picks alpha 0.00-0.10 in almost every group: told to choose how much to believe
the proxy, it chooses barely at all. **3.6% of the ceiling at best, and a single global trust level loses.**

The reason is note 107. `ehat - e` is one slot of the truth's own change, and the truth's high-frequency
component is large and nearly white (rho1 = -0.54 against the -0.667 of pure noise). The proxy is mostly that
change, not our error. The decoder is fine; the information is not there.

## The leak, and the correction

My first feasible run reported **+0.00060 held out on both months**. That was wrong. The state cost in the
decoder's objective was the *realised* change against the truth, `(v' - v_true)^2 - (v_A - v_true)^2`, which is
not available at decision time - so even at alpha = 0 the decoder was already moving runs to reduce a known
error, and its own alpha = 0 baseline was an oracle worth +0.0016. The honest cost is the displacement
`(v' - v_A)^2`, which for an unbiased estimator is the expected increase in squared error. With that fix the
alpha = 0 decoder is exactly production, and the numbers above are what remains. Both scripts now use it.

## Reading

The branch is closed, and with it the (c, d) extension and any finer conditioning: they all sit under the same
ceiling and the same missing information. What the exercise did establish is worth keeping - **the gap between
MSE-optimal prediction and score-optimal action on this task is +0.0168, and essentially all of it is
unreachable**, which is the same conclusion notes 96-107 reached from six other directions.
