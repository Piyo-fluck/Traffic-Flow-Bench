# 118. Debugging the residual: three axes, the schema, and a hypothesis that was tested rather than assumed

Date 2026-09-25. Code `analysis/t1_residual_axes.py`, `t1_cohort_ridge.py`, plus schema checks.
Follows note 117, which found 15.7% of the error variance is the part of the truth's own high-frequency
component the model misses (it captures about 73% of it, and tracks the smooth part at corr +0.01), leaving 84%
unaccounted for.

## Where the residual repeats itself

Correlation of the residual with itself, pairwise-complete, each against a control that breaks the pairing:

| axis | offset | free flow | control | congested | control |
|---|---|---|---|---|---|
| space | 1 link | **+0.1345** | +0.0075 | +0.2203 | +0.0204 |
| space | 3 links | +0.0488 | +0.0005 | +0.1052 | +0.0036 |
| space | 10 links | +0.0048 | +0.0011 | +0.0526 | +0.0104 |
| time | 1 slot | **+0.0709** | +0.0132 | **+0.3918** | +0.0701 |
| time | 3 slots | +0.0161 | +0.0151 | +0.2151 | +0.0632 |
| days | 1 / 7 / 14 | +0.003 / +0.022 / +0.024 | +0.009 / +0.010 / +0.011 | +0.067 / +0.088 / +0.091 | +0.055 / +0.043 / +0.051 |

- **No day-to-day repetition.** There is no standing bias at particular (link, time of day) to remove.
- Spatial structure is real but small, and **does not enter S_LWR**, which differences along time at one link.
- The S_LWR-relevant number is time lag 1 in free flow: **+0.0709**, on 96% of both-target transitions.
- Congestion is different - time lag 1 of +0.39, still +0.17 at lag 5 - but is 4% of cells.

## rho_t is a symptom, not a dial

S_state sees `sum e^2` and cannot see the temporal correlation of the error; S_LWR sees `sum |de|`, and
`Var(de) = 2 sigma^2 (1 - rho_t)`. Raising rho_t from 0.29 to 0.7 would be worth about +0.0038 in total, to 0.9
about +0.0068.

But it cannot be raised by adding anything: for a prediction perturbed by n,
`Var(de) = 2 s0^2 (1 - r0) + 2 sn^2 (1 - rn)`, so any n with `rn < 1` makes it worse, and `rn = 1` is a constant
offset over a run - exactly note 111's design, which leaves internal deltas untouched by construction. The only
way to raise rho_t is to predict the truth's own wobble, which is the irreducible part.

## The released schema has no unused proxy

Columns `tfb/data.py` never reads:

| column | what it is |
|---|---|
| `density_occ_linear_vehpkm` | occupancy x **285.0**, a single ratio on the whole panel; blanked at targets |
| `is_imputed` | readable at targets, but **exactly collinear with `pct_observed`** (1 <-> pct 75-89) |
| `is_observed` / `is_missing` | constant at targets (1 and 0) |
| `corridor_id`, `link_id`, `milepost`, `direction` | metadata already in hand |

And the residual does not care about the one that survives masking: RMSE in vehicles per 5 min per lane, imputed
against measured, is 2.025/2.007, 2.393/2.367, 2.978/2.978 on three panels - imputed cells are 1% *harder*.
Note 01 F4's "pct does not change the noise" now holds from the residual side too.

## The travelling-cohort prediction, tested and failed

The user's reading is that most of the 84% is microscopic realisation variance - the platoon that happened to
pass - correlated across cells but not conditionable on released macro state. That makes a falsifiable
prediction: a platoon on link i at time t is on link i+k at t + travel time, so the correlation should peak at a
lag that grows with the offset. One slot of lag is 9.57 km at the free-flow median of 114.9 km/h.

| offset | km | implied lag | lag 0 | lag 1 | lag 2 | control |
|---|---|---|---|---|---|---|
| -25 | 9.07 | 0.95 | **+0.0085** | +0.0022 | +0.0026 | +0.0011 |
| -3 | 0.96 | 0.10 | **+0.0599** | +0.0087 | +0.0049 | +0.0048 |
| -1 / +1 | 0.28 | 0.03 | **+0.1493** | +0.0250 / +0.0216 | +0.0101 | +0.0072 |
| +3 | 0.96 | 0.10 | **+0.0599** | +0.0122 | +0.0057 | +0.0034 |
| +25 | 9.07 | 0.95 | **+0.0085** | +0.0040 | +0.0072 | -0.0007 |

**No ridge.** The peak is at lag 0 at every offset out to 30 km; the surface is symmetric upstream and
downstream, where a moving platoon would be asymmetric; and it decays over about 1 km, where a platoon shared
inside one five-minute bin would stay flat out to 9.6 km.

So the common cause is local, with a length scale near 1 km, and does not travel within the five-minute frame.
What it is, this measurement does not say, and no third mechanism is offered here - two were proposed today
(queue-release timing, and the error as the truth's own wobble) and both were wrong.

**The failed prediction does not touch the main thesis.** "Correlated residuals are not usable residual signal"
holds whether or not the cause propagates: the hidden state is absent from the released inputs, so
`E[e | X_released]` stays near zero, which is what every corrector, retrieval and cross-panel test has found.
