# 04 — Task 3: a validated local physics proxy; the CNN lifts S_physics from 0.32 to about 0.76

Date: 2026-09-14 · Code: `tfb/physics.py` · Script: `analysis/t1_physics01.py` · Log: `work/t1_physics01.log`

## How Task 3 is scored (read from `src/task3/score_task3.py`)
- **State**: every eligible cell of the regime's days. Non-target cells take
  the released truth, target cells take our Task 1 values. Ineligible cells
  are dropped.
- **S_FD** (weight 1/3): the triangular FD, per lane, with parameters from
  `fd_parameters.csv` (v_f = q0.99 speed, capacity = q0.999 flow,
  k_crit = cap/v_f, k_jam). It is computed as
  `1 - Σ|q_lane - q_FD(k_lane)| / Σ|q_lane|` with k = q/v. It is exactly
  computable locally.
- **S_LWR** (weight 2/3): `1 - Σ|ΔN - rhs| / Σ|rhs|` over 5-min transitions,
  where N = (q/v)·L and rhs = dt·(q_in + r_on - q_out - r_off). The boundary
  flows are organizer-held. They are projected onto the released
  observations, so the truth satisfies conservation.
- Mode A (all panels): a transition at a link with an attached ramp counts
  only when a valid ramp cell exists at t. On D12_I5_N 92% of transitions
  pass, on D7_I210_W 74%.

## The proxy
Take rhs := ΔN_true (the truth's own accumulation change), then
`S_LWR_proxy = 1 - Σ|ΔN_pred - ΔN_true| / Σ|ΔN_true|`.

**Validation**: on D12_I5_N the proxy gives the naive baseline
S_physics **0.3244**. The organizer quotes **0.3242** for that baseline scored
with the real boundary flows. The proxy overstates only the perfect answer
(1.0 against the real ceiling of about 0.96).

## Results (holdout test days, Jan–Feb 2031)
| Method | D12_I5_N S_state | S_FD | S_LWR~ | S_phys~ | D7_I210_W S_state | S_LWR~ | S_phys~ |
|---|---|---|---|---|---|---|---|
| Baseline profile | 0.612 | 0.973 | 0.000 | 0.324 | 0.766 | 0.079 | 0.379 |
| Time-anomaly interp | 0.900 | 0.986 | 0.105 | 0.399 | 0.895 | 0.408 | 0.599 |
| M2 LightGBM | 0.946 | 0.990 | 0.541 | 0.691 | 0.933 | 0.558 | 0.700 |
| **M3 CNN** | **0.960** | 0.991 | **0.652** | **0.765** | **0.945** | **0.638** | **0.753** |
| Truth | 1 | 0.990 | 1 | 0.997 | 1 | 1 | 0.994 |

By regime (M3, D12_I5_N), S_LWR~ is R1 0.78, R2 0.68, **R3 0.50**.

### All 10 panels, M3 pooled model (40 epochs), test days
Script `analysis/t1_phys_eval.py m3all` · Log `work/t1_phys_eval_m3all.log`

| Panel | S_state | S_FD | S_LWR~ | S_phys~ | S_LWR~ R1/R2/R3 |
|---|---|---|---|---|---|
| D7_I10_E | 0.9524 | 0.983 | 0.644 | 0.757 | 0.78/0.67/0.49 |
| D7_I10_W | 0.9568 | 0.987 | 0.647 | 0.761 | 0.78/0.68/0.49 |
| D7_I210_E | 0.9463 | 0.987 | 0.623 | 0.744 | 0.76/0.65/0.46 |
| D7_I210_W | 0.9449 | 0.985 | 0.648 | 0.760 | 0.78/0.67/0.49 |
| D7_I405_N | 0.9391 | 0.983 | 0.631 | 0.748 | 0.76/0.66/0.47 |
| D7_I405_S | 0.9469 | 0.986 | 0.657 | 0.767 | 0.78/0.68/0.50 |
| D12_I5_N | 0.9617 | 0.991 | 0.667 | 0.775 | 0.79/0.69/0.52 |
| D12_I5_S | 0.9584 | 0.990 | 0.670 | 0.777 | 0.80/0.70/0.52 |
| D12_I405_N | 0.9672 | 0.993 | 0.656 | 0.769 | 0.78/0.69/0.50 |
| D12_I405_S | 0.9487 | 0.992 | 0.642 | 0.758 | 0.78/0.66/0.48 |
| **Family mean** | **0.9523** | | **0.649** | **0.762** | |

- S_LWR~ is remarkably uniform across panels (0.62–0.67). What drives it is
  the share of transitions that touch a target (the regime), not the
  corridor: R1 about 0.78, R2 about 0.67, R3 about 0.49 everywhere.
- The Task 1 + Task 3 contribution is 0.35 × 0.9523 + 0.15 × 0.762 =
  **0.4475**, against 0.294 for the baseline, about +0.15 of total score.
  The proxy overstates S_LWR near the top (its ceiling is 1.0, the real one
  about 0.96), so expect a little less on the leaderboard.

## Findings
- **F15**: S_FD is saturated for any sensible answer (0.97–0.99). All of
  Task 3's discrimination sits in S_LWR.
- **F16**: S_LWR is very sensitive to density. With M3, the density error at
  targets has median 1.7% (D12_I5_N) and 2.2% (D7_I210_W). The conservation
  signal |ΔN|/N has median only 2.7% and 3.4%, so the error is the same order
  as the signal. That is why interpolation methods score S_LWR ≈ 0–0.4
  despite S_state ≈ 0.9.
- **F17**: The metric sees the *time difference* of density errors
  (ΔN_err(t+1) − ΔN_err(t)). Errors that stay constant across consecutive
  cells cancel. An error at a target next to a released cell does not
  cancel. R3 suffers most because more transitions touch targets.
- In total-score terms, moving from the baseline to M3 is worth about
  0.15 × (0.76 − 0.35) ≈ **+0.06** through Task 3 alone. The remaining
  headroom to the ceiling of about 0.95 is about +0.03.

## Hypotheses for the next loop
- **H5**: Adding a ΔN-consistency loss (an L1 of the ΔN error normalised by
  |ΔN_true|, which is the proxy itself) to the CNN raises S_LWR at little
  cost in S_state.
- **H6**: Released ramp flows (on/off, with validity) as input channels help
  the flow estimate and the conservation, because flow steps between
  stations happen at ramps.
- **H7**: Seed or architecture ensembles of the CNN reduce the
  uncorrelated part of the errors, which is the part S_LWR punishes.
