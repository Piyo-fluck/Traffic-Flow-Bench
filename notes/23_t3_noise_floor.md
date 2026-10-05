# 23 — Task 3: S_LWR is close to (not at) the noise floor; the "real 0.51" was an attribution assumption

> **Correction (2026-09-15, `analysis/t3_floor_sim.py`, log `work/t3_floor_sim.log`).** The smoothed-truth
> "oracle" below is biased: it scores below the CNN on S_state, so it cannot establish a floor. A proper floor
> simulates an ideal predictor. At target cells it uses the truth plus fresh noise of the measured size (robust σ
> from second differences, per panel, speed separately for free flow and congestion). Family mean, train holdout:
>
> | | S_state | S_LWR~ |
> |---|---|---|
> | CNN (current) | 0.9523 | 0.6486 |
> | ideal predictor, noise ×1.0 | 0.9574 | 0.6844 |
> | ideal, noise ×0.5 / ×1.5 | 0.9787 / 0.9361 | 0.8425 / 0.5259 |
>
> The reachable headroom is +0.036 S_LWR~, not +0.009. That is about +0.004 total, or +0.005–0.006 together with
> S_state. The largest gaps are in D7_I210_W and D12_I5_N/S (+0.06–0.07). This floor assumes independent speed and
> flow noise. S_LWR only sees k = q/v, and the floor is very sensitive to the noise size, so the density noise is
> checked directly in `analysis/t3_density_noise.py`.
>
> **Density-noise check** (`analysis/t3_density_noise.py`, log `work/t3_density_noise.log`, 10 panels, train):
> - The speed and flow noise are independent: the correlation of the second differences of log q and log v is
>   0.0004.
> - The measured noise on log k (σ 0.0316) equals what independent noise implies (0.0312). The noise is
>   multiplicative, about 3% on density in both free flow and congestion (0.0306 in congestion).
> - The floor with density noise at the measured size is S_LWR~ **0.639**, against the CNN's 0.649. The additive
>   version above gave 0.684.
>
> **Final: the CNN sits between the two floor estimates (0.639–0.684).** T3 headroom is 0 to +0.036 S_LWR~, which
> is at most about +0.004 total. The README's "perfect answer 0.96" reproduces independent random detector
> noise and is unreachable by any method. No hidden noise structure (correlation, occupancy) exists to exploit.

Date: 2026-09-15 · Scripts: `analysis/t3_density_diag.py`, `analysis/t3_headroom.py`, `analysis/t1_pseudo_phys.py`
· Logs: `work/t3_density_diag.log`, `work/t3_density_decomp.log`, `work/t3_headroom.log`, `work/t1_pseudo_phys_validation.log`

## What the official S_LWR measures (score_task3.py)

- The state uses the organizer truth at eligible cells (`is_score_eligible` filter, line 64), and our Task 1
  values at the published targets.
- rhs = the organizer boundary flows. They are "projected onto the released observations, so a participant who
  reproduces what was published satisfies the conservation identity exactly". So rhs = the truth's own ΔN.
- S_LWR ≈ 1 − Σ|ΔN_state − ΔN_true| / Σ|ΔN_true| over every scored transition. Only transitions that touch a
  target cell can carry a residual. This is exactly the local proxy, apart from the observable-link and Mode C
  filters.

## Train holdout, saved `m3all` predictions, 10 panels (family mean)

| State at the target cells | S_state | S_LWR~ |
|---|---|---|
| CNN (current) | 0.9523 | 0.6486 |
| density from linear time interpolation of observed k, (v, q) projected onto it | 0.9231 | 0.5285 |
| truth smoothed over 3 slots (near noise-free oracle) | 0.9487 | 0.6580 |
| oracle only at congested targets | 0.9483 | 0.6501 |
| oracle only at free-flow targets | 0.9521 | 0.6568 |

- **Residual breakdown** (E = 0.364): 0.254 comes from gap-edge transitions (one end is truth), 0.110 from
  gap-interior transitions. Congested ends contribute 0.074 of E, although only 3.9% of target cells are congested.
- **Even a near-noise-free oracle gains only +0.009 S_LWR~.** The CNN is already at the noise floor for
  density. The edge residual |k̂ − k_true| mostly contains the truth's own observation noise.
- Density-targeted fixes therefore have no room: a density head, a projection, or a ΔN loss (note 05). The
  `M3_KHEAD` option added to `t1_m3_cnn.py` is kept, switched off by default, and not trained.
- In-month pseudo metric on validation (20% of visible cells hidden): ens3 S_state~ 0.9425 / S_LWR~ 0.7644, and
  ens3 + month base 0.9494 / 0.7702.

## The "real S_LWR ≈ 0.51" (note 06) depends on an assumption

Note 06 split the gain of submission A over the baseline (+0.1375) by assuming the validation S_state equals the
holdout value 0.952. The implied S_LWR is very sensitive to that assumption:

| Assumed S_state on validation | 0.952 | 0.93 | 0.91 |
|---|---|---|---|
| Implied real S_LWR | 0.51 | 0.59 | 0.66 (= the proxy) |

Real Task 1 masks are contiguous gaps, and validation is a different run, so a validation S_state of about
0.91–0.94 is plausible. The "gap" between the proxy and the real score is most likely a split mistake, not
missing headroom.

## What it changes

- Task 3 is not a lever. The most it can add is about +0.001 total.
- Task 1 and Task 4 are near their ceilings. To go beyond about 0.84, the gain has to come from Task 2 (S_queue).
