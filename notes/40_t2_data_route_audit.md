# 40 — Task 2 data-route audit: the excluded routes hold no usable edge; the horizon is fully blanked

Date: 2026-09-15 · Script: `analysis/t2_route_audit.py` · Log: `work/t2_route_audit.log` · Measurement only.

The routes were checked because onset is information-limited (note 39) and the question was whether the gap to
the top (~0.92) comes from data our rule excludes.

## Route A — same-day data after the blanked hour (T+95 … T+150)

The test adds "visibly queued at T+95 … T+150" to the cached model probabilities on the blank-aware official-like
windows (onset: re-rank with p + β·later_q, k from the expected-IoU rule; ongoing: add later-queued links with p ≥ 0.3).

| Δ IoU (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| onset, β = 1 (≈ later-only ranking) | +0.022 (1/2) | +0.015 (1/4) | −0.002 (1/3) | +0.006 (3/2) |
| onset, β = 0.25 | ±0 | ±0 | ±0 | ±0 |
| ongoing, add later-queued | +0.004 | +0.003 | −0.000 | −0.001 |

Small and inconsistent: more panels get worse than better. One to two hours later the morning queue has spread,
so where it is later does not pin down where it started. Being future information for an online forecast, it is not
worth a rule change.

## Route D — ineligible horizon cells

MASK_SPEC says the horizon is blanked on eligible cells, but the released masked layer carries **no values at all in
the official horizons**: 0 of 174,000 horizon cells, across 8 panels × 2 splits. The rebuild closed this route.

## Overall

| Route | Result |
|---|---|
| A: same day, after the blanked hour | Small and inconsistent (this note) |
| B: other days of the month | Onset worse (note 25) |
| C: horizon ramp flows | Worse (note 14) |
| D: ineligible horizon cells | Nothing left to read |

**The gap to the top does not appear to come from data we exclude.** Recommendation: keep the data rule unchanged.
