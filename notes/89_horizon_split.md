# 89. Horizon-split residual correctors

Date 2026-09-22. Code `occ_lab/horizon_split.py`, log `occ_lab/logs/horizon_split.log` on Ada. The user's step 3
after note 86. Everything is AI's recipe (15 features, the tail's masked-layer windows, L2 on y - p0, 400 trees,
prob update); only the number of correctors changes.

    one       AI: one corrector over the six horizons (horizon is a feature)
    near/far  two correctors, T+5..T+20 and T+25..T+30
    six       one per horizon

Family IoU gain over the frozen blend at AI's setting (eta 0.75, gate 0.10), split into the part earned on the
near horizons and on the far ones (each part = only that group corrected):

| test | one (AI) | near/far | six |
|---|---|---|---|
| Jan -> Feb, obs | **+0.0168** (near +0.0068, far +0.0074) | +0.0149 (+0.0060, +0.0072) | +0.0145 |
| Jan -> Feb, full | **+0.0172** (+0.0060, +0.0083) | +0.0153 (+0.0053, +0.0077) | +0.0160 |
| tail -> March | +0.0046 (+0.0011, +0.0022) | **+0.0050** (+0.0012, +0.0029) | +0.0036 |
| tail -> April | +0.0042 (+0.0015, +0.0010) | **+0.0045** (+0.0027, +0.0006) | +0.0027 |

## Reading

- Inside the tail, splitting loses (-0.0019 / -0.0019 for near/far); on the scored months near/far is ahead of
  AI by +0.0004 / +0.0003 - the same size as the Newton variant's local lead, which lost on the LB (note 88).
- Six correctors lose everywhere: the per-horizon models are too thin.
- Where the gain comes from is not stable across months: in March the far horizons carry two thirds of it, in
  April the near ones do. The T+25 / T+30 concentration of errors (note 52) does not turn into a far-horizon
  concentration of what the corrector can fix.
