# 148 Learning the onset decision rule (vs the fixed ratio rule)

Scripts: `occ_lab/onset_learned_decode.py`; residual variant in the session scratchpad (same data, same features).

Candidate sets per window: prefixes of the p ordering (k 1..12) and of the top region (k 1..8). Oracle best set:
val 0.926-0.928, priv 0.923-0.935 vs the ratio rule 0.816-0.829 / 0.862-0.887: the headroom is large.

| rule (vs ratio rule, 6 seeds, cross-month) | base val | base priv | swapped val | swapped priv |
|---|---|---|---|---|
| set IoU regression, arg-max | +0.0051 | -0.0157 | -0.0029 | -0.0286 |
| lambdarank | -0.0077 | -0.0342 | -0.0201 | -0.0329 |
| residual (k_ratio +-1/2, top-region), margin 0.05 | -0.0002 | -0.0040 | -0.0024 | +0.0003 |
| residual, margin 0.1 | 0.0000 | -0.0025 | -0.0003 | +0.0001 |
| c cross-month | +0.0153 | +0.0011 | 0.0000 | -0.0014 |
| c TTT per month-panel (last <= 4 days, shrink) | +0.0013 | +0.0063 | -0.0097 | +0.0079 |

~330 labelled onset windows per month: which set wins is not predictable from the probabilities' summaries; every
learned rule loses or ties. On the current candidate (swapped) c = 1 stays best. Training-month windows would need
out-of-fold member probabilities (the members are fitted on them) - not available.
