// filepath: notes/60_t2_onset_pathload_oracle.md
# 60 — T2 onset: demand / path-load ranks the missed bottleneck far better than M, but its ceiling is 0.45

Date: 2026-09-16 · Script: `analysis/t2_onset_pathload_oracle.py` · Log: `work/t2_onset_pathload_oracle.log` (10 s)
· Data: `work/t2_onset_pathload_oracle.csv`

The user's track ②: the missed onset windows are a demand problem, not a location problem, so infer path/OD demand
from the window's own history (a "causal Task 4") and rank the candidate bottlenecks by future load. **Oracle
first**, as agreed: if knowing the future demand does not rank the true bottleneck first, no estimator can.

Solve per panel: `min_x ||A x − y||² + λ||x − b||²`, closed form `x = b + Aᵀ(AAᵀ + λI)⁻¹(y − A b)` (the official
`[A; √λI]` stack, one factorisation of a links×links matrix instead of an NNLS over 2.6k–5.1k paths), negatives
clipped. Candidates and windows as in notes 51/59.

| ranking score | all windows: top-1 / top-3 | **missed windows (40): top-1 / top-3** |
|---|---|---|
| M's own probability (reference) | **0.76** | **0.05** |
| local flow / capacity now | 0.21 / 0.56 | 0.35 / 0.57 |
| path load from history, extrapolated | 0.25 / 0.35 | **0.42** / 0.55 |
| the same with a zero prior | 0.27 / 0.35 | 0.40 / 0.55 |
| **ORACLE: T+30 flows known** | 0.28 / 0.61 | 0.38 / 0.68 |
| **ORACLE: T+30 demand known** | 0.24 / 0.34 | **0.45** / 0.50 |

## Reading

1. **The signal is real.** On exactly the windows M gets wrong, demand-based ranking finds the true bottleneck
   0.35–0.45 of the time against M's 0.05. These are not the same errors.
2. **The ceiling is the formulation, not the estimation.** The oracles that know the future (0.38–0.45) barely beat
   the estimate built from history alone (0.42). Improving the ODME, the extrapolation or the travel-time routing
   cannot lift this line much above 0.45 top-1.
3. **Standalone it is far worse than M** (0.21–0.28 vs 0.76 over all windows), so it can only ever be an auxiliary
   signal; replacing M's ranking with it would wreck the 88% of windows M already gets right.
4. The released weak prior (a PM-period object) makes no difference: `path` 0.42 vs `path0` 0.40 on the missed
   windows. The corridor's mainline links are ~100% observed (137/146, 350/352, 81/81 …), so the ODME has no
   unobserved links to infer — its only added value would be temporal routing, and the oracle shows that is not
   where the missing information is either.

Against the criteria fixed before the run (≥ 0.5 → pursue; 0.2–0.5 → auxiliary only; < 0.2 → dead), this lands in
the **auxiliary band**. Combined with note 58's calibration — onset-side gains have never transferred to the LB —
the expected value of building the causal estimator is low.

## The modification worth testing instead

Both this note and note 59 fail the same way: **absolute demand and absolute past activity are highest at the
habitual bottleneck**, which is exactly what M already picks. The fix is to rank by the *anomaly*, not the level:

    score_i = (today's demand at candidate i) / (the train profile for this weekday type and slot at candidate i)

so a candidate that is 20% busier than usual outranks the habitually busiest one. The same script can produce it
with one extra line (the per-candidate profile already exists as `load_tq`). That is the next cheap test of track ②
before any estimator is built.
