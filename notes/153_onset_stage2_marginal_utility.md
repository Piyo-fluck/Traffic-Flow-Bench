# 153 Onset step 2: second site by learned marginal utility dU(c | b)

Script: `occ_lab/onset_stage2_du.py` (on `onset_rerank.prepare()`). Stage 1 = note 152's IoU-LambdaRank + swap
(cross-month); S_b = the ratio rule on its output; teacher dU(c|b) = IoU(S_b u S_c) - IoU(S_b), S_c = the ratio rule
restricted to region c. LightGBM regressor, 12 low-dof features (no link id); threshold from 5-fold OOF on the fit
month only; at most one added region per window. 6 seeds.

Pairs: ~570 per month, dU > 0 only 13-14 (sum +4.0..+4.6), dU < 0 ~300 (sum ~ -100).

| vs stage 1 | all IoU | multi-site windows (29) | single-site windows | single worsened | adds / run |
|---|---|---|---|---|---|
| applied to validation | +0.0042 ± 0.0022 | mean +0.026 (sum +0.44) | sum +0.47 | 1.5 | 9.2 |
| applied to private | +0.0038 ± 0.0021 | 0.000 | sum +0.67 | 0.3 | 4.2 |

Positive both ways and the single-site windows are not broken - most of the gain is in single-site windows (a queue
split by a region boundary gets its other part back), as the dU teacher intended. True multi-site windows improve
only on validation. Seed-dependent: 1 of 6 seeds per month picks no threshold (no adds). vs production: stage 1 +
stage 2 ~ +0.019 / +0.039; vs note 145's swap ~ +0.006 / +0.015.
