# 160 Ongoing: the onset machinery on the queues away from the visible one (note 159's 2/3 of the loss)

Script: `occ_lab/og_away.py [--auc | --reweight]`, `occ_lab/og_cellstack.py`. Regions: outside the T-queue zone
(<= 2 links), top 5 peaks (>= 4 apart, p >= 0.05) of AU's max-over-horizons p, +-4 links; ~8.9-9.5k regions per month.

1. Region logit shifts learned from the IoU change (delta -3..+3, regressor + margin, cross-month, 6 seeds):
   best +0.0008 / +0.0008 all, -0.0000 / +0.0008 days 1-5 (margin 0.005). Oracle of the shifts ~ +0.02 per window.
2. Is there information? Region label 'any true queue in the region within the horizon', cross-month AUC:
   | | all regions | decision band (AU region p 0.2-0.8) |
   |---|---|---|
   | AU region p | 0.922-0.925 | 0.65-0.70 |
   | feature classifier | 0.963 | 0.87-0.90 |
   T+20..30 label: AU 0.88-0.89 -> 0.945-0.949 (band 0.65-0.69 -> 0.87-0.88). But 83-89 % of band regions are
   positive: the label says 'somewhere in the region', not which cells.
3. Likelihood-ratio shift per region x horizon (classifier logit - AU-only classifier logit, beta 0.25-1):
   within +-0.001 (beta 0.5: +0.0004 / -0.0000 all, +0.0007 / -0.0000 days 1-5), worse at beta 1.
4. Cell-level stacker fitted on the OTHER month only (note 156's features), with / without the away evidence
   (signed distance to the T-queue, zone flag, observed share 15 min / at T, AU max and +-4 max, note 145 headroom):
   | | all val | all priv | d1-5 val | d1-5 priv |
   |---|---|---|---|---|
   | stack a0.5 | +0.0077 | +0.0054 | +0.0066 | +0.0024 |
   | stack + away a0.5 | +0.0080 | +0.0056 | +0.0044 | +0.0032 |
   The away evidence adds nothing at cell level (the stack already has the distances); the cross-month cell stacker
   itself is the consistent positive (+0.002..+0.008), the same family as note 156.
