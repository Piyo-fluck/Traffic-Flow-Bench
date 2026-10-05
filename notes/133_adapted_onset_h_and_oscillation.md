# 133 - Onset H adapted to the run by boosting continuation: large on the lab pool, nothing with the official windows' data

`occ_lab/adapt_h.py`: before each day, LightGBM (7 leaves, lr 0.05, l2 10) on the run's released pseudo onset events
(some link queued at t, nothing queued in the panel in [t - 90 min, t), every event of every earlier day and of the
same day before its first origin), features = H's own onset_X, init_score = logit(H); prediction logit(H) + f(X);
the adapted H replaces H (0.35) in the production onset blend.

| onset family IoU (val / priv) | all earlier days | only the last 4 | last 2 | last 1 |
|---|---|---|---|---|
| H alone 0.765 / 0.798 -> adapted, 1000 rounds | **0.870 / 0.878** | 0.768 / 0.766 | 0.756 / 0.792 | 0.767 / 0.798 |
| in the blend, vs production | **+0.046 / +0.020** (6/0, 8/1 rescue/spoil) | +0.006 / -0.002 | -0.006 / -0.005 | 0 / 0 |

The official windows sit early in the month (window_index.csv: onset day mean 3.3, median 3, max 10; ongoing mean
4.8, median 4), so at prediction time they have 0-4 earlier days (about 2). The lab gain comes from windows late in
the month with up to 30 earlier days of pseudo events; restricted to the data an official window actually has, the
adaptation is neutral to negative. Not usable for the submission.

## Oscillation check (for oscillator-type models)

Train truth, 4 panels, the longest queue of a link-day when >= 3 h (145 cases): the speed inside a queue varies by a
median sd of **0.5 km/h**; the ACF of its 5-min differences is -0.30 at lag 1 and within +-0.06 beyond (no periodic
return); the spectrum of the differences rises smoothly toward short periods with no peak. The simulator has no
stop-and-go oscillation at 5-min resolution, so an oscillator prior has nothing to represent.
