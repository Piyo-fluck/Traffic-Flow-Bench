# 156 A: tabular in-context learner (TabICLv2) on the uncertain ongoing cells, conditioned on the scored month

Script: `occ_lab/icl_ongoing.py [--cross]` (TabICLv2 2.2.0, BSD-3, checkpoint tabicl-classifier-v2-20260212;
150k context rows in ~7 s on the 4090). Cells with AU's final p in [0.05, 0.95] (~260-280k per month, half
eligible); per day d the context = eligible uncertain cells of days < d of the same month (all panels);
`--cross` adds the other scored month's cells (cap 120k rows). Features 'stack' = 7 members' p, their +-2 max,
AU p0 / final p, horizon, v/vf history of the link and +-1/+-2 neighbours, distances to the queue at T, queued share
of +-8, time; 'raw' = without any member / AU p. Control: LightGBM on the same context. q = a p + (1-a) p_AU, tau 0.5.

| vs AU (ongoing family IoU) | all val | all priv | days 1-5 val | days 1-5 priv |
|---|---|---|---|---|
| same month only: ICL stack a0.5 | +0.0056 | +0.0068 | -0.0017 | -0.0043 |
| same month only: LGB stack a0.5 | +0.0044 | +0.0061 | -0.0056 | -0.0050 |
| same month only: ICL raw a0.5 | -0.0271 | -0.0364 | -0.0116 | -0.0199 |
| + other month: ICL stack a0.5 | +0.0091 | +0.0091 | +0.0004 | +0.0048 |
| + other month: LGB stack a0.5 | +0.0073 | +0.0075 | +0.0059 | +0.0034 |
| + other month: LGB stack a1.0 | +0.0090 | +0.0054 | +0.0070 | +0.0019 |

Brier on the modelled cells (+other month): AU 0.1002 / 0.1090, ICL 0.0903 / 0.1007, LGB 0.0892 / 0.0992.
The history-only in-context learner is far below the members. The gain comes from the DATA (a stacker on the
uncertain cells with the scored months' released labels), not from the in-context learner: LightGBM on the same
context is equal or better, and on days 1-5 better on validation.
