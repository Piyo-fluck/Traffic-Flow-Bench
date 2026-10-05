# 151 Stage 1 (which bottleneck) learned on the training months with out-of-fold H

Scripts: `occ_lab/h_oof.py` (H retrained per calendar-month fold: priors and training origins without the month,
dropout prior, 2 bags x 600 rounds; scores the month's first-onset windows from observations; 9 folds, ~2 min),
`occ_lab/stage1_big.py` (candidate classifier on 1,722 training-month windows; features = note 129 with H as the
only member + note 145 capacity with the prior WITHOUT the window's month).

| | top-1 of the true bottleneck |
|---|---|
| training months, leave-one-month-out: classifier | 0.852 |
| training months: H's own region mass | 0.857 |
| scored months: production blend mass | 0.804 |
| scored months: big classifier (H features) | 0.785 |
| scored months: prior4 (note 145, cross-month) | 0.849 |

Swap onset IoU vs production (6 seeds): prior4 +0.0128 / +0.0248; big classifier alone -0.0030 / -0.0028; prior4 +
big score stacked +0.0128 / +0.0238. Member-level (swap H's own peaks, re-blend): H alone 0.7652 -> 0.7652 (val),
0.7981 -> 0.7958 (priv); blend unchanged.

With 5x the data the classifier does not beat H's own mass on the training months: the candidate features (state,
structure, same-month history, capacity, time) carry nothing H has not already learned there. The stage-1 gains of
notes 129 / 145 are corrections of the run shift in the scored months (H top-1 0.857 out of fold on the training
months vs the blend's 0.804 on the scored months) - more training-month data cannot teach them.
