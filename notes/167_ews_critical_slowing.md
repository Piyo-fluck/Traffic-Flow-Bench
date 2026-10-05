# 167 Early-warning signals (critical slowing down) for onset: no signal

Date 2026-09-27. Code `occ_lab/ews_eda.py`. Idea from the tipping-point literature: before a bifurcation the
state variable's variance and lag-1 autocorrelation rise. Training months, 1186 h_oof first-onset events; per link,
statistics over the 12 released slots before T; target = queued at T+30. Mean AUC over events:

| statistic | AUC | AUC in the top level quintile | AUC among H's top-10 links |
|---|---|---|---|
| level (mean q/cap) | 0.609 | 0.465 | 0.525 |
| var q/cap | 0.577 | 0.453 | 0.540 |
| lag-1 AC q/cap | 0.502 | 0.499 | 0.500 |
| var v/vf | 0.532 | 0.495 | 0.523 |
| lag-1 AC v/vf | 0.506 | 0.519 | 0.508 |
| lag-1 cross-corr with downstream q | 0.490 | 0.488 | 0.502 |
| q trend (last 4 - first 4) | 0.559 | 0.424 | 0.530 |
| var q late - early | 0.468 | 0.442 | 0.491 |
| H | 0.983 | 0.975 | 0.860 |

The variance's raw AUC is the flow level; within a level stratum and among H's candidates every statistic is ~0.5.
Consistent with a CTM-type simulator: free-flow speed is flat and there is no slowing-down precursor.

The other candidate from the same browsing round (low-rank tensor completion, LATC) was already measured as a
ceiling in note 99 (zero).
Note 142 already measured CSD statistics on the candidate peaks (negative); this note repeats it on all links with the level-stratified AUC.
