# 150 Where the onset still fails, and a stage-2 site-count decision

Scripts: `occ_lab/onset_stage2.py [ratio|peak|hi]`; decomposition / missed-site checks in the session scratchpad.
Probabilities: note 145 prior4-swapped (the current candidate), lab onset pool, ratio rule.

Loss decomposition (331 windows, total IoU loss 43.0):

| kind | windows | share of loss |
|---|---|---|
| perfect | 261 (79 %) | 0 |
| wrong region | 20 | 41 % |
| truth at 2+ regions | 32 | 36 % (sites 24-52 links apart; mostly under-covered: truth 3-7 cells, pred 2-4) |
| right region, extent wrong | 18 | 24 % (14 too big: pred 3.2 vs truth 1.8 cells) |

Stage 2 (per window: keep R / add the runner-up region's set / only the top region; LightGBM gain regressors with a
margin, cross-month, 6 seeds). Oracle over these options: +4..+6 IoU per month (~+0.03 per window); 'add' helps 3-4
windows per month and hurts ~50-57, in every form (runner-up ratio set, its peak cell, its cells p >= 0.5 - the last
is already in R).

| option, margin 0.05 | val | priv |
|---|---|---|
| add (ratio set) | 0.0000 | -0.0036 |
| only top region | +0.0004 | +0.0074 |
| add (p >= 0.5) + only | +0.0004 | +0.0082 |

The second site is not in the evidence: the missed true cells (>= 4 links from R) have p median 0.07, and no speed
precursor anywhere - min v/vf over +-2 links in the last 15 min before T is 0.98 at missed, hit and non-queued links
alike (the breakdown happens in the unseen 30 min; onset evidence is demand / time / site history, not speed).
