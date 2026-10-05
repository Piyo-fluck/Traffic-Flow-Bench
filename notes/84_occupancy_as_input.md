# 84. Occupancy as a model input: a better member by D_m, no score

Date 2026-09-21. Code `occ_lab/` (`train_occ.py`, `eval_occ.py`, `cache_occ.py`, `sweep_occ.py`, `gate_occ.py`),
run on Ada. Follows note 82 (occupancy carries residual signal on the disputed cells: LOPO AUC 0.772 -> 0.785)
and note 83 (two new architectures failed the orthogonality test).

## What was trained

The production CNN and transformer recipes, unchanged (windows from `earliest_onset` + <=3000 ongoing candidates,
BCE + 0.5 Lovasz, 30k steps, OneCycle 1e-3, ES on 2031-01/02), with the input grown from 12 to 15 channels:
occupancy x10, occupancy over the link's weekday x tod mean clipped to [0, 5], and the change of occupancy across
the history x10 - zero where the slot is not observed, exactly as v and flow are handled. Four seeds each.
Training sees the true layer with `ctx.el` (the production convention); inference is given the released
`obs_occ` with the window's own `ok` mask.

## D_m, the criterion

P(member right | production blend wrong) over the eligible ongoing cells, against the cost P(member wrong | blend
right), and the cell-level error correlation with the blend.

| member | rescue val / priv | spoil val / priv | corr |
|---|---|---|---|
| occ_cnn (4 seeds) | 0.315 / 0.276 | 0.0093 / 0.0110 | 0.63 |
| occ_tfm (4 seeds) | 0.210 / 0.221 | 0.0069 / 0.0077 | 0.71 |
| cnn3 (production) | 0.305 / 0.250 | 0.0076 / 0.0082 | 0.66 |
| tfm8 (production) | 0.103 / 0.097 | 0.0028 / 0.0027 | 0.87 |
| gs1 (production) | 0.277 / 0.273 | 0.0118 / 0.0124 | 0.62 |
| DCRNN / spatial SSM (note 83) | 0.13-0.14% of all cells | 1.76-1.86% of all cells | 0.43-0.55 |

**Changing the input beat changing the architecture.** occ_cnn rescues more of the blend's errors than the cnn3 it
would replace, at a lower error correlation, while the DCRNN and the SSM were an order of magnitude behind.

## Why that does not become score

The blend is right on 98.5% of the cells and wrong on 1.5%. A member pays flat only if

    spoil < rescue x 0.015   ->   for occ_cnn: 0.0093 needed to be under 0.0047.

It is twice that, so the rescue cancels. The weight sweep agrees: the best mixture that survives both splits is
`cnn +0.05 occ_cnn / tfm +0.05 occ_tfm` at **+0.0004 (val) / +0.0008 (priv)** family IoU - about +0.0001 total
before note 58's realisation factor. Replacing a whole slot loses (`tfm8 -> occ_tfm` -0.0137 / -0.0200).

## The gate does not work either

Restricting the occupancy model to the near-threshold cells, where the base rate is not 98.5%:

| band | n (val) | blend / occ_cnn / half-and-half, validation | private |
|---|---|---|---|
| \|p-.5\| < 0.05 | 9,642 | 0.536 / 0.566 / 0.569 | 0.541 / **0.511** / 0.517 |
| \|p-.5\| < 0.10 | 19,607 | 0.560 / 0.572 / 0.578 | 0.566 / **0.525** / 0.537 |
| \|p-.5\| < 0.20 | 42,144 | 0.624 / 0.600 / 0.623 | 0.634 / **0.561** / 0.592 |

On the private month the occupancy CNN is *worse* than the blend exactly where the blend is unsure, and the
validation edge (+0.03 on the narrowest band, 9.6k cells) does not reproduce. So occ_cnn's D_m advantage comes
from cells where the blend is confidently wrong - the same place a confidently-wrong disagreement spoils
confidently-right cells. The best gated variant is +0.0007 / +0.0000, i.e. the flat mixture again.

## Reading

Occupancy is real information (note 82's AUC still stands) but the three production members already extract what
it implies about the next 30 minutes through v/vf and q/cap. This is now the third independent attempt to add a
member - two architectures, one input - and all three land inside the same band; note 80's picture of the blend
(opposite biases, half-independent errors, 2.9% disputed) is unchanged. No candidate was built from this.

## LB: AH = 0.87899 (submission 56410560), +0.00028 against Z

- AH = Z with the ongoing blend changed to `gs1 0.20 / tfm8 0.55 / occ_tfm 0.05 / cnn3 0.15 / occ_cnn 0.05`
  (`scripts/t2_occ_blend_predict.py` -> `scripts/t2_patch_ongoing.py`, onset kept byte-identical). **31 queue
  rows differ from Z.**
- The local estimate was +0.0004 / +0.0008 ongoing family, i.e. about +0.0001 total and "inside the noise"; the
  LB gave +0.00028, roughly three times the estimate. Local Task 2 estimates have never predicted the LB in
  either direction (note 58's realisation band was derived from gains, not from near-zero deltas), and 31 rows
  is too small a change to read a realisation factor from.
- Against AI (0.87902, note 86) the difference is 0.00003 - not distinguishable. Both beat Z.
