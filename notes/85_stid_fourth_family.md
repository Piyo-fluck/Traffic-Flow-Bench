# 85. STID with occupancy: the most orthogonal member so far, and still no score

Date 2026-09-21. Code `occ_lab/train_stid.py`, `cache_stid.py`, `sweep_occ.py`, `sweep_stid_fine.py`, trained on
Ada (4 seeds, 30k steps, ~12 min each). Follows note 83 (DCRNN, spatial SSM) and note 84 (occupancy as input).

## The model

STID (Shao et al. 2022): identity instead of graph machinery. Per link the 15 x 12 history (the note-84 channel
set, occupancy included) is flattened and projected to 96, then concatenated with a **link-identity embedding**
(1,450 global ids across the eight panels), a **time-of-day embedding** (288 slots) and a **day-of-week
embedding**, 32 each; three residual MLP blocks; a linear head to the six horizons at once. No mixing across
links - the identity is what carries space. 297k parameters, identity dropout 0.10, production loss and schedule.

This is the representation none of the production members has: gs1 sees a feature snapshot with no identity, cnn3
is translation invariant along the corridor, tfm8 is permutation invariant across links apart from its channels.
None of them can learn "link 47 at 08:35 on a Tuesday behaves like this".

## It worked, as a representation

| member | rescue val / priv | spoil val / priv | corr | own cell accuracy |
|---|---|---|---|---|
| **stid (4 seeds)** | 0.307 / 0.265 | **0.0251 / 0.0258** | **0.48 / 0.50** | 0.9651 / 0.9631 |
| occ_cnn (4 seeds) | 0.315 / 0.276 | 0.0093 / 0.0110 | 0.63 | 0.9807 / 0.9779 |
| cnn3 | 0.305 / 0.250 | 0.0076 / 0.0082 | 0.66 / 0.68 | 0.9822 / 0.9802 |
| tfm8 | 0.103 / 0.097 | 0.0028 / 0.0027 | 0.87 / 0.88 | 0.9840 / 0.9832 |
| gs1 | 0.277 / 0.273 | 0.0118 / 0.0124 | 0.62 / 0.61 | 0.9777 / 0.9764 |
| DCRNN / SSM (note 83) | 0.13-0.14% of cells | 1.76-1.86% of cells | 0.43-0.55 | - |

**The lowest error correlation of any member, combined with a real rescue rate.** The DCRNN and the SSM also had
low correlations, but they rescued almost nothing; STID rescues as much as cnn3 while agreeing with the blend far
less. The identity hypothesis is right about where the representation sits.

## It still does not pay

Break-even for a flat mixture is `spoil < rescue x 0.015`, i.e. 0.0042. STID is at 0.025, six times over, so
every weight above a token amount loses: `tfm slot 0.30 tfm8 + 0.30 stid` is **-0.0158 / -0.0179**, flat 0.30
is -0.0121 / -0.0144. Down at 0.01-0.02 out of the tfm8 slot it is finally neutral: `AH + stid 0.01` gives
+0.0004 / +0.0009 against the blend, where AH alone (the occupancy five-member mixture) already gives
+0.0004 / +0.0008. STID contributes nothing beyond noise at the only weight it survives.

## Why the gate fails too, precisely

The natural rescue for a noisy-but-orthogonal member is to use it only where the blend is unsure and it is
confident. Measured on exactly those cells:

| selection | n (priv) | blend correct | **STID correct** |
|---|---|---|---|
| \|blend-.5\| < 0.10, \|stid-.5\| > 0.30 | 10,475 | 0.564 | **0.470** |
| \|blend-.5\| < 0.10, \|stid-.5\| > 0.45 | 5,479 | 0.569 | **0.494** |
| \|blend-.5\| < 0.20, \|stid-.5\| > 0.45 | 12,044 | 0.637 | **0.531** |

**STID's confidence is worthless where it would matter**: on the cells the blend finds ambiguous, a confident
STID is a coin flip (0.47-0.53) while the blend is already at 0.56-0.64. Every gated variant loses.

## Reading

Four attempts now - recurrent graph decoder, directed spatial scan, occupancy input, identity MLP - and the
constraint has moved from "no new member is orthogonal" to something sharper: **orthogonality is available, but
no member has calibrated confidence on the cells that decide the score.** That is note 82's result seen from the
other side (the disputed cells carry calibrated coin-flip probabilities). A fifth member would have to beat the
blend *conditionally on the blend being unsure*, and nothing tried does.

FNO was the next item on the user's list; it is the same kind of bet (a different function class over the same
inputs) and would be judged by the same conditional test.
