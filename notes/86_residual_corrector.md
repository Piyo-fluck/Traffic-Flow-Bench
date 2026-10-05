# 86. Residual corrector on the frozen ensemble: the first real ongoing gain since T

Date 2026-09-21. Code `occ_lab/residual_diag.py`, `cache_oof.py`, `residual_oof.py`, `fit_corrector.py`,
`scripts/t2_residual_predict.py`. The user's diagnosis after note 85: the problem is not diversity but
**conditional competence** - STID rescues 0.27-0.31 of the blend's errors yet is a coin flip exactly where the
blend is unsure, so no member knows *when* it is better than the base.

## The change of target

Stop asking a new model to predict queues. Freeze the production blend and predict what it got wrong:

    p0      = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3
    r       = y - p0
    r_hat   = f(member probabilities, disagreement, |p0 - 0.5|, horizon, link position)
    p_final = clip(p0 + eta * r_hat where |r_hat| >= gate, 0, 1)

f is a 400-tree LightGBM. The point is that the easy cells - 98.9% of them, where p0 is already right - enter the
objective as r ~ 0, so the model is paid to leave them alone instead of re-deriving them.

## Month-to-month diagnostic

Fitted on one scored month, applied to the other (the members saw neither):

| fit -> apply | blend | corrected | delta |
|---|---|---|---|
| validation -> private | 0.8233 | 0.8312 | **+0.0079** |
| private -> validation | 0.8357 | 0.8474 | **+0.0117** |

and the error profile changed shape, which is the whole point:

| | rescue | spoil |
|---|---|---|
| STID as a member (note 85) | 0.265-0.307 | 0.0251 |
| residual corrector | 0.231-0.243 | **0.0024-0.0034** |

Roughly the same rescue at a tenth of the damage - the first thing measured that sits below the 0.0042
break-even. Fitting only on the cells the blend gets wrong destroys it (-0.039, and -0.79 at eta 0.5): there
r = +-1 by construction, so the model learns "always flip" and its gain collapses onto logit_blend (78%).

## The admissible version

A corrector fitted on April cannot be used on March. The train period's tail is the way out: members are fitted
on days before 2031-01-01 (`FIT_END`; the LightGBM through `ctx.fitd`, the nets through their `es` split, which
only selects a checkpoint), so **2031-01/02 is out-of-fold for every member and earlier than both scored
months**. `cache_oof.py` builds 3,881,167 eligible cells there (blend wrong on 1.14%), `fit_corrector.py` fits
the corrector, and one corrector serves both months.

| alpha (weight on blend-wrong cells) | validation | private |
|---|---|---|
| **0 (unweighted)** | **+0.0044 .. +0.0046** | **+0.0045 .. +0.0049** |
| 3 | -0.0027 .. -0.0037 | -0.0010 .. -0.0016 |
| 9 | -0.0103 .. -0.0105 | -0.0080 .. -0.0088 |

Every (eta, gate) of the unweighted fit is positive on both months; the worst is +0.0008 / +0.0009. Up-weighting
the blend-wrong cells raises rescue (0.21 -> 0.30) but raises spoil faster (0.0026 -> 0.0055) and loses. At
**eta 0.75, gate 0.10** the pair is +0.0044 / +0.0049 with rescue 0.150 / 0.163 and spoil 0.0022 / 0.0024.
Feature gain: position 28%, gs1 24%, |p0-0.5| 18%, occ_tfm 17%, logit_blend 14% - STID enters only through
`logit p_stid - logit p0`, which is how note 85 said it should.

Halving against the month-to-month diagnostic (+0.008/+0.012 -> +0.0045) is expected: 2031-01/02 is a different
season and a smaller pool.

## Candidate AI

Z with the ongoing decisions replaced by the corrected ones (`scripts/t2_residual_predict.py` -> ongoing rows
only -> `scripts/t2_patch_ongoing.py` -> `swap_task_rows`). **322 queue rows differ from Z**, onset untouched.
Local estimate +0.0047 ongoing family, i.e. about +0.0007 total - roughly seven times candidate AH's occupancy
mixture, and the largest local ongoing gain since candidate T.

Open risk: the corrector is fitted on our pseudo windows (52% history coverage, note 76) and applied to the
official ones (76%). Every member ships under the same mismatch, but the corrector reads member disagreement,
which the coverage difference moves directly.

## LB: AI = 0.87902 (submission 56410485), +0.00031 against Z

**A new best.** Z 0.87871 -> AI 0.87902. Local estimate was +0.0047 ongoing family, i.e. +0.0007 total; the
realised 0.00031 is a factor 0.44, the bottom of note 58's 0.44-0.76 band. The first LB gain from Task 2 since
candidate T - every ongoing change between T and AH either lost or landed inside the noise.
