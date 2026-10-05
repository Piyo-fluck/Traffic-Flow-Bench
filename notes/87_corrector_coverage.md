# 87. The residual corrector under a change of history coverage

Date 2026-09-22. Code `occ_lab/cache_cov.py`, `occ_lab/residual_cov.py`, log `occ_lab/logs/residual_cov.log`
on Ada. The user's step 1 after note 86: candidate AI's corrector was fitted on train-tail windows about 0.50
observed and is applied to official windows about 0.765 observed (note 76), while it reads |p0 - 0.5| and member
disagreement, which move with coverage.

## Environments

The same 2031-01/02 held-out windows as `cache_oof.py` (same selection, 8 panels), every member re-inferred under
five history availabilities. Labels are unchanged (masked-layer queues on eligible cells), so only inputs differ.

| env | how | mean coverage |
|---|---|---|
| obs | the masked layer as released - what AI was fitted on | 0.495 |
| c50 / c60 / c70 | per (window, link), a share lambda of links keep the released mask, the rest see the true layer; whole links switch, so gaps keep their temporal structure | 0.505 / 0.600 / 0.700 |
| full | the true layer on every eligible cell | **0.764** (official ongoing 0.765) |

Where a panel's released mask is already above 0.50, c50 equals obs (lambda 1.00).

## Correctors

All unweighted L2 on r = y - p0, 400 trees (AI's recipe). `coverage` = the window's observed share, per window.

    AI         fitted on obs, AI's 15 features
    full       fitted on full, + coverage
    all        fitted on all five environments, + coverage
    all-nopos  as all, without the link-position feature

## Test 1: fit on 2031-01, score 2031-02 in every environment (gain over the frozen blend, family IoU)

At AI's setting (eta 0.75, gate 0.10), best setting in brackets:

| | obs | c50 | c60 | c70 | **full** |
|---|---|---|---|---|---|
| blend | 0.8725 | 0.8736 | 0.8835 | 0.8918 | 0.8958 |
| AI | +0.0168 | +0.0180 | +0.0161 | +0.0170 | **+0.0172** (+0.0172) |
| full | +0.0098 | +0.0095 | +0.0110 | +0.0162 | +0.0188 (+0.0226) |
| **all** | **+0.0177** | **+0.0191** | **+0.0183** | **+0.0188** | **+0.0212** (+0.0248) |
| all-nopos | +0.0128 | +0.0124 | +0.0125 | +0.0120 | +0.0162 (+0.0165) |

- **AI does not collapse at official coverage**: +0.0168 at 0.50 and +0.0172 at 0.76. The mismatch the user
  worried about costs AI nothing measurable here - consistent with AI's +0.00031 on the LB.
- **Training across coverages is better everywhere**, and most at official coverage: `all` beats AI by +0.0040 at
  AI's setting and by +0.0076 at the best setting in `full`.
- Training on `full` alone loses at low coverage: a corrector fitted on one coverage does not extrapolate far.

## Test 2: fit on the whole tail, score the two months (pseudo windows, about 0.52 covered)

| | validation | private |
|---|---|---|
| AI | +0.0046 | +0.0042 |
| all | +0.0048 | +0.0041 |
| full | -0.0014 | -0.0025 |
| all-nopos | +0.0006 | +0.0002 |

- At 0.52 coverage `all` equals AI, as it should - the scored-month pseudo windows cannot show a coverage gain.
- `full` loses: fitted at 0.76, applied at 0.52 - the mirror image of the risk the user raised, and it is real
  when the gap is that large in that direction.

## Two findings beyond the question

1. **Position is not a shortcut - it is the part that transfers.** Without it the corrector keeps +0.012-0.016
   inside the tail (test 1) but falls to +0.0002-0.0006 on March and April (test 2). What carries across seasons
   is the per-link bias of the blend (position plus the panel's member profile identify the link); what the other
   features learn is mostly season-specific.
2. **In-season gains are four times the cross-season ones**: January -> February +0.017-0.025, the tail -> March
   / April +0.004-0.005. The corrector is data-hungry for recent data. For a window on day d the rule allows every
   released observation before its origin, including the scored month's own earlier days - the same causal
   past-day scheme as idea 9b (note 67), which has not been tried for the corrector.

## LB: AJ = 0.87820 (submission 56446463), -0.00082 against AI, -0.00051 against Z

- AJ = Z with the ongoing decisions from the `all` corrector (five environments + coverage,
  `occ_lab/fit_corrector_cov.py`, `occ_lab/models/residual_lgb_cov.txt`), eta 0.75, gate 0.10 as in AI.
  Official-window coverage read 0.75-0.77, inside the training range. 321 queue rows differ from Z, 199 from AI.
- It lost to both AI (0.87902) and Z (0.87871). The in-tail test predicted the opposite at official coverage
  (+0.0212 vs AI's +0.0172); the scored-month test called it a tie. AI stays the best.
