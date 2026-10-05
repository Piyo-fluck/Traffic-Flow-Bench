# 81 - Ongoing: a cell-level stacker on the disputed cells, trained on the scored month's own past days

Date: 2026-09-20. Script: `analysis/t2_disputed_stack.py`. Follows note 80 (where the blend's error sits) and the
2026-09-20 clarification (any released data at or before T may be used, so the month's earlier days carry labels).

## Method

- The three ongoing members (gs1, tfm8, cnn3) agree on 97.1% of the eligible cells and are 99.3% right there.
  On the 2.9% they dispute, the production blend (0.2/0.6/0.2 at tau 0.50) is only 70-72% right.
- Only those disputed cells are re-decided, by a LightGBM fitted on the disputed cells of the **same split's
  earlier days** (expanding window, panels pooled, refitted for every day). Everything else is candidate T.
- Features: the three probabilities and their mean / spread / min / max, the horizon step, the link's speed ratio,
  flow over capacity and queued flag at the end of the history, the distance to the nearest link queued at T, the
  blend's neighbouring predictions, the link position and the panel id.
- Legality: labels come from released observations of days before the window's origin. Causal by construction.

## Results (official-like ongoing windows)

| setting | split | disputed-cell accuracy | ongoing family | window delta |
|---|---|---|---|---|
| 150 rounds, 31 leaves, from day 5 | validation | 0.703 -> **0.785** | 0.8357 -> 0.8488 (**+0.0131**) | +0.0171 [+0.0145, +0.0197] |
| | private | 0.715 -> **0.762** | 0.8233 -> 0.8319 (**+0.0086**) | +0.0092 [+0.0068, +0.0115] |
| 300 rounds, 63 leaves, from day 10 | validation | 0.703 -> 0.788 | +0.0133 | +0.0180 [+0.0153, +0.0207] |
| | private | 0.715 -> 0.767 | +0.0098 | +0.0111 [+0.0086, +0.0136] |

- Both months positive, window CIs well away from 0, and insensitive to the settings.
- Total-score equivalent: **+0.0020 / +0.0013**; after the note-58 halving, about +0.0007 to +0.0010 on the LB.
- The ceiling for this direction (perfect decisions on the disputed cells) is +0.012 total, so the stacker
  currently captures about a sixth of it.

## Why this differs from the gates and specialists that failed before

- Notes 51/53/65/73: those chose between whole windows or re-ranked onset candidates, on 169 windows of evidence.
  Here the unit is a cell, there are about 54,000 disputed cells per month, and the training labels come from the
  run itself.
- Note 62's band specialist replaced the models on a whole band of cells; this one only arbitrates where the
  members already disagree, and leaves the 97.1% they agree on untouched.

## Production build and the LB

- `scripts/t2_disputed_stack_predict.py` computes the three members on the OFFICIAL windows (gs1 from
  `work/sub_t2gs`, CNN x3 and transformer x8 from their checkpoints), finds the disputed cells and decides them
  with a stacker fitted on the pseudo windows of the same split on **earlier dates only**.
  `scripts/t2_cnn_blend_predict.py --og-pred-dir` takes those decisions for the ongoing windows.
- On the official windows: **1,667 disputed ongoing cells, 224 decisions changed** (validation 138 rows,
  private 86, over 28 windows).
- **LB: AD = 0.87848 (submission 56395893, 2026-09-20), -0.00023 against Z (0.87871).**
  - The local estimate was +0.0020 (validation family +0.0131), so this is the largest local-to-LB gap so far on
    the ongoing side, and it went the wrong way.
  - Scale: 138 changed rows on the public month, against 3.4 million ongoing cells in the pseudo-evaluation. The
    public LB sees 5 ongoing windows per panel.
- Z remains the best.

## Second version: trained on train-month windows (official coverage) - also negative on the LB

- Note 76's mismatch was traced further: the train month's TRUE layer has coverage 0.764 over eligible cells,
  exactly the official windows' 0.761. So the official histories are simply the unmasked observations, and
  train-month windows are the right training distribution.
- `scripts/t2_stack_train_month.py` fits one LightGBM on 800 ongoing windows per panel built from the true layer
  (65,602 disputed cells, 40% queued). Top gains: p_mean 0.46, p_gs1 0.15, dist_queue 0.10, pos 0.07.
- Honest check at official coverage (fit on train days before 2031-01, evaluated on the later ones):
  **disputed-cell accuracy 0.706 -> 0.841, ongoing family 0.9583 -> 0.9709 (+0.0126, total +0.0019), window
  +0.0128 [+0.0099, +0.0159]**.
- On the scored-month pseudo windows the same model is worse (0.703 -> 0.687), which is expected: those windows
  are 52% observed, a distribution this stacker never saw.
- Official windows: 491 of the 1,667 disputed cells changed (265 validation rows, 226 private).
- **LB: AE = 0.87755 (submission 56396428), -0.00116 against Z.**

## Where this leaves the stacker

| version | trained on | local check | LB |
|---|---|---|---|
| AD | the scored month's earlier pseudo windows (52% coverage) | pseudo windows +0.0131 / +0.0086 | 0.87848 (-0.00023) |
| AE | train-month windows (76% coverage, official-like) | train holdout +0.0126 | 0.87755 (-0.00116) |

Both local checks were positive and clear of 0, and both submissions lost, the better-matched one by more. The
disputed cells are 2.9% of the pseudo cells but only 1,667 cells over the 80 official windows, so on the LB this
is a very small, high-variance intervention.
