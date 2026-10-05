# 94. A residual corrector for onset: closed

Date 2026-09-23. Code `occ_lab/cache_onset_oof.py`, `occ_lab/onset_residual.py`. The ongoing corrector (note 86)
is the only thing that ever moved the LB from the ongoing side, so the same construction was tried on the onset
half, on top of the blend candidate AN ships.

    p0      = 0.35 H + 0.35 ad9b + 0.30 STID
    r_hat   = f(member probabilities, disagreement, |p0 - 0.5|, hour, link position)   fitted on the tail
    p_final = clip(p0 + eta * r_hat), then the expected-IoU ratio rule as usual

## The material is thin

The out-of-fold onset material - the tail's onset windows, one per day by the official selector - is

    415 windows, 60,875 eligible links, 1,460 of them queued (2.34%)

against the ongoing corrector's 3,881,167 cells: **64 times thinner**. `--extra` (taking every other origin that
satisfies the selector) added almost nothing, because the condition admits about one origin per day. The fitted
corrector's gain is 54% on H alone - it mostly re-weights the blend toward LightGBM.

## It loses at every strength

| eta | validation | private | set size change |
|---|---|---|---|
| AN blend | 0.8092 | 0.8564 | 4.56 / 4.41 links per window |
| 0.10 | -0.0190 | -0.0005 | 0.04 links |
| 0.25 | -0.0188 | -0.0006 | 0.08 |
| 0.50 | -0.0188 | -0.0268 | 0.14 |
| 1.00 | -0.0238 | -0.0431 | 0.22 |

At eta 0.10 the predicted set changes by 0.04 links per window on average and validation still falls 0.019.
**Onset is decided by a top-k set rule, so a correction that moves k by one in a handful of windows moves the
family mean** - there are only 169 / 162 onset windows per month. This is the same brittleness the seed
experiment showed (note 93): what matters for onset is the SCALE of the probabilities, not only their order.

Branch closed. The onset half keeps AN's three-member blend.
