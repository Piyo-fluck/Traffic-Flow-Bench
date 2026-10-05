# 147 Onset idea 3: shape-aware (block / copula) decoding of the T+30 set

Script: `occ_lab/onset_block_decode.py` (local). Base = production onset blend; swapped = note 145 prior4 swap.

Truth shape at T+30 (lab onset pool): 2.4 cells per window; over ELIGIBLE links only 1.16 blocks, 86% a single
block (the raw 1.58 blocks came from ineligible gaps). The ratio rule's set: 4.6 cells, holds 88% of the truth cells,
3% within 2 links of it, 9% elsewhere (another region).

Decoder: marginals kept; per candidate region a Gaussian copula latent rho Z_r + sqrt(1-rho^2) e_i; per-region level
sets (optionally hull = contiguous block) chosen by Monte-Carlo expected IoU, coordinate ascent from the ratio set.

| vs ratio rule (swapped) | val | priv |
|---|---|---|
| rho 0 | -0.0101 | -0.0077 |
| rho 0.8 | -0.0077 | +0.0004 |
| rho 1.0 | -0.0075 | +0.0036 |
| rho 0 hull | -0.0078 | +0.0033 |
| cross-month rho | -0.0075 | +0.0036 |

Same on the production probabilities (-0.004 .. -0.017 val). Even rho 0 (independent, the ratio rule's own model)
loses: optimising the exact expected IoU on these probabilities over-trusts them; the ratio rule's approximation is the
better decision for this calibration.

Eligibility-aware decoding (ineligible links are free in the IoU): eligibility at T+30 is not predictable from the
history (agreement 0.53-0.54; observed in the last 15 min misses 12% of eligible links -> -0.10); even the ORACLE
eligibility mask changes the ratio rule by -0.001 (base) / -0.009 (swapped). Negative.
