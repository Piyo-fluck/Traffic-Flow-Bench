# 88. Logit-space updates for the residual corrector

Date 2026-09-22. Code `occ_lab/logit_update.py`, log `occ_lab/logs/logit_update.log` on Ada. The user's step 2
after note 86: r = y - p0 is the negative BCE gradient with respect to the logit, so the boosting update should be
applied in logit space.

## Variants (all on AI's 15 features and AI's training data, the tail's masked-layer windows)

    prob     p1 = clip(p0 + eta * r_hat)                    candidate AI
    logit    p1 = sigmoid(logit p0 + eta * r_hat)           the same r_hat as a logit step
    newton   p1 = sigmoid(logit p0 + eta * z_hat)           LogitBoost: z = (y - p0) / (p0(1-p0) + 1e-3), clipped
                                                            to +-4, weights p0(1-p0)

## Results (family IoU gain over the frozen blend; best setting, rescue / spoil)

| test | prob (AI) | logit | newton |
|---|---|---|---|
| Jan -> Feb, obs coverage | +0.0179 (0.323 / 0.0021) | **+0.0184** at eta 6 (0.367 / 0.0026) | +0.0174 (0.338 / 0.0022) |
| Jan -> Feb, full coverage | +0.0172 | +0.0172 | +0.0172 |
| tail -> March | +0.0046 (0.152 / 0.0022) | +0.0041 | **+0.0051** at 0.75 / gate 0.50 (0.149 / 0.0022) |
| tail -> April | **+0.0048** at 0.75 / gate 0 (0.181 / 0.0026) | +0.0042 | +0.0047 at 0.50 / gate 0 (0.149 / 0.0020) |

At one setting for both months: AI's prob 0.75 / 0.10 gives +0.0046 / +0.0042; newton 0.75 / 0.25 gives
+0.0051 / +0.0045.

## Reading

- Applying the same r_hat as a logit step does not help: it wins only inside the tail at obs coverage (eta 6) and
  loses on both scored months.
- The Newton / LogitBoost fit is the only variant ahead of AI on both months, by +0.0005 / +0.0003 family
  (about +0.00006 total), with a slightly lower spoil (0.0020-0.0022). That is small against the noise, and note
  87's AJ lost on the LB where local tests had it ahead.
- At official-like coverage (full) the three are identical (+0.0172).

## LB: AK = 0.87874 (submission 56452651), -0.00028 against AI, +0.00003 against Z

- AK = Z with the ongoing decisions from the Newton corrector (`occ_lab/fit_corrector_newton.py`,
  `occ_lab/models/residual_lgb_newton.txt`, `scripts/t2_residual_predict.py --update newton --eta 0.75
  --gate 0.25`). 375 queue rows differ from Z, 81 from AI.
- It lost to AI (0.87902) and is level with Z. Locally it was ahead of AI by +0.0005 / +0.0003 family. AI stays
  the best.
