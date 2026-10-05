# 19 — Task 2: dropping the train recurrence priors wins in the new run (+0.078 pseudo S_queue)

Date: 2026-09-14 · Script: `analysis/t2_shift_remedies.py` · Logs: `work/t2_shift_remedies_{validation,private}.log`

Protocol:
- Pseudo windows come from the validation month.
- Days 0–14 are adaptation data, and days 15+ are evaluation.
- Priors use past days only.

| Variant | onset | ongoing | pseudo S_queue |
|---|---|---|---|
| E (submitted) | 0.550 | 0.789 | 0.669 |
| E + month-past prior κ = 4 | 0.585 | 0.814 | 0.700 |
| **no priors** (models retrained with all prior features set to 0) | **0.672** | 0.822 | **0.747** |
| adapt (+ adaptation-day pseudo windows, weight 5, month priors) | 0.637 | 0.849 | 0.743 |
| adapt, no priors | 0.627 | **0.858** | 0.743 |

- The train-run recurrence priors (the top-gain features) mislead in a new simulation run. This is the
  location-embedding memorisation that the STAEformer literature warns about. Dropping them is fully
  compliant, since no month data is used, and cheap.
- Test-time adaptation on past-day pseudo windows helps ongoing most (0.858).
- Onset has only 27 evaluation windows, so its numbers are noisy.

## Private confirmation (log `work/t2_shift_remedies_private.log`)
| Variant | onset | ongoing | pseudo S_queue |
|---|---|---|---|
| E | 0.683 | 0.781 | 0.732 |
| E + month | 0.630 | 0.812 | 0.721 |
| no priors | 0.619 | 0.819 | 0.719 |
| **adapt** | **0.749** | 0.855 | **0.802** |
| adapt, no priors | 0.726 | 0.854 | 0.790 |

- **Test-time adaptation is the consistent winner**: +0.074 (validation) and +0.070 (private) pseudo
  S_queue over E, positive on onset in both months and +0.06–0.07 on ongoing.
- "No priors" helps ongoing in both months, but its onset flips sign (+0.12 validation, −0.06 private),
  so it is not robust on its own.
- **Caveat found in production (log `work/t2_adapt_predict.log`)**: 90.6% of the official validation/private
  windows are on days 1–7 of the month, and 98.8% on days ≤ 14 (onset max day 7 on validation and 9 on
  private; ongoing max 12 and 17). With past days only, most official windows have 0–6 days to adapt on.
  The weekly-block production therefore equals E + month prior for block 0, and changes only a handful of
  block-1 windows. The pseudo-eval above, which scored days ≥ 15 with 15 adaptation days, **overstated
  what is reachable**. The run-shift fix must be built into training instead (next: `analysis/t2_prior_robust.py`,
  scored on days < 14).
- Production must respect past days only. Use weekly blocks: the windows of week b are predicted by a
  model trained on train plus the pseudo windows of days before the block. Week 0 uses E with month
  priors. Script: `scripts/t2_adapt_predict.py`.
