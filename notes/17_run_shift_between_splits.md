# 17 — Validation/private are a different simulation run: recurrent queues move (H17 supported)

Date: 2026-09-14 · Script: `analysis/t2_eda07_run_shift.py` · Log: `work/t2_eda07_run_shift.log`
Follows [16](16_lb_E_and_val_gap.md). The rebuild notice says validation/private were "generated again from new
random seeds", while the training month is unchanged.

## Test
Weekday queue-frequency maps P_q(slot, link), built at the same visibility (visible eligible cells of each
masked layer), over 05:00–20:00. Each map is compared with the map from train truth.

| Panel | Corr: train-vis / val / priv | Top-5 links per hour, Jaccard: train-vis / val / priv |
|---|---|---|
| D7_I10_E | 0.999 / 0.802 / 0.909 | 0.78 / 0.52 / 0.65 |
| D7_I10_W | 0.997 / 0.837 / 0.920 | 1.00 / 0.61 / 0.64 |
| D7_I210_E | 1.000 / **0.997 / 0.995** | 1.00 / **1.00 / 1.00** |
| D7_I210_W | 1.000 / 0.912 / 0.812 | 0.92 / 0.59 / 0.49 |
| D7_I405_N | 0.999 / 0.971 / 0.970 | 0.86 / 0.53 / 0.58 |
| D7_I405_S | 0.999 / 0.873 / 0.793 | 0.88 / **0.26 / 0.25** |
| D12_I5_N | 0.999 / 0.782 / **0.653** | 0.85 / 0.46 / 0.47 |
| D12_I5_S | 0.999 / 0.706 / 0.843 | 0.88 / 0.45 / 0.47 |

- Visibility alone barely changes the map (train-vis vs train: corr ≥ 0.997).
- Validation and private reshuffle where and when queues recur: the top-5 overlap falls to about
  0.25–0.65, except D7_I210_E, where one fixed, geometry-driven bottleneck is identical in all runs.

## Findings
- **F34**: Train priors transfer only partly to the scored months. That explains the systematic
  validation shortfall of the prior-heavy Task 2 methods (rules −0.13, model −0.20 against the holdout).
  Our holdout could not show it, because train is one continuous run.
- **F35**: The same shift likely hurts Task 1. The CNN's base channels (weekday × tod profile) come from
  train, so on validation/private the anomalies it sees are relative to a partly wrong baseline.

## Within-run check of the T1 month base (log `work/t1_month_base_check.log`)
All-panel CNN on the Jan–Feb holdout. The base is the fit-day profile, plus the calendar month's own
deviation shrunk by κ pseudo-days.

| κ | S_state | S_phys~ | 0.35·S_state + 0.15·S_phys~ |
|---|---|---|---|
| none | 0.9523 | 0.7616 | 0.4475 |
| 10 | 0.9517 | 0.7603 | 0.4471 |
| 4 | 0.9510 | 0.7579 | 0.4465 |
| 2 | 0.9504 | 0.7559 | 0.4460 |

Train is one run, so there is no run shift to correct and the deviation adds only noise. At κ = 10 the
cost is small (−0.0004). The benefit can only show inside a shifted month, so the next step is the
pseudo-evaluation inside validation/private.

## First T2 month-prior attempt
`scripts/t2_m1_predict.py --month-kappa 4` with the saved models changes the predictions little.
Overlap with the train-prior predictions is onset 0.90–0.95 and ongoing 0.97. P_on, the top onset
feature, stays train-only, and the blend dilutes the month signal. To be judged by the pseudo-evaluation.

## Options
- **T1 (offline task; the whole evaluation period is explicitly allowed)**: build each month's own
  weekday-type × tod profile from its visible masked-layer cells, and use it as the CNN base. Retrain with
  per-month profiles on train too, so training and inference match.
- **T2 (online task)**: build the recurrence priors from the scored month's own released data. This needs a
  decision on scope:
  - (a) strictly past days of that month, which is fully online;
  - (b) all other days of the month, which includes later days;
  - (c) keep train-only priors.
