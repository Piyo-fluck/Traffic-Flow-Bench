# 26 — Task 2: an official-like pseudo-eval reproduces E's LB and puts G at about +0.02 total

Date: 2026-09-15 · Script: `analysis/t2_official_like_eval.py` · Log: `work/t2_official_like_eval.log`

## Windows (mimicking `config/synthetic_release_v1.json` and the released window_index; see note 24)

- **Onset**: the first queue of the morning (04–14 h). No visible queue in the previous 90 min, and the queue
  at T+30. One event per day.
- **Ongoing**: a visible history queue and a queue in the horizon; persistence IoU ≤ 0.9; eligible history
  coverage ≥ 0.7; origins at 07–21 h.
- Scored on days 1–7 (the official days) and on days 1–14 (more samples), with the saved models. E =
  `sub_t2m1`; G = `sub_t2robust` (onset trained with prior dropout, ongoing with no priors).

## Results (family mean)

| | E onset / ongoing / S_queue | G onset / ongoing / S_queue | ΔS_queue | Δtotal |
|---|---|---|---|---|
| validation, days 1–7 | 0.391 / 0.731 / 0.561 | 0.472 / 0.755 / 0.613 | +0.052 | +0.016 |
| validation, days 1–14 | 0.534 / 0.729 / **0.631** | 0.643 / 0.758 / 0.700 | +0.069 | +0.021 |
| private, days 1–7 | 0.443 / 0.739 / 0.591 | 0.539 / 0.778 / 0.658 | +0.067 | +0.020 |
| private, days 1–14 | 0.540 / 0.743 / 0.642 | 0.614 / 0.775 / 0.694 | +0.053 | +0.016 |

- **Calibration**: E's validation estimate (0.631, days 1–14) matches the LB-implied S_queue of about 0.635
  (note 16). This is the first pseudo-eval that reproduces the LB level of Task 2.
- G is ahead on onset in 11 of 16 panel-splits and on ongoing in 12 of 16. The gain holds in both months.
- The generic pseudo windows (notes 18–22) put G at only about +0.01 total and ranked some variants
  differently. **Decisions made on the generic windows need re-checking on these windows**, notably the stress
  features, bagging and thresholds.
- **H = G + 5 onset bags** (`work/sub_t2robust_bag5`, log `work/t2_official_like_eval_H.log`). The ongoing model is
  identical to G's.

  | Δtotal vs E | val d1–7 | val d1–14 | priv d1–7 | priv d1–14 |
  |---|---|---|---|---|
  | G | +0.0156 | +0.0206 | +0.0201 | +0.0158 |
  | H | +0.0148 | +0.0220 | +0.0228 | +0.0204 |

  H ≥ G in 3 of 4 conditions, with a mean of about +0.020 total. H + T1 month base (F1) ≈ 0.8254 + 0.020 + 0.0025 ≈ **0.848**.
- **I = H + demand-stress features** (`work/sub_t2robust_bag5_stress`, log `work/t2_official_like_eval_I.log`)

  | onset / ongoing | val d1–7 | val d1–14 | priv d1–7 | priv d1–14 |
  |---|---|---|---|---|
  | H | 0.466 / 0.755 | 0.652 / 0.758 | 0.557 / 0.778 | 0.645 / 0.775 |
  | I | 0.449 / 0.774 | 0.637 / 0.773 | 0.490 / 0.782 | 0.583 / 0.785 |

  Stress helps ongoing in all 4 conditions (+0.004 … +0.019, as in notes 22) and hurts onset in all 4.
- **J = H's onset + I's ongoing** (per window condition; `work/sub_t2J`). Estimated Δtotal vs E:
  +0.0177 / +0.0245 / +0.0236 / +0.0219 (val d1–7 / val d1–14 / priv d1–7 / priv d1–14). This picks the
  better component per condition after seeing the results, which carries a mild selection bias. The stress
  effect on ongoing was consistent in every earlier experiment.
- Estimate: G alone ≈ 0.841–0.846 LB. With the T1 month base (+0.0025) and small extras, that is about
  0.845–0.85. The next candidates are H (G + 5 onset bags) and I (H + stress features), each checked on these
  windows.
