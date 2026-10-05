# 25 — Task 2: a month map from the other days does not fix onset; label noise is negligible

Date: 2026-09-15 · Scripts: `analysis/t2_month_loo_eval.py`, `analysis/t2_label_noise.py`
· Logs: `work/t2_month_loo_eval.log`, `work/t2_label_noise.log`

## Option B (measurement only): scoring priors from the same month

In-month pseudo windows with days < 14, family mean.
- "rule" = the top-3 links by P_on + 0.5·P_q at T+30.
- E = `sub_t2m1` models; R = `sub_t2robust` onset model.

| Scoring priors | validation onset: rule / E / R | E ongoing | private onset: rule / E / R | E ongoing |
|---|---|---|---|---|
| train table | 0.456 / **0.314** / 0.366 | 0.788 | 0.386 / 0.376 / **0.402** | 0.791 |
| month past days, κ = 4 | **0.546** / 0.313 / 0.335 | 0.796 | 0.401 / 0.314 / 0.345 | 0.801 |
| every other day (LOO), κ = 1 | 0.458 / 0.230 / 0.310 | 0.821 | 0.414 / 0.312 / 0.319 | 0.823 |
| every other day (LOO), κ = 4 | 0.484 / 0.228 / 0.361 | 0.818 | 0.422 / 0.301 / 0.348 | 0.825 |

- **Feeding the month map to the models makes onset worse** (E 0.31–0.38 → 0.23–0.31). The models were
  trained on train-run priors, and the month's P_q/P_on have a different scale and shape.
- The ongoing gain from month priors (+0.033 for E) only brings E up to the no-prior ongoing model
  (0.819 / 0.833). There is no net gain over G.
- The prior-only rule beats the models on validation onset (0.456–0.546 vs 0.314) but not on private. The two
  months disagree.
- **Conclusion: option B is not the missing piece.** Nothing to propose for submissions.

## Label noise (the official truth is the noise-free state)

Train, eligible cells, 3-slot median of the speed as a stand-in for the underlying state (family mean):
- queued cells within 1 / 2 km/h of the cut: 0.31% / 0.61%
- label flips: 0.03%
- IoU(noisy labels, smoothed labels): 0.992 over all cells, 0.985 over ongoing horizons

At most about 1.5% of ongoing IoU is at stake. Not a lever.

## First-morning-onset windows change the picture (`analysis/t2_first_onset_eval.py`, log `work/t2_first_onset_eval.log`)

One event per day: the first visible queue in 04:00–14:00 with none in the previous 90 min, with T = event − 30 min.
These are like the official onset windows.

| validation, days < 14, family mean | onset IoU |
|---|---|
| E | 0.534 |
| **G (R onset model)** | **0.643** |
| rule (train table) | 0.474 |
| location frequency of first onsets, top-3: train / past days / LOO | 0.413 / 0.379 / 0.494 |
| yesterday's onset links | 0.405 |

- **The generic pseudo windows understate onset (0.3–0.4) and compress the gaps between methods.** Per panel,
  G ≥ E on 8/8 validation panels (mean +0.11). On private (per panel; the family mean was NaN because
  D7_I10_W had no event), G > E on 5/7.
- The first-onset location repeats well within a month (the event hits the other days' modal link ±1 in
  75–86% of cases), but top-k location rules are weak. Onset cells per event are 1.5–4, and the exact links vary.
- Calibration check: E's official-like estimate, S_queue ≈ (0.534 + 0.79) / 2 ≈ 0.66, is close to the
  LB-implied 0.635. For G, ≈ (0.643 + 0.82) / 2 ≈ 0.73, i.e. about +0.02 total (≈ 0.846).
- Follow-up with official-like ongoing windows too (persistence IoU ≤ 0.9, coverage, hours):
  `analysis/t2_official_like_eval.py`.
