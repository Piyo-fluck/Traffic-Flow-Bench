# 11 — Task 2 rules: robust under a temporal holdout; onset gains from blending the priors; ongoing already optimal

Date: 2026-09-14 · Script: `analysis/t2_eda04.py` · Log: `work/t2_eda04.log`
Follows [09](09_t2_candidates_heuristics.md) and [10](10_t2_onset_recurrence_prior.md).

## Protocol
- **test**: candidates on Jan–Feb 2031 days, with priors fitted on
  Jun–Dec 2030 only. This mimics train → Mar/Apr. There are 33–87 onset and
  180–1348 ongoing candidates per panel.
- **official**: the official train windows (5 per condition and panel),
  with priors leave-one-day-out over all train days.

## Onset (grid: day group × smoothing × score × k)
| Rule | test | official |
|---|---|---|
| Current: weekday/weekend, ±2 slots, P_on, top-3 | **0.598** | 0.591 |
| **weekday/weekend, ±1 slot, P_on + 0.5·P_q, top-3** | 0.595 | **0.649** |
| all days pooled, ±1, P_on + 0.25·P_q, top-3 | 0.600 | 0.642 |
| weekday/weekend, ±1, P_on + 1.0·P_q, top-3 | 0.590 | 0.645 |
| … top-4 variants | 0.585–0.592 | 0.638–0.644 |

- **F30**: The recurrence rule survives a temporal holdout. Priors fitted
  on Jun–Dec predict Jan–Feb onsets at the same level as leave-one-day-out
  (about 0.60), so the seasonal drift is small. That is what makes the rule
  usable for Mar/Apr.
- Blending in P_q (the "is usually queued here" signal) and smoothing less
  (±1 slot) helps on the official windows (+0.06) and is neutral on test
  (−0.003). k = 3 stays best. Per-day-of-week priors (dow7) never made the
  top list, which suggests too few days per cell.
- Adopted: **weekday/weekend, smoothing ±1, score = P_on + 0.5·P_q, top-3**.

## Ongoing (grid: speed cut × history steps × extra onset links)
| Rule | test | official |
|---|---|---|
| **min v/vf over the last 3 steps < 0.8** (current) | **0.781** | **0.768** |
| < 0.8, last 4 | 0.781 | 0.768 |
| < 0.85 or < 0.75, last 3–4 | 0.780 | 0.766–0.767 |
| last 6 steps | 0.774–0.775 | 0.765–0.766 |
| + 1 onset-prior link at T+30 | 0.771 | 0.763 |

Per-step errors of the current rule on test candidates:

| Step | IoU |
|---|---|
| T+5 | 0.88 |
| T+10 | 0.85 |
| T+15 | 0.82 |
| T+20 | 0.79 |
| T+25 | 0.77 |
| T+30 | 0.74 |

False positives (queues dissolving) and false negatives (queues spreading)
grow at about the same rate.

- **F31**: The simple ongoing rule is at its optimum. What remains is
  per-link dynamics: which queues dissolve and which spread. That calls for
  a learned per-cell model (H8), not more thresholds.

## Expected Task 2 score with the refined rules
S_queue ≈ (onset 0.60–0.65 + ongoing 0.77) / 2 ≈ **0.69–0.71**, against 0.30
for persistence, which is about +0.12 total score.
