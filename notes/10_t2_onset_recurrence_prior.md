# 10 — Task 2 onset: predict only T+30 at the top recurrent links → IoU 0.025 → ~0.60

Date: 2026-09-14 · Script: `analysis/t2_eda03.py` · Log: `work/t2_eda03.log`
Builds on H9 and H10 ([09](09_t2_candidates_heuristics.md)): official onset
origins sit 30 min before onset, and onset happens at recurrent places.

## Setup
- **Earliest-onset candidates** mimic the official selection: no ongoing
  queue in the history, nothing queued at T+5 … T+25, and ≥ 1 queued cell at
  T+30. That gives 134–401 per panel (2,352 total), about one per day.
- **Priors**, leave-one-day-out and smoothed over ±2 slots (±10 min):
  - `P_q`: P(link queued at slot t | weekday vs weekend).
  - `P_on`: P(link is an onset cell, i.e. a queued T+30 cell of an
    earliest-onset origin, at slot t | weekday type).
- **Prediction**: 1 only at T+30, for the top-k links of the prior, or for
  the links above a threshold.

## Results (onset IoU, family mean over the 8 panels)
| Rule | Earliest-onset candidates | Official train onset windows (40) |
|---|---|---|
| Baseline (persistence) | 0.04 | 0.025 |
| P_q top-1 | 0.303 | 0.328 |
| P_q top-2 | 0.494 | 0.496 |
| P_q top-3 | 0.576 | 0.597 |
| **P_q top-4** | 0.583 | **0.606** |
| P_q top-6 | 0.528 | 0.561 |
| P_q > 0.1 | 0.470 | 0.490 |
| P_on top-2 | 0.531 | 0.550 |
| **P_on top-3** | **0.610** | 0.591 |
| P_on top-4 | 0.600 | 0.595 |

## Findings
- **F28**: Onset is overwhelmingly **recurrent**. Picking the 3–4 links with
  the highest time-of-day recurrence at T+30 recovers about 60% IoU. The
  history is not needed, and adds nothing at the speed level (F25).
- **F29**: Top-k beats thresholding. The truth is 1–8 cells, so a fixed
  small budget matches the target size. k = 3–4 is the sweet spot, and
  k = 6 already loses to false positives.
- The two priors are close. P_on is better on the large candidate set,
  P_q on the 40 official windows (small sample). A combined score, or a
  learned model, should do better than either.

## Impact estimate
S_queue = mean over (onset, ongoing): (0.60 + 0.77) / 2 ≈ **0.68**, against
0.30 for persistence, which is about **+0.11 total score**. That combines
the onset rule with the ongoing gap-filling rule of [09](09_t2_candidates_heuristics.md).

## Next
- Build the rule-based Task 2 submission. Ongoing: min v/vf over the last 3
  history steps < 0.8. Onset: P_on top-3 at T+30. The priors use all train
  days.
- Then learn it: a per-cell model for ongoing (H8), and a ranking model for
  onset combining P_q, P_on, the weekday, the history and the neighbours.
