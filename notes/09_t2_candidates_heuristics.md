# 09 — Task 2: gap-filling persistence lifts ongoing to 0.77; onset has no speed precursor

Date: 2026-09-14 · Script: `analysis/t2_eda02.py` · Log: `work/t2_eda02.log`
Candidate windows: every train day, origins every 30 min, same condition
rule as the release, at least one queued horizon cell. That gives 153–479
onset and 663–6224 ongoing candidates per panel. Local labels as in
[08](08_t2_structure_and_persistence.md).

## Heuristics (IoU, family mean over the 8 Task 2 panels)
| Heuristic | Candidate onset | Candidate ongoing | Official-train onset | Official-train ongoing |
|---|---|---|---|---|
| persist (queued at T−5) | 0.041 | 0.637 | 0.025 | 0.590 |
| persist, any of the last 2 steps | 0.043 | 0.751 | 0.025 | 0.716 |
| persist, any of the last 3 steps | 0.043 | 0.769 | 0.025 | 0.736 |
| persist + 1 station upstream | 0.038 | 0.694 | 0.025 | 0.646 |
| persist growing upstream (0.25 stations/step) | 0.037 | 0.691 | 0.025 | 0.644 |
| **min v/vf over the last 3 steps < 0.8** | **0.105** | **0.774** | 0.025 | **0.768** |
| min v/vf over the last 3 steps < 0.7 | 0.074 | 0.772 | 0.025 | 0.759 |
| v/vf at T−5 < 0.75 | 0.094 | 0.641 | 0.025 | 0.611 |

## Findings
- **F24**: For ongoing windows the win is gap filling. Using the last 3
  steps (or a slightly looser speed cut, 0.8·vf) instead of the single T−5
  cell lifts the official-window IoU from **0.59 → 0.77**. The T−5 cell is
  often missing or ineligible (17% of history cells are), and persistence
  then drops that link. Upstream dilation or growth *hurts*: queues in this
  data do not grow steadily upstream within 30 minutes.
- **F25**: Onset has **no speed precursor** in most panels. The links that
  queue in the horizon have v/vf at T−5 of about **1.0** (median; q10
  0.87–0.98). Only D7_I10_W shows a gradual drop (median 0.73). The
  transition from free flow to below 0.6·vf happens within about 10 minutes.
- **F26**: Onset timing concentrates at particular times of day. On
  D7_I210_E, 279 of 292 onset candidates start at the first horizon step,
  with origins at 6:30–7:00. This points to recurrent, scheduled bottleneck
  activation, so a **link × time-of-day prior** should carry the signal the
  history lacks.
- **F27**: The official onset windows differ from our candidates. In the
  local labels they hold only 1–5 queued horizon cells, against a candidate
  median of 9–31, and no heuristic scores on them (0.025 comes from the one
  window where the local labels hold nothing). Their selection rule needs
  to be understood before tuning on candidates.

## H10 confirmed: official onset origins sit exactly 30 min before onset
Profile of the 40 official train onset windows (local labels; `work/t2_official_windows_profile.csv`):
- **In 39/40 the first queued horizon step is 6 (T+30)**. The remaining
  window has no locally-queued cell, a label-noise case. Each queued link
  appears only at that last step: n_true = n_links = 1–8, median 3.
- The true onset cells sit at recurrent places. Their link × tod ×
  weekday-type queue frequency has median **0.34** (IQR 0.23–0.53), against
  0.19 for the 99th percentile over all horizon cells.
- Origins fall between 4:40 and 14:35. Weekends are frequent (16/40).
- Ongoing windows are big PM queues: n_true mean 158, origin hour median 16.5.

**Consequence**: in an onset window the only cells that can be queued are at
**T+30**, so predictions at T+5 … T+25 can only add false positives. The
task reduces to picking the few links where a queue starts at T+30, and the
recurrence prior is the main signal.

## Hypotheses
- **H8**: A learned per-cell model (history of the link and its neighbours,
  distance to slow links, time of day, recurrence prior) beats the 0.77
  ongoing heuristic.
- **H9**: Onset is predictable mainly from recurrence (link × tod ×
  weekday-type priors), not from the history.
- **H10**: Official onset windows are the *earliest* onset origins (the
  queue appears late in the horizon), which would explain their tiny
  targets.
