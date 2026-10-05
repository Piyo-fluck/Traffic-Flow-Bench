# 08 — Task 2: window structure, local labels validated, and onset windows score ~0

Date: 2026-09-14 · Script: `analysis/t2_eda01.py` · Log: `work/t2_eda01.log` · Per-window table: `work/t2_eda01_windows.csv`
Scorer read: `src/task2/score_task2.py`, `src/task2/queue_utils.py`

## How Task 2 is scored
- The truth file has one row per (window_id, timestamp, link_id) with
  `queue_true` and `is_score_eligible`. IoU is computed **on eligible cells
  only**. A missing prediction counts as 0.
- queued ⇔ speed ≤ v_cut. `fd_parameters.csv` has no `v_cut` column, so
  v_cut = **0.6 × links.csv free_speed_kmh**. That value equals the fd free
  speed on every link of every panel (max diff 0.000).
- The mean is taken over windows, then over the two conditions per panel,
  then over panels, then over the 4 families (the D12_I405 panels are
  excluded).

## Window structure (train; validation and private have the same shape)
- There are 10 windows per panel: 5 `queue_onset` and 5 `queue_ongoing`.
- History is **12 steps, T−60 … T−5 min**. The horizon is **T+5 … T+30**
  (6 steps). The origin slot T itself is in neither.
- `window_history.parquet` equals the unmasked truth: the raw observation,
  with no Task 1 mask. Ineligible cells are NaN or flagged.
- In the *train* masked layer, the horizon and the 60-min buffer are **not**
  blanked: the blank share there is 0.35–0.50, the normal blank rate. Train
  truth is valued for about 83% of horizon cells, so local labels can be
  built.

## Local labels are a faithful proxy
Local label: truth speed ≤ v_cut on eligible cells. That is the noisy
observation, not the organizer's noise-free state. With it, persistence (the
queue at T−5 carried over the horizon) scores family mean **0.3074** on the
train windows. The official validation figure for that baseline is 0.3017
(0.2518 in the repo README).

## Findings
| Condition | Persistence IoU | Queued cells in horizon (mean) | Queued links at T−5 | Typical origin |
|---|---|---|---|---|
| queue_ongoing | **0.59** (0.45–0.71 by panel) | 20–478 | 3–88 | 15–18 h (PM peak) |
| queue_onset | **0.025** | **1–5** | 0 | 7–12 h |

- **F22**: Half of S_queue is the onset condition. Persistence scores about
  0 there, because nothing is queued in the history. An onset window's truth
  is tiny (1–5 cells out of 6 × S), so a few correct cells give a large IoU,
  and a wide net gives a small one.
- **F23**: Ongoing windows are dominated by large, persistent PM queues.
  Persistence already gets 0.59, and the misses are growth/shrink at the
  queue ends.
- The biggest lever in the whole competition is **onset detection** (+0.15
  S_queue ≈ +0.045 total at IoU 0.3), followed by better ongoing
  propagation.

## What the validation/private release shows around a window, and our usage rule
Masked-layer blank share around the windows (D12_I5_N, D7_I210_W; both splits):

| Span | Blank share | Meaning |
|---|---|---|
| T−3h … T−60 | 0.34–0.45 | normal Task 1 masking: visible |
| T−60 … T−5 (history) | 0.17 | raw observation, no Task 1 mask (same as `window_history`) |
| **T itself** | 0.34–0.49 | visible at the normal rate |
| T+5 … T+30 (horizon) | **1.00** | blanked |
| T+35 … T+90 (buffer) | **1.00** | blanked |
| T+95 … | 0.35–0.45 | visible again |

**Rule we adopt (conservative)**: Task 2 predictions use only the window's
history (T−60 … T−5) plus priors learned from train. The organizers' own
code calls reading the state at T "not part of the Task 2 participant
contract" and "one extra observation leaked" (comment in
`build_task2_persistence_submission.py`). Anything after the horizon is
future information. Using same-day data before T−60 is left unused until we
decide otherwise explicitly.

## Next
- Generate thousands of candidate windows from train truth, using the same
  condition rule (ongoing ⇔ ≥ 2 queued visible cells with ≥ 2 on one link),
  to study onset precursors and to train models.
- Heuristics to test: queue-end dilation upstream for ongoing; a
  near-threshold speed-drop detector for onset.
