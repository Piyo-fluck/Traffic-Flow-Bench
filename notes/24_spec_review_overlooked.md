# 24 — Spec review ("what are we overlooking?"): hypotheses checked against the official docs, code and data

Date: 2026-09-15. Sources:
- `trafficflowbench-public/README.md`
- `docs/SCORING_SPEC.md`, `docs/MASK_SPEC.md`, `docs/DATA.md`, `docs/TASK3_LWR_COVERAGE_MODES.md`
- `src/task2/score_task2.py`, `src/task3/score_task3.py`
- `data/kaggle_public/config/synthetic_release_v1.json`
- raw parquet files (D12_I5_N)

## Hypotheses rejected

| Hypothesis | Check | Result |
|---|---|---|
| Occupancy survives at Task 1 target cells, so density could be recovered | D12_I5_N train, R3 day: masked vs unmasked | `occupancy` present at 0.000 of blanked eligible cells |
| The `density_occ_linear_vehpkm` column survives at targets | same | 0.000 present. The column is released but we never loaded it; at visible cells corr with q/v = 0.970 |
| The official T3 state includes ineligible cells | `state_for_regime` | Filters `is_score_eligible` (line 64). The proxy matches |
| The Task 2 history includes slot T | `window_history.parquet` | 12 slots, T−60 … T−5. There are no rows at T, so our handling is correct |

## Facts that matter

- **Task 2 truth comes from the noise-free underlying state**, not the noisy observation (SCORING_SPEC).
  Our train labels come from the noisy observed layer. The effect is small near the 0.6·vf cut, but it is not
  zero.
- **Window selection** (`synthetic_release_v1.json`):
  - 5 onset and 5 ongoing windows per corridor and split; origins at least 360 min apart;
    `minimum_history_coverage` 0.7.
  - `maximum_persistence_iou` 0.9: ongoing windows where persistence scores above 0.9 are excluded, so ongoing
    windows are ones where the queue changes.
  - The release says "normal windows almost only at the morning peak onset".
- **Official onset origins sit at 04–14 h** (timestamp clock), which is the morning-peak onset. **97.5% of onset
  windows and 83% of ongoing windows fall on days 1–7.** A past-days-only month prior therefore has 0–6 days
  of data.
- The morning first-queue location should repeat from day to day within a run (same demand draw), but it moves
  between runs (note 17). A queue map from the other days of the same month addresses exactly that. Its effect
  is measured in `analysis/t2_month_loo_eval.py` (option B, measurement only).
- Other items:
  - `scenario_contract.capacity_multiplier = 1.2` and `incident_severity = 1.0`. Validation has 5 incidents,
    private 7.
  - Ineligible cells are released with values in about 30% of cases (pct_observed 0–67).
  - The window history also carries `occupancy`, `pct_observed` and ineligible cells. Our T2 features use none
    of them (only eligible speed and flow).
- A perfect Task 1 answer scores S_physics 0.9598 (README, D12_I5_N), i.e. S_LWR ≈ 0.95. The noisy truth
  satisfies conservation exactly, so S_LWR rewards reproducing the noise itself. A noise-free predictor only
  reaches about 0.66 (note 23). This is the same for every participant.
