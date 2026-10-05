# 15 — Task 2 learned models: S_queue 0.688 → 0.836 on the temporal holdout (+0.148 over rules)

Date: 2026-09-14 · Script: `analysis/t2_m1_models.py` · Log: `work/t2_m1_models.log`
Follows the user's call (after [14](14_t2_ramp_route_proof_negative.md)): the gap to the top is method,
so the methods need to be refined.

## Negative probes first
- **Flow precursor for onset (H14)**, `analysis/t2_eda06_onset_precursors.py`: q/cap and k/kcrit barely
  separate onset links (AUC ≈ 0.59). Onset links sit at k ≈ 0.46–0.9·kcrit, not near critical. Downstream
  density helps a little (AUC 0.674). The recurrence prior dominates (AUC 0.945). Naive products with
  the prior hurt the top-3 pick (0.586 → 0.45).
- **Temporal blending post-process for T1/T3**, `analysis/t1_post_smooth.py`: it lowers both S_state and
  S_LWR~ at every weight (w = 0.1 costs −0.0013 in 0.35·S_state + 0.15·S_phys). The CNN is already
  time-optimal, so blending only adds bias.

## Models
Pooled over the 8 Task 2 panels, LightGBM binary, history T−60 … T−5 plus priors only.
- Days: Jun–Nov trains, Dec early-stops and tunes the decisions, Jan–Feb tests.
- Priors come from Jun–Dec. They are leave-one-day-out inside Jun–Dec and plain for Jan–Feb.

**Onset ranker**, per (window, link) at T+30.
- 298k rows; best iteration 528.
- Features: link history (v/vf, q/cap, k/kcrit, queued flags, trends, neighbours ±1/2/4), distance to
  slow links, static attributes, time of day, day group, panel, and the priors (P_q at T+25/30/35, P_on,
  neighbour priors, rank and relative prior within the window).
- Top gain: `pq30_rel` 0.38, `pon30` 0.31.
- **Decision: k = round(1.2 · Σp) per window, clipped to 1 … 6.** On Dec this gives 0.876, against
  0.674 for the best fixed k. The true onset set size varies from 1 to 8, and Σp estimates it.

**Ongoing per-cell model**, per (window, step, link).
- 2.86M sampled rows: cells within 10 stations of a slow link, queued cells, and 2% of the rest.
  Best iteration 2889 (cap 3000).
- Top gain: `vr_last` 0.55, `kc_last` 0.12, `pq_h` 0.07, `pq_diff` 0.06.
- Decision: p ≥ τ, with τ = 0.6 (the edge of the grid; to extend).

## Results (Jan–Feb 2031 windows, family mean over 8 panels)
| | Rules v2 | Model |
|---|---|---|
| Onset | 0.5945 | **0.7754** |
| Ongoing | 0.7810 | **0.8964** |
| **S_queue** | 0.6877 | **0.8359** |

Per panel, onset / ongoing (rules → model):

| Panel | Onset | Ongoing |
|---|---|---|
| D7_I10_E | 0.410 → 0.776 | 0.770 → 0.885 |
| D7_I10_W | 0.530 → 0.675 | 0.811 → 0.823 |
| D7_I210_E | 0.612 → 0.680 | 0.953 → 0.985 |
| D7_I210_W | 0.611 → 0.865 | 0.795 → 0.910 |
| D7_I405_N | 0.656 → 0.958 | 0.747 → 0.897 |
| D7_I405_S | 0.405 → 0.696 | 0.783 → 0.890 |
| D12_I5_N | 0.840 → **0.798** | 0.693 → 0.884 |
| D12_I5_S | 0.692 → 0.754 | 0.696 → 0.897 |

## Findings
- **F32**: Choosing *how many* onset links to flag per window is the big onset win. A calibrated model's
  Σp estimates the set size; a fixed k cannot adapt to windows with 1 vs 8 onset links.
- **F33**: For ongoing, the per-link state at T−5 (v/vf, k/kcrit) plus the recurrence prior at the
  target step, and its change from the history time, separate persisting from dissolving cells far better
  than a fixed cut.
- Expected LB effect: validation was a hard month (rules ≈ 0.56). If the model keeps the same relative
  gain, T2 on validation is about 0.70–0.84, which is **+0.04 to +0.08 total**.

## Extended grid (log `work/t2_m1_models_grid2.log`)
- Onset α on Dec: 1.1 → 0.872, **1.2 → 0.876**, 1.3 → 0.852. It stays at 1.2.
- Ongoing τ on Dec: 0.6 → 0.896, **0.65 → 0.898**, 0.7 → 0.897. The test ongoing IoU becomes 0.898.
- The ongoing model early-stops at 2889 even with a 5000-round cap, so it is not under-trained.
- **Final holdout S_queue: 0.8367** (rules 0.6877).
- Production settings: onset 600 rounds, ongoing 3300 rounds (about +15% for about 25% more
  training days), α = 1.2, τ = 0.65.

## Next
- Done: extend the τ and α grids, and raise the ongoing round cap (see above).
- Production: train on every train day, and build features for validation/private from
  `window_history.parquet`.
