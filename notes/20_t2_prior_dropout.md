# 20 — Task 2: prior dropout (domain randomisation) gives a consistent gain on the official-window days

Date: 2026-09-15 · Script: `analysis/t2_prior_robust.py` · Log: `work/t2_prior_robust.log`

The official windows sit on days 1–14 (note 19), so no scored-month data is available to adapt on. The fix is
built into training instead, and scored on in-month pseudo windows with days < 14.

| Model | validation onset / ongoing / S_queue | private onset / ongoing / S_queue |
|---|---|---|
| E (train priors) | 0.314 / 0.788 / 0.551 | 0.376 / 0.791 / 0.584 |
| no priors | 0.324 / **0.817** / 0.571 | 0.345 / **0.830** / 0.587 |
| **prior dropout (p = 0.5)** | **0.354** / 0.812 / **0.583** | **0.410** / 0.815 / **0.613** |
| shuffled priors (half the days, ±2 slots) | 0.333 / 0.788 / 0.560 | 0.363 / 0.795 / 0.579 |

- Prior dropout is the only variant that improves both conditions in both months: +0.032 and +0.029
  S_queue. The model learns how far to trust a prior instead of assuming it is always right.
- Best combination: **onset from the dropout model, ongoing from the no-prior model**. That gives about
  0.586 and 0.620 S_queue, +0.035 over E, or about +0.01 total. It uses no scored-month data.
- The absolute onset level stays low (0.35–0.41) in a new run. The "where does it start" problem is still
  open, and physics-based forecasting is the next step.
