# 53 — T2 ONGOING: the per-window threshold is partly learnable (+0.015); per-CELL thresholds add nothing

Date: 2026-09-16 · Scripts: `analysis/t2_ongoing_tau_gate.py`, `analysis/t2_ongoing_cell_tau.py` ·
Logs: `work/t2_ongoing_tau_gate.log`, `work/t2_ongoing_cell_tau.log` (12 s and 11 s)

Note 52 measured an oracle of **+0.080 ongoing IoU** for a per-window threshold. This note measures what is
reachable. Every rule is **fitted on one scored split and applied to the other, in both directions**, so the numbers
are out-of-sample. That is a derivation/evaluation device under the user's rule (derive and evaluate on a scored
month: yes; train a submitted model on it: no); a production gate must be fitted on the train run.

## 1. Window-level rules (`t2_ongoing_tau_gate.py`)

| rule (fitted on the other split) | fit val → priv | fit priv → val | share of the oracle |
|---|---|---|---|
| production τ 0.60 | 0.8154 | 0.8207 | — |
| global τ (fitted) | +0.0093 [+0.007, +0.012] | +0.0118 [+0.009, +0.015] | 0.11 / 0.15 |
| τ per panel | +0.0083 | +0.0135 | 0.10 / 0.17 |
| τ per queue-size quintile | +0.0079 | +0.0118 | 0.10 / 0.15 |
| τ per mean-probability quintile | +0.0090 | +0.0113 | 0.11 / 0.14 |
| τ per hour quintile | +0.0058 | +0.0118 | 0.07 / 0.15 |
| **LightGBM window features → best τ** | **+0.0144 [+0.011, +0.018]** | **+0.0176 [+0.014, +0.021]** | **0.18 / 0.22** |
| LightGBM → argmax predicted gain | +0.0109 | +0.0144 | 0.13 / 0.18 |
| ORACLE per-window τ | +0.0808 | +0.0803 | 1.00 |

- **A learned per-window threshold beats the best fixed threshold in both directions**, by +0.005 ongoing. It
  captures about a fifth of the oracle. Features: queue size and trend at T, number of queue runs, probability
  mean/max/mass/entropy, the share of cells above 0.4–0.7, coverage, hour, weekday, panel, corridor length. The
  predicted τ ranges over 0.20–0.85 with a median of 0.45.
- Single-variable groupings (panel, queue size, hour, probability mass) capture no more than the global τ: the
  signal is a combination, not one axis.

## 2. Cell-level thresholds (`t2_ongoing_cell_tau.py`)

Motivated by note 52's anatomy: the missed cells are queue GROWTH, which is exactly "not queued at T, close to the
queue edge". A threshold keyed on (horizon step, queued-at-T or not, distance to the nearest queue edge) — 60
groups, all known at prediction time — fitted by coordinate ascent on the mean window IoU.

| rule | fit val → priv | fit priv → val |
|---|---|---|
| global τ 0.50 | +0.0093 | +0.0118 |
| τ per horizon step | +0.0089 | +0.0115 |
| τ per (queued at T, distance to edge) | +0.0092 | **+0.0127** |
| τ per (step, queued at T, distance) | +0.0092 | +0.0115 |

- **No cell-level structure beats the global threshold.** The fitted tables do show a consistent qualitative
  pattern — outside the queue the threshold drops to 0.35–0.50 (admit growth), inside it falls with the horizon
  (0.55 at T+5 → 0.35–0.40 at T+30) — but the gain is already contained in the single global shift to 0.50, and
  the group values do not agree between the two fitting directions (e.g. outside, d3–4, T+5: 0.35 vs 0.25).
- **Conclusion: the missed growth is not a thresholding problem.** The probabilities do not rank the cells the
  queue will reach above the cells it will not; lowering the threshold there buys as much noise as signal. Fixing
  growth needs better probabilities (model side), not a better decision rule.

## 3. What is actually bankable

| item | ongoing | total | status |
|---|---|---|---|
| **τ 0.60 → 0.50** | **+0.009…+0.012** | **≈ +0.0015** | robust, out-of-sample in both directions, CI excludes 0 (also note 43) |
| learned per-window τ gate | +0.014…+0.018 | ≈ +0.0023 | needs a production gate fitted on the train run — the train-run ongoing probabilities are **not cached**, so this means LightGBM + CNN×3 inference over the train windows first |
| per-cell τ | 0 | 0 | closed |

## 4. Where the remaining ongoing headroom actually is

The decision side is now bounded: about +0.02 ongoing (≈ +0.003 total) in total, of which +0.012 is free. The
oracle's remaining +0.06 is window-specific noise that no feature set reached here. The anatomy (note 52) says the
model side is where the loss is:

- **queue growth**: 33–40% of the newly reached cells are missed, 42–47% of all errors;
- **dissolution**: 15–17% of the clearing cells are held, 17–18% of all errors;
- both concentrate in the last two horizon steps (43–45% of the loss).

Next experiments should therefore target the probabilities of boundary cells: a growth-weighted loss, features for
the upstream arrival rate and shockwave speed at the queue tail, or a model that predicts the queue END positions
per horizon step instead of per-cell occupancy.
