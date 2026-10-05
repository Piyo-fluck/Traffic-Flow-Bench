# 55 — T2 ONGOING: the oracle ceiling of every error class → +0.01 total cannot come from ongoing alone

Date: 2026-09-16 · Script: `analysis/t2_ongoing_ceiling.py` · Log: `work/t2_ongoing_ceiling.log` (12 s)

The user asked to keep working on ongoing until the outlook reaches about **+0.01 total**. This note measures what
each line of work can be worth **at most**, by correcting M's blend to the truth on one class of cells at a time.
Conversion: Δtotal = 0.15 × Δongoing (S_queue is half onset, half ongoing, at weight 0.30).

M's blend, τ 0.50 (the better production setting, notes 52/53): ongoing 0.8325 (val) / 0.8247 (priv).

| ORACLE fix | val Δongoing | priv Δongoing | Δtotal |
|---|---|---|---|
| **grow** (note 54's target) | +0.0519 | +0.0563 | **+0.008** |
| **dissolve** | +0.0446 | +0.0450 | **+0.007** |
| stay | +0.0352 | +0.0339 | +0.005 |
| far | +0.0269 | +0.0297 | +0.004 |
| **every cell within 1 link of the T-queue edge** | +0.0740 | +0.0701 | **+0.011** |
| every cell within 2 links | +0.0929 | +0.0873 | +0.013 |
| every cell within 4 links | +0.1148 | +0.1122 | +0.017 |
| one horizon step, T+5 … T+30 | +0.017 … +0.031 | +0.016 … +0.034 | +0.003 … +0.005 |

The four classes sum to the whole残り loss (+0.159 ≈ 1 − 0.8325), i.e. perfect ongoing prediction is worth
**+0.025 total** and nothing more.

## What this says about the +0.01 target

- **+0.01 total requires +0.067 ongoing IoU — about 40% of every remaining error.** That is more than the entire
  growth bucket solved perfectly (+0.052…+0.056), and it is the level of "every cell within one link of the current
  queue edge predicted exactly right".
- Note 54's shockwave features delivered +0.004…+0.006 in the blend, i.e. **6–9% of the growth oracle**. Reaching
  +0.01 total from ongoing alone would need 12–15× that. I do not think that is attainable: the remaining growth
  errors are cells where the tail advances without any visible precursor in the history, the same information limit
  that notes 17/21/39/44 hit for onset.
- **The most concentrated target is the queue edge**: ±1 link is worth +0.011 total on its own, and it is exactly
  where the shockwave physics applies. Every further feature should be aimed there rather than at whole classes.
- Dissolution (+0.007 total oracle) is nearly as large as growth and is completely untouched so far, so it is the
  best remaining ratio of headroom to effort.

## The realistic route to +0.01 is stacking, not one line

| lever | Δtotal | status |
|---|---|---|
| ongoing τ 0.60 → 0.50 | +0.0015 | measured, notes 52/53 |
| ongoing shockwave v1 | +0.0005…+0.0008 | measured, note 54 |
| ongoing shockwave v2 (two-sided, dissolution) | ? | running |
| candidate O: transformer as the ongoing partner | +0.003 | measured (note 45), never submitted |
| N's global-state ongoing features | +0.0008 | LB-implied (note 48) |
| hill-climbing the blend weights over LGBM / CNN / TFM | +0.001…+0.002 | not tried (the weights were set by hand) |
| onset: train template as a shape prior | +0.0015 | measured, note 51 |
| T1 extra seeds / longer test-time training (T3 follows) | +0.002…+0.003 | note 32, partly unused |

The sum is +0.011…+0.014, but the ongoing items overlap (they all raise the same probabilities), so the honest
expectation is **+0.008…+0.010 if every item lands**. That is the only path to the target, and it needs the
ensemble work the user wanted to keep for last.
