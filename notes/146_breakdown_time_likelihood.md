# 146 Onset idea 4: candidate-specific breakdown-time likelihood

Script: `occ_lab/bottleneck_timing.py` (local; 6 LightGBM seeds, cross-month; on top of note 129 / note 145 prior4).

Per candidate: share of first-onset events (panel's first queue of the day) touching the site +-2 links, time-kernel
weighted around T+30 (sigma 15/30/60 min), weekday-matched, unweighted; site hazard (free at T-5 min -> breakdown
within +-15 min of T+30); gaps to the window's best candidate. Sources: training months (~270 days) and the scored
month's last <= 4 earlier days. Extra: trust = how often the training prior named this month's released first
bottlenecks (last <= 4 days), and a hard gate on it.

Standalone the signal is strong (top-1 of the true bottleneck: training prior unweighted 0.755, month recent sigma 15
min 0.827; production mass ~0.80), but the training prior's reliability is month-dependent: top-1 0.852 in the
private month, 0.663 in the validation month.

| features (swap IoU vs production) | val | priv | mean |
|---|---|---|---|
| note 129 | +0.0142 | +0.0098 | +0.0120 |
| + prior4 (note 145) | +0.0128 | +0.0248 | +0.0188 |
| + train L (4) | -0.0015 | +0.0256 | +0.0120 |
| + month L (4) | +0.0107 | +0.0192 | +0.0150 |
| + prior4 + month L | +0.0113 | +0.0276 | +0.0194 |
| + prior4 + train L + trust | +0.0064 | +0.0329 | +0.0196 |
| + prior4 + train/month L + trust | +0.0069 | +0.0329 | +0.0199 |
| + prior4 + gated train L + month L | +0.0081 | +0.0286 | +0.0184 |

Every training-prior variant loses validation (the classifier fitted on the month where the prior works applies it to
the month where it does not); trust features cannot be learned from two months (little within-month variation).
Best mean +0.0011 over prior4 with validation -0.006: no variant beats prior4 in both months.
