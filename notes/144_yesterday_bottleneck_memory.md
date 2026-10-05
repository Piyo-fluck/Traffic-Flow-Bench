# 144 Onset idea 1: yesterday's inferred first bottleneck as a candidate feature

Script: `occ_lab/bottleneck_memory.py` (local, a few minutes). Seed repeat: LightGBM seeds 0-5.

Setting as the official data allows: day d-1's first-onset window is blanked, so its bottleneck is inferred as
argmax(production onset prob of that window x queue visible 65 min after breakdown, dilated +-2 links); the breakdown
time is known (origin + 30 min). Per candidate peak: distance to it (capped 30), near (<= 2), |dt| between the two
breakdowns, near x (dt <= 30 min), availability. Added to note 129's candidate classifier, cross-month.
228 windows have an inferred yesterday (116 val / 112 priv with a true bottleneck).

Swap onset IoU vs production, mean over 6 seeds:

| features | validation | private | mean |
|---|---|---|---|
| note 129 | +0.0142 | +0.0105 | +0.0123 |
| + yesterday memory | +0.0118 | +0.0137 | +0.0128 |

Top-1 on windows with a yesterday: not improved (val 0.810-0.836 -> 0.810-0.819, priv 0.857-0.875 -> 0.857-0.875).
Per seed the sign flips (val worse in 3/6, priv better in 5/6); memory adds 1 rescue and sometimes 1 spoil.
Difference +0.0005 = well under the seed spread (+-0.004). Noise level: the production mass and note 129's
features already carry what yesterday's location says.
