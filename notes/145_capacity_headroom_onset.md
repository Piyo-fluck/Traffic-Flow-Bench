# 145 Onset idea 2: learned discharge-capacity headroom per candidate bottleneck

Scripts: `occ_lab/bottleneck_capacity.py` (features, cross-month eval over LightGBM seeds 0-5),
`occ_lab/bottleneck_fit.py --prior4`, `scripts/t2_onset_blend_predict.py --bn-model occ_lab/models/bottleneck_prior4_lgb.txt --bn-prior4`.

Features per candidate peak (added to note 129's classifier), all released at or before T (slot T excluded):
site capacity q95_free (95th pct of free-flow flow) and q_brk (median highest flow in the 3 slots before an observed
breakdown), number of breakdowns, today's demand (last 3 slots of the window history, max +-2 links) and its 30-min
slope, demand forecast at T+30 = demand x the flow ramp T -> T+30 at this time of day; headrooms demand / capacity
(4), their gap to the window's best candidate (4), capacity step vs both sides (orientation-free).

Swap onset IoU vs production, 6 seeds, lab onset pool (cross-month):

| features | val | priv | mean | rescue / spoil |
|---|---|---|---|---|
| note 129 | +0.0142 | +0.0098 | +0.0120 | 3.7 / 0 |
| + month capacity (all earlier days) | +0.0152 | +0.0221 | +0.0187 | 6.2 / 0 |
| + training-month capacity only | +0.0123 | +0.0209 | +0.0166 | 5.7 / 1.0 |
| + month (<= 4 earlier days) | +0.0088 | +0.0250 | +0.0169 | 6.8 / 0.8 |
| **+ training prior + month <= 4 days** | +0.0128 | +0.0248 | **+0.0188** | 6.7 / 1.0 |

Ablation (month capacity): q_brk headroom alone +0.0180, q95 headroom +0.0140, capacity step alone +0.0144 - every
group helps. Headroom alone ranks the true bottleneck poorly (top-1 0.34-0.54); it works as a correction to the
production mass.

The lab onset pool skips days holding an official onset window, so it has ~no days 1-5 windows; official windows see at
most 4 earlier days. Site capacity is stable across months: training months vs scored month corr 0.82 (q95, q_brk),
so the training months (~270 days, same sensors) supply it; `prior4` = the training-month capacity + the month's last
<= 4 earlier days = what an official window sees. Val -0.0014 vs note 129, priv +0.0150.

Official: vs AW's onset, 1 validation window (4 cells) and 3 private windows change. `work/sub_AXZ/submission.csv` =
AX + this onset (16 queue cells differ from AX). Not submitted.
