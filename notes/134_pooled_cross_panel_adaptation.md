# 134 - Cross-panel pooled adaptation: negative - the run shift is corridor-specific

`occ_lab/pseudo_adapt_pooled.py`: one SSL w192 CNN per scored run adapted day by day on the pseudo windows of ALL
eight panels (same-day windows only before the earliest first origin of any panel), against note 106's per-panel
adaptation, 3 seeds. Ongoing / onset IoU of the single network; "d<=4" = days 1-5, where the official windows are
(note 133).

| ongoing (val / priv), mean of s20-s22 | all windows | days 1-5 |
|---|---|---|
| none | 0.810 / 0.791 | 0.869 / 0.866 |
| per panel (shipped recipe) | **0.849 / 0.829** | **0.877 / 0.877** |
| pooled, 60 steps / day | 0.843 / 0.820 | 0.872 / 0.853 |
| pooled, 240 steps / day | 0.849 / 0.826 | 0.875 / 0.854 |

Pooling eight times the data never beats adapting each corridor on its own windows, on the early days included
(s22 private early drops to 0.82). What the adaptation learns is specific to the corridor, not a run-wide factor.

Also measured: on days 1-5 the adaptation is worth only +0.008 / +0.011 (vs +0.039 / +0.038 over the month) - the
official population gains far less from run adaptation than the lab pool suggests, and the early windows are easier
(0.87 without adaptation).
