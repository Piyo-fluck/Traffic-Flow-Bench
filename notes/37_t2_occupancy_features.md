# 37 — Task 2: occupancy features (from breakdown-probability models) hurt onset and barely help ongoing

Date: 2026-09-15 · Script: `analysis/t2_occupancy.py` · Log: `work/t2_occupancy.log`

Breakdown-probability models (bivariate Weibull, stochastic capacity) predict breakdown from the speed and
occupancy of the incoming traffic. Occupancy is in the official window history, but no T2 model had used it. The
added features per link are:
- occ_last, occ_max3, occ_mean, occ_trend;
- occ_max3 of the links ±1 and ±2.

The comparison is paired: H's onset configuration and I's ongoing configuration, with and without the occupancy
features. It is scored on the blank-aware official-like windows.

| IoU (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| onset base → +occ | 0.793 → 0.781 (1/4) | 0.765 → 0.763 (3/3) | 0.824 → 0.815 (0/2) | 0.798 → 0.782 (1/5) |
| ongoing base → +occ | 0.773 → 0.777 (6/2) | 0.773 → 0.777 (6/2) | 0.802 → 0.801 (5/3) | 0.758 → 0.763 (6/2) |

- On ongoing the model uses occupancy heavily: 39% of the gain, and `occ_last` 0.343 is the top feature, ahead
  of `vr_last` at 0.218. Yet the IoU moves by only −0.001…+0.005.
- Occupancy is essentially density, which the speed/flow features (`kc_*`) already carry. It replaces existing
  information rather than adding new information.
- Onset gets worse in every condition.
- Conclusion: not adopted (at most about +0.0005 total).
