# 02 — Task 1: spatial neighbours help as anomalies, with no lag, and weakly on sparse corridors

Date: 2026-09-14 · Script: `analysis/t1_eda02.py` · Logs: `work/t1_eda02_<panel>.log`
Follows from [01](01_t1_structure_noise_time_interp.md), hypothesis H1.

## Findings

### F6. Station levels differ, so raw spatial interpolation fails
- Night free-flow speed differs by station and follows `links.csv` free_speed
  (corr 0.96 on D7_I210_W, 0.91 on D12_I5_N). The flat 105 km/h in
  `lwr_mainline_topology.csv` is **not** the data's free speed.
- Mean flow/lane is identical between most neighbours (median adjacent
  |diff| 2 veh/h/lane). A few stations step by up to **967** (D7_I210_W) or
  664 (D12_I5_N) veh/h/lane, where ramps or lane drops sit.
- Raw adjacent speed differences in free flow: RMS 6.2 (D7_I210_W) and 3.2
  (D12_I5_N) km/h. That is much larger than the noise, which is why raw space
  interpolation scored only 0.56 / 0.79.

### F7. The anomaly (value − weekday×tod profile) is spatially coherent, depending on density
| corr of anomaly | D12_I5_N (258 short links) | D7_I210_W (81 links) |
|---|---|---|
| speed, s vs s+1 | 0.926 | 0.674 |
| speed, k=2 / 3 / 5 / 10 | 0.88 / 0.83 / 0.76 / 0.64 | 0.56 / 0.43 / 0.16 / 0.05 |
| flow, s vs s+1 | 0.930 | 0.804 |
| speed, t vs t+1 | 0.968 | 0.883 |

- Time neighbours are always more informative than space neighbours.
  The gap is small on dense corridors and large on sparse ones.

### F8. No detectable propagation lag between adjacent stations at 5-min resolution
- The cross-correlation of the anomaly between s and s+1, in congested
  periods, peaks at lag 0 and is symmetric for both speed and flow on both
  panels.
- Adjacent stations are too close for a wave to take more than one 5-min step,
  so shifted (lagged) spatial neighbours add nothing for adjacent stations.

### F9. Blending time and space anomalies beats time alone
S_state on train targets:

| Probe | D12_I5_N | D7_I210_W |
|---|---|---|
| base + time anomaly | 0.9083 | 0.9017 |
| base + space anomaly | 0.8734 | 0.8530 |
| base + 0.8·time + 0.2·space | **0.9150** | **0.9071** |
| base + 0.5·time + 0.5·space | 0.9104 (best flow: q RMSE 30.5) | 0.8982 |

- The best weight differs by channel: flow wants about 0.5 space, speed about
  0.2. It also depends on corridor density and on the gap length.
- Score falls with the time gap between bracketing observations. With the time
  anomaly on D12_I5_N: gap 1–2 → 0.920, 3–4 → 0.909, 5–8 → 0.892, ≥9 → 0.875.

## Implications
- Fixed blend weights are the wrong tool. The optimal weight of each
  neighbour depends on its distance in time and space, on which neighbours
  are visible, on the channel, and on the corridor. That is exactly what
  space–time kriging, or a learned model, estimates.
- Interpolate anomalies, not raw values, especially for flow.
- Use `links.csv` free_speed as the station free speed.
