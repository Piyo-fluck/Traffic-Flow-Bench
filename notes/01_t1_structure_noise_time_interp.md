# 01 — Task 1: mask structure, noise floor, and time interpolation gets 0.90+

Date: 2026-09-14 · Script: `analysis/t1_eda01.py` · Logs: `work/t1_eda01_<panel>.log`
Panels: D12_I5_N (258 stations, 93 sensored links) and D7_I210_W (81 stations, all sensored), train split.

Our array scorer (`tfb/metrics.py`) reproduces the official Task 1 score exactly:
baseline on D12_I5_N gives 0.6255, the same as `score_task1.py`.

## Findings

### F1. The mask structure is simple and the same on every panel
- Each link has exactly one station. We order stations by the topology's
  `order_index`, and milepost increases monotonically along that order.
- The cell quality stats are identical on both panels (uniform by design):
  valued 83.5%, eligible 76.4%, **valued but ineligible 7.1%**, missing 16.5%.
- Targets are exactly 20/30/50% of the eligible cells on R1/R2/R3 days.
  Targets are never ineligible cells.
- In the masked layer, every visible value equals the truth. Ineligible cells
  keep their values: they are released and usable as input.

### F2. The baseline loses on congestion and on the daytime
| | D12_I5_N | D7_I210_W |
|---|---|---|
| Baseline S_state (S_v / S_q) | 0.6255 (0.461 / 0.818) | 0.7826 (0.700 / 0.880) |
| Share of targets under 60 km/h | 5.0% | 6.5% |
| Share of the speed squared error from those cells | **60%** | **45%** |
| Speed RMSE, 15-18h vs 00-03h | 25.0 vs 0.9 | 11.0 vs 1.4 |
| Speed squared-error share of the worst 10% of days | 30% | 35% |

### F3. The neighbourhood of a target is rich in time
- P(same station visible at t±5min): 0.56 on each side. Both sides visible:
  0.33 overall; R1 0.50, R3 0.22.
- The gap between the bracketing visible cells in time has median 3 steps,
  p90 6, p99 13. Only 1.3% of targets are one-sided (at the day edge).
- The adjacent stations at the same t are visible about as often as the
  time neighbours, but they are much less informative (F5).

### F4. Noise floor, and with it the Task 1 ceiling
- Estimated from the second difference in time of eligible truth, robust σ:
  - D12_I5_N: speed **0.87 km/h** (free) / 0.27 (congested); flow **17.9 veh/h/lane**
  - D7_I210_W: speed 1.28 / 0.53; flow 20.5
- `pct_observed` does not change the noise. The σ of ineligible cells
  (pct 1–74) matches that of pct 100 cells, so ineligible values are as good
  as eligible ones as input.
- The truth is the noisy observation, so the RMSE floor is about σ, and the
  practical Task 1 ceiling is about **0.96** (D12_I5_N) or 0.95 (D7_I210_W).

### F5. Probes: time interpolation alone takes the score from 0.63 to 0.91
S_state on train targets (S_v / S_q):

| Probe | D12_I5_N | D7_I210_W |
|---|---|---|
| Baseline | 0.6255 | 0.7826 |
| **Linear interpolation in time, any visible cell** | **0.9124** (0.894/0.934) | 0.8983 (0.871/0.931) |
| Same, eligible cells only | 0.9078 | 0.8943 |
| Baseline + time-interpolated anomaly | 0.9083 (0.882/0.939) | **0.9017** (0.872/0.936) |
| Linear interpolation in space (raw values) | 0.7863 | 0.5624 |
| Mean of time and space | 0.8785 | 0.7697 |

- Using the ineligible values adds about +0.004.
- For flow, the anomaly form (base + interpolated deviation) is better on both
  panels. For speed, raw interpolation is better on D12_I5_N.
- R3 is the hardest regime everywhere, because the gaps are longer.

## Hypotheses this opens
- **H1**: The spatial signal is not useless. Raw levels differ between
  stations (ramps, per-link free speed), so it should be interpolated as an
  anomaly and weighted by lag. We are testing this in `t1_eda02`.
- **H2**: A smoother that uses more than two points in time (Kalman/RTS,
  local regression) and models the noise will close part of the gap between
  RMSE ≈ 2.5–3 km/h and the floor of about 0.9–1.3 km/h.
- **H3**: Most of the remaining error sits in congestion onset and recovery,
  where time interpolation smears the shock. Physics (the wave speed
  upstream) should help there.
- **H4**: Speed and flow may be better modelled jointly (through occupancy /
  the fundamental diagram) than as two independent channels.
