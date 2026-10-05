# 05 — Negative result: ramp inputs + ΔN-consistency loss do not help the CNN

Date: 2026-09-14 · Script: `analysis/t1_m3_cnn.py` with `M3_V2=1 M3_DNLOSS=0.05` · Log: `work/t1_m3v2dn05_2panels.log`
Tests H5 and H6 from [04](04_t3_physics_proxy.md). Same protocol and panels (D12_I5_N, D7_I210_W) as M3 v1.

## Setup
- **v2 inputs**: seven extra channels. Link length, the valid on/off ramp
  flows per lane and their validity flags, and the on/off attachment flags,
  from `tfb/physics.load_ramps`.
- **ΔN loss**: `Σ|ΔN_err(t+1) − ΔN_err(t)| / Σ|ΔN_true|` over transitions
  between cells with known truth, which is the Task 3 proxy itself, with
  weight λ = 0.05. The training value of this term fell from 1.08 to 0.33.

## Results (test days, family mean of the 2 panels)
| Model | S_state | S_LWR~ | S_phys~ | 0.35·S_state + 0.15·S_phys~ |
|---|---|---|---|---|
| M3 v1 | **0.9525** | 0.645 | 0.759 | **0.4472** |
| M3 v2 + ΔN loss 0.05 | 0.9509 | 0.649 | 0.762 | 0.4471 |

The per-panel pattern is the same. D12_I5_N S_LWR~ went from 0.652 to 0.658,
while its speed RMSE got worse (0.86 → 0.89 on R1).

## Takeaways
- **F18**: With this architecture, S_LWR is not a separate target that can
  be traded for. The ΔN loss moves it by only about +0.005 and costs about
  the same in S_state. The conservation residual is essentially the model's
  estimation error seen through q/v, so what moves S_LWR is more accurate
  cells, not more consistent ones.
- The ramp channels did not buy an obvious flow gain either (D12_I5_N flow
  RMSE 24.3 → 24.7). A clean V2-only ablation was not run.
- Next: capacity (a wider network) and variance reduction (a seed
  ensemble), both of which improve accuracy directly.
