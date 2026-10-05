# 07 — Task 1: a wider CNN does not help alone; ensembling helps, mostly through S_LWR

Date: 2026-09-14 · Scripts: `analysis/t1_m3_cnn.py` (`M3_WIDTH`, `M3_SEED`), `analysis/t1_phys_eval.py`
Logs: `work/t1_m3w128_2panels.log`, `work/t1_m3s1_2panels.log`, `work/t1_phys_eval_<tag>_2p.log`
Protocol as before: D12_I5_N + D7_I210_W, fit Jun–Nov, select on Dec, test on Jan–Feb 2031.
The ensembles are plain means of the saved test predictions.
Tests H7 from [04](04_t3_physics_proxy.md), plus the capacity question from [05](05_t1_v2_ramps_dnloss_negative.md).

## Results (test days, family mean of the 2 panels)
| Model | Params | S_state | S_LWR~ | S_phys~ | 0.35·S_state + 0.15·S_phys~ | Δ vs v1 |
|---|---|---|---|---|---|---|
| v1, width 64, seed 0 (`m3`) | 0.89M | 0.9525 | 0.645 | 0.759 | 0.4472 | — |
| v1, width 64, seed 1 (`m3s1`) | 0.89M | 0.9524 | 0.647 | 0.760 | 0.4474 | +0.0002 |
| width 128, seed 0 (`m3w128`) | 3.55M | 0.9523 | 0.650 | 0.762 | 0.4477 | +0.0005 |
| mean of seeds 0+1 (`m3ens2`) | | 0.9535 | 0.657 | 0.767 | 0.4488 | +0.0016 |
| mean of w128 + seed 0 (`m3w128m3`) | | 0.9537 | 0.659 | 0.769 | 0.4491 | +0.0019 |
| **mean of seeds 0+1 + w128 (`m3ens3`)** | | **0.9540** | **0.662** | **0.771** | **0.4495** | **+0.0023** |

## Findings
- **F20**: Width 128 (4× the parameters) does not beat width 64 on the test
  days (S_state 0.9523 vs 0.9525), although its Dec selection score was
  higher (0.9558 vs 0.9546). It trains to a lower loss (0.0199 vs 0.0226)
  and gains on D12_I5_N (0.9616 vs 0.9603) but loses on D7_I210_W
  (0.9431 vs 0.9446). Capacity is not the bottleneck, and the extra
  capacity partly overfits the sparse corridor. It costs about 2.3× the
  time per epoch.
- **F21**: Seed-to-seed variance of a single model is small (S_state
  0.9525 vs 0.9524), but the errors are partly independent. Averaging two
  seeds gives **+0.001 S_state and +0.012 S_LWR~**. That is consistent with
  F17: S_LWR punishes the uncorrelated part of the errors (the time
  differences of the density error), which averaging removes.
- The gains shrink as members are added: 2 members +0.0016, 3 members
  +0.0023 (with w128 adding some architectural diversity). The S_LWR~ gain
  is proxy-based and the proxy is optimistic ([06](06_lb_A_B_and_proxy_calibration.md)),
  but the ranking between models should hold. Expect roughly +0.002 on the
  leaderboard from a 3-model ensemble.

## Recommendation for the next Task 1 production run
- Train 3 models on all panels and every train day: width 64 with seeds 0
  (already done: `work/sub_m3v1`) and 1, plus width 128 with seed 0. Average
  their validation/private predictions.
- GPU cost (RTX 4090, run one at a time): about 25 min for each new width-64
  seed and about 55 min for width 128. If time is tight, use seed 1 only
  (+0.0016 of the +0.0023).
- The expected gain is small (≈ +0.002). Task 2 (≈ +0.21 headroom) should
  take priority for effort. The ensemble can run on the GPU in the
  background while Task 2 work proceeds.
