# 03 — Task 1 methods, round 1: a learned model nearly reaches the speed noise floor

Date: 2026-09-14 · Protocol: `tfb/t1eval.py`. Fit on Jun–Nov 2030, early-stop or
select on Dec 2030, test on Jan–Feb 2031 (59 days). The profile is built from
fit days only. Panels: D12_I5_N and D7_I210_W. The mean is family-weighted.
Follows [01](01_t1_structure_noise_time_interp.md) and [02](02_t1_spatial_structure.md).

## Results (test days)
| Method | Script | D12_I5_N | D7_I210_W | Mean |
|---|---|---|---|---|
| Baseline (profile from fit days) | `t1_m1_blend.py` | 0.6118 | 0.7655 | 0.6886 |
| Base + time anomaly | `t1_m1_blend.py` | 0.8999 | 0.8947 | 0.8973 |
| Base + 0.8·time + 0.2·space | `t1_m1_blend.py` | 0.9074 | 0.9004 | 0.9039 |
| **M1**: per-(time gap, space gap) linear blend | `t1_m1_blend.py` | 0.9108 | 0.9022 | 0.9065 |
| **M2**: LightGBM, 36 neighbourhood features, pooled | `t1_m2_lgbm.py` | **0.9459** | **0.9327** | **0.9393** |
| **M3**: dilated CNN inpainting (0.89M params, 25 epochs, 2 panels) | `t1_m3_cnn.py` | **0.9603** | **0.9446** | **0.9525** |

M2 detail (test days): speed RMSE 1.39 / 1.47 / 1.73 km/h (R1/R2/R3) on
D12_I5_N, and 1.89 / 1.94 / 2.25 on D7_I210_W. Flow RMSE 26–31 veh/h/lane.
Speed is now within about 0.5–1 km/h of the noise floor (0.87 and 1.28).

### M3 trained on all 10 panels (40 epochs, one pooled model), test days
Log: `work/t1_m3_all.log`. Family-weighted mean **0.9523**.

| Panel | S_state | speed RMSE R1/R2/R3 | flow RMSE |
|---|---|---|---|
| D7_I10_E | 0.9524 | 1.16/1.14/1.24 | 28–29 |
| D7_I10_W | 0.9568 | 1.06/1.08/1.11 | 25–26 |
| D7_I210_E | 0.9463 | 1.40/1.41/1.44 | 30 |
| D7_I210_W | 0.9449 | 1.42/1.54/1.68 | 28–29 |
| D7_I405_N | **0.9391** (worst) | 1.49/1.54/1.58 | 35–37 |
| D7_I405_S | 0.9469 | 1.35/1.38/1.49 | 29–30 |
| D12_I5_N | 0.9617 | 0.83/0.92/1.01 | 24 |
| D12_I5_S | 0.9584 | 1.01/0.92/1.18 | 25 |
| D12_I405_N | **0.9672** (best) | 0.71/0.79/0.82 | 21 |
| D12_I405_S | 0.9487 | 0.96/1.00/1.31 | 35–37 |

- Pooled training transfers: the two panels from the 2-panel run both got
  slightly better (D12_I5_N 0.9603 → 0.9617, D7_I210_W 0.9446 → 0.9449).
- The Dec selection score plateaued at 0.9540 from epoch 37 on, so 40
  epochs is enough for this model size.
- The weak spots are flow on D7_I405_N and D12_I405_S (about 36 veh/h/lane),
  and speed on the D7 corridors (1.4–1.7 km/h).

## Findings
- **F10**: Learned blend weights (M1) behave physically but add little
  (+0.003). As the time gap grows, the weight moves from time to space. With
  space gap 2 on D12_I5_N, the speed weights (time, space) are (0.78, 0.22) at
  time gap 2 and (0.48, 0.54) at gap 9. Flow leans on space much more than
  speed does: (0.51, 0.48) at time gap 2.
- **F11**: A non-linear model is worth +0.033 over the best linear blend.
  The top-gain features are the time interpolation and the two bracketing
  values (speed), and time + space interpolation (flow). The gain therefore
  comes from how the neighbours are combined, not from new information:
  congestion is a non-linear state.
- **F12**: M2 stopped at best_iter ≈ 2990 of the 3000-round cap for both
  channels, so it is still under-fitted. The number above is a lower bound for
  M2.
- R3 (50% mask) is still the weakest regime, but the R1–R3 spread shrank
  from about 0.015 (interpolation) to about 0.009 (M2).
- Raising M2's cap to 10k rounds moved best_iter to 3779 (speed) and 3102
  (flow), but the test score stayed at 0.9393. **M2 has saturated.**
- **F13**: The CNN (M3) beats M2 by +0.013. On D12_I5_N its speed RMSE is
  0.86 / 0.98 / 1.02 km/h (R1/R2/R3), which is *at* the estimated noise
  floor (0.87). Flow is 24–25 veh/h/lane against a floor of 17.9, so the
  remaining Task 1 headroom is mostly flow and the sparse corridor
  (D7_I210_W: speed 1.36–1.73). Its selection score on the Dec days (0.9546)
  was still creeping up at epoch 25.
- **F14**: M2 adds almost nothing to M3. The per-channel blend grid
  (`analysis/t1_ensemble.py`, log `work/t1_ens_m2_m3.log`) peaks at M2
  weight 0.1 (speed) / 0.2 (flow) with 0.9527, against 0.9525 for M3 alone.
  A 50/50 blend is worse (0.9497). **Drop M2 from the Task 1 pipeline.**
  Better diversity would come from other CNN seeds or architectures.
- Treat the noise-floor estimate (the MAD of the second difference) as
  approximate rather than a hard bound. An RMSE at the floor means the model
  recovers nearly the clean state.

## Next
- M2: raise the round cap and use more rows (every panel). Watch the
  generalisation to panels the model did not train on.
- Compare M3 against M2 on the same test days, then consider ensembling them.
- Check what the physics score (Task 3) makes of each reconstruction. The
  best RMSE may not give the best LWR conservation.
