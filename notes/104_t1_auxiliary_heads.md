# 104. Auxiliary heads for the Task 1 CNN

Date 2026-09-24. Code `analysis/t1_m3_cnn.py`, env flag `M3_AUXH` (empty by default, so production is unchanged).
The user's third proposal from the Kaggle survey: keep the network and add heads that are trained and then thrown
away, so the shared trunk has to represent level, temporal derivative and spatial derivative separately.

    heads    d_t q = q[t+1] - q[t-1],   d_x q = q[i+1] - q[i-1],   occupancy
    training from the full truth on the same masked cells, weight lambda each
    inference only the original speed and flow channels are read

The direction is consistent with what note 101 found: cutting the flow term degraded speed as well, so the second
channel is already acting as an auxiliary signal for the first.

## Result

Same seed, epochs, inputs, masks and architecture as A; only the extra heads differ.

| run | auxiliary scaling | lambda | S_state | speed RMSE | flow RMSE |
|---|---|---|---|---|---|
| **A** (production) | - | - | **0.9523** | **1.307** | **29.67** |
| G2 | each target scaled to unit sd | 0.1 | 0.9510 | 1.356 | 29.94 |
| G | d/SCALE_Q and occ x10 | 0.1 | 0.9400 | 1.794 | 32.20 |
| H | the same | 0.3 | 0.9325 | 2.077 | 33.80 |

G and H were a flawed test of my own making: measured on the train truth, the auxiliary targets came out with
standard deviations of 4.9 (d_t q / 100), 5.4 (d_x q / 100) and 1.34 (occ x 10) against about 1.0-1.2 for the
main targets, so squared they dominated the objective at any lambda. Rescaling each to unit sd recovers most of
the loss (0.9400 -> 0.9510).

**With the scaling fixed, the auxiliary heads still lose: 0.9510 against 0.9523, worse on 10 panels out of 10 and
in both months.** Speed degrades most (1.356 against 1.307), which is the same signature as every other change
to the training objective (note 101).

## Where this leaves the Task 1 training design

Everything tried on the training side now sits below the production configuration, and by a similar margin:

| variant | S_state |
|---|---|
| **A: additive targets, LOSS_W (0.75, 0.25), two heads** | **0.9523** |
| F: LOSS_W (0.60, 0.40) | 0.9513 |
| G2: + auxiliary heads, lambda 0.1 | 0.9510 |
| B: relative target parameterisation | 0.9504 |
| D2: LOSS_W (0.873, 0.127), loss scale matched | 0.9505 |
| E: per (panel, regime) metric weights | 0.9502 |

Six perturbations of the objective and the parameterisation, all 0.0010-0.0021 below A, none above. The margins
are single-seed and of the order of seed noise, so the honest reading is not that each is exactly that much
worse, but that **the production recipe sits at a local optimum of the design space we can search**, and that the
remaining error is not reachable by changing what the network is asked to predict.
