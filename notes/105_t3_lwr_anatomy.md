# 105. Task 3 head-on: where S_LWR's residual is, and why the obvious fixes do not reach it

Date 2026-09-24. Code `analysis/t3_lwr_anatomy.py`, `analysis/t3_density_bias.py`,
`analysis/t3_temporal_smooth.py`. Task 3 had never been attacked directly; note 23 bounded its headroom, and this
adds the structure of the residual.

## The leverage is real

    total = 0.35 S_state + 0.15 S_phys,   S_phys = (S_FD + 2 S_LWR) / 3
          = 0.35 S_state + 0.05 S_FD + 0.10 S_LWR

One point of S_LWR is worth 0.10 of the total, so a trade against S_state pays whenever it buys more than 3.5
times what it costs. Production: S_FD 0.9875 (saturated), **S_LWR 0.6486**.

## Where the residual is

S_LWR = 1 - sum |(e[t+1] - e[t]) L| / sum |dN_true|, with e the density error at target cells and the truth
everywhere else. By regime and by how many of the two cells are targets (10 panels, local test block):

| | R1 | R2 | R3 | mean |
|---|---|---|---|---|
| production | 0.7770 | 0.6732 | **0.4872** | **0.6458** |
| errors made constant within each run of target slots | 0.8173 | 0.7633 | 0.7314 | 0.7706 |

| transition | count | residual | \|dN true\| | ratio |
|---|---|---|---|---|
| no target | 6,326,929 | 0 | 3,467,542 | 0.000 |
| **one target** | 5,986,868 | **1,979,135** | 3,272,873 | 0.605 |
| both targets | 1,919,061 | 846,229 | 1,027,108 | 0.824 |

**Seventy per cent of the residual sits on transitions with a single target cell**, where the other end is the
truth and the residual is simply |e L| - nothing can cancel. The error correlation across a both-target
transition is already +0.290.

## Two obvious fixes, both measured, both empty

- **Density bias.** The submitted density is k = q_pred / v_pred = E[q|X] / E[v|X] while the metric wants
  E[q/v|X], so a Jensen gap could show up as a systematic scale error. Measured: mean density error +0.0046
  against a mean true density of 40.3, i.e. **+0.01%**, and a multiplicative sweep on the submitted flow peaks
  exactly at c = 1.00 (S_LWR 0.6486; 0.98 gives 0.5962, 1.02 gives 0.5930). No bias to correct.
- **Temporal smoothing.** Pulling target cells towards their valid temporal neighbours in the submitted state
  should correlate the errors and anchor them to visible truth. At w = 0.05 the total moves +0.00004 (S_LWR
  +0.0003, S_state unchanged) and every larger weight loses. The one-target transitions, which carry the
  residual, have nothing to cancel against. (A first version of this test mixed in ineligible neighbours as
  zeros and looked catastrophic; that was a bug, not a result.)

## Reading, against note 23

Note 23 bounded the reachable headroom at 0 to +0.036 S_LWR (about +0.004 total) with two floor estimates,
0.639 and 0.684, that bracket the CNN's 0.649. Today's decomposition says the same thing from the other side:
the residual is the absolute density error at single-target transitions, which is exactly the quantity Task 1
cannot reduce further (notes 96-104).

One premise of note 23 needs correcting: it attributes the edge residual to "the truth's own observation noise".
There is no observation noise - `obs == true` to the last bit over 5.95M cells (note 103). The unpredictable
component is the simulator's own high-frequency dynamics, not measurement error. The floor conclusion survives,
since it is unpredictable either way.

## Two follow-ups, both negative

**1. Predicting the density increment on target-visible edges** (`analysis/t3_edge_delta.py`). On those edges the
truth is known on one side at scoring time, so the quantity wanted is `dN = L (k[t+1] - k[t])` - an increment,
not a level near 40 veh/km. A LightGBM with an MAE loss, out of fold by day, given the visible anchor, the CNN's
own values, the previous visible increments, the upstream and downstream densities, time of day, link, regime
and the direction of the edge:

| estimator | sum of the residual over the 5,986,868 one-target edges | vs current |
|---|---|---|
| **CNN (current)** | **1,979,134** | - |
| zero increment (what the smoothing of note 105 implied) | 3,272,872 | +65.4% |
| LightGBM on the increment | 2,094,624 | **+5.8%** |
| 0.25 blend with the CNN | 1,978,450 | -0.03% |

The model's largest gain feature is the CNN's own implied increment (38%), followed by the edge direction (33%).
**A dedicated increment model with the visible anchor cannot beat what the CNN already implies**, which says the
network is already using that anchor close to optimally.

**2. The exact S_LWR residual as a training loss** (`analysis/t1_m3_cnn.py`, `M3_LWR`, default off). Not a proxy
on the density level (`M3_KHEAD`) nor a conservation identity (`M3_DNLOSS`), but the metric's own numerator:
`sum |e[t+1] - e[t]| L` with e zeroed outside the loss cells, reconstructing k from the outputs and passing the
true density and link length as target channels so the inputs stay identical to A.

| | S_state | S_FD | S_LWR | 0.35 S_state + 0.15 S_phys |
|---|---|---|---|---|
| **A** | **0.9523** | 0.9875 | **0.6486** | **0.44755** |
| lambda 0.1 | 0.9486 | 0.9930 | 0.640 | 0.4457 |

**Optimising the metric's numerator made that very metric worse** (S_LWR 0.6486 -> 0.640) while costing S_state;
S_FD rose (0.9875 -> 0.9930) but carries only 0.05 of the weight. The lambda 0.3 run was stopped. This is the
same shape as every objective change in note 101 and note 104: the model degrades more than the targeted term
improves.
