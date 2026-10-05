# 112. What S_LWR really scores, how big Task 3 actually is, and the length mismatch nobody has used

Date 2026-09-24. Code `analysis/t3_official_rhs.py`. Sources: `src/task3/score_task3.py`, `docs/BASELINES.md`.

## The official objective, settled

The scorer is

    rhs   = dt * (inflow + on_ramp - outflow - off_ramp)
    E_LWR = sum |dN - rhs| / sum |rhs|,   S_LWR = max(0, 1 - min(1, E_LWR))

with `dN` from the submitted state. The obvious reading - that `rhs` is built from the released neighbouring
flows, so the metric asks for something other than accuracy - is **wrong**, and the release says so:
`load_boundary_flux` states the flows are organizer-held, projected onto the released observations, and that a
topology-derived substitute "put even the exact truth at S_LWR 0.0".

Measured, to be sure of it. Building `rhs` from released mainline flows and ramps, on the transitions the scorer
keeps (10 panels, local test block):

| set | n | \|rhs_obs\| | \|dN_true\| | \|dN_true - rhs_obs\| | \|dN_ours - dN_true\| |
|---|---|---|---|---|---|
| all scored | 4,149,144 | 7.4611 | 0.7189 | 7.7286 | 0.1356 |
| target-touching | 1,249,627 | 7.3272 | 0.7258 | 7.6042 | 0.4379 |

The observable right-hand side is **ten times larger than the true dN**, and a perfect answer would score
E_LWR 1.0365, i.e. S_LWR 0. Exactly as documented. **`rhs` is not computable from the release**, the organizer's
version is close to `dN_true`, and `tfb/physics.py`'s proxy - `|dN_pred - dN_true| / |dN_true|` - is the right
objective after all. The branch is closed by measurement rather than by reading.

## Task 3 is ten times bigger than note 23 said

`docs/BASELINES.md` publishes, scored with the real flux over the ten validation corridors:

| | naive baseline | perfect answer |
|---|---|---|
| `S_physics` | 0.3549 | **0.9636** |

With S_FD near 1.0 that puts a perfect answer at **S_LWR about 0.945**, not 1.0: the organizer's flux is anchored
only where observations were published, so target-touching transitions keep a residual even for the truth. Our
proxy returns exactly 1.0 there by construction, so it is optimistic by that structural term.

Production sits at proxy 0.6486, so the reachable range is roughly 0.6486 -> 0.945, **about +0.030 on the
total**. Note 23 put the Task 3 headroom at +0.004 and called it closed; that estimate was built on note 01's
"noise floor", which note 107 showed is not noise. **Task 3 has the largest unclaimed pool outside Task 2.**

## The lever: S_LWR weights by link length, S_state does not

E_LWR's numerator is `sum |e[t+1] - e[t]| * L`. Link length spans 0.010 to 7.771 km, median about 0.29. The
residual is therefore concentrated where our training is not:

| | share of the S_LWR residual | share of the targets |
|---|---|---|
| **top 10% of links** | **43%** | 10% |
| top 25% of links | 65% | 25% |

per-link residual against link length: **corr +0.806**, and the pattern holds on all ten panels (32-51% for the
top decile).

`S_state` weights every target cell equally; `S_LWR` weights it by `L`. Nothing in the training loss knows this.
Reallocating accuracy from short links to long ones is therefore close to free in `S_state` - the pooled RMSE
only cares about total squared error - while `S_LWR` gains in proportion to `L`. A 10% error reduction on the top
decile, paid for by a 1.1% rise everywhere else, is roughly RMSE-neutral and takes 4.3% off the LWR residual.

This is not note 105's `M3_LWR` (the metric's own numerator as a loss, which destabilised training and lowered
S_LWR to 0.640) and not note 101's `(panel, regime)` reweighting (uniform within each group). It is a per-cell
weight `w = (L / median L)^gamma` on the existing loss, with gamma swept - the one reweighting axis the metric
explicitly asks for and the only one never tried.

## The length weighting was run, and it does nothing (2026-09-24)

`M3_LENW=gamma` weights every target cell by `(L / median L)^gamma`, normalised to mean 1 and clipped at 10x, on
the existing loss (`analysis/t1_m3_cnn.py`, backup `work/t1_m3_cnn.py.bak_lenw`). Single runs, 25 epochs, seed 0,
10 panels, local test block:

| run | S_state | S_FD | S_LWR | 0.35 s + 0.05 f + 0.10 l |
|---|---|---|---|---|
| gamma 0 (control) | 0.95091 | 0.98749 | 0.63641 | **0.44583** |
| gamma 0.25 | 0.95094 | 0.98746 | 0.63633 | **0.44583** |

S_state +0.00003, **S_LWR -0.00008**. The totals agree to five digits. The weighting did reach the optimiser -
epoch 1 loss is 0.1548 against 0.1519 and the two runs follow different trajectories, and every one of the ten
per-panel lines differs - but the solution lands in the same place. The gamma 0.5 and 1.0 runs were stopped after
this, on the user's call.

What this retires is the reasoning in the section above, not the measurement behind it. The residual really is
concentrated on long links (top decile 43% of the residual, 10% of the targets, corr +0.806), and S_state really
does not weight by length. But **there is no accuracy to reallocate**: telling the network to care 2.3x more
about long links and 0.42x less about short ones moves neither score by more than 1e-4. The error is cell-
specific and irreducible wherever you point the loss, which is notes 96-107 again from an eighth direction.

The pool identified above is unchanged and still unclaimed: a perfect answer scores S_LWR about 0.945 against our
0.6486, roughly +0.030 on the total, and note 23's +0.004 headroom remains void. No route into it is known.
