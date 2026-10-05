# 116. The corridors share a strong driver, and none of it reaches the residual

Date 2026-09-25. Code `analysis/t1_cross_panel.py`.

Every structure test so far has been inside one panel: mask geometry and time neighbours (note 96), the lattice,
the CTM and nearest-neighbour retrieval (note 103), the upstream advection scan over +-160 links (note 107). The
ten corridors are different roads but the **same simulation run**, so a shared stochastic driver the model cannot
see would show up as a residual correlation across panels at the same (day, slot) - and the other corridors'
cells ARE released at test time, so it would be usable.

Per panel and (day, slot) of the local test block: the mean relative density residual over that moment's target
cells, and the mean flow per lane over its eligible cells. Both de-meaned by slot within the panel, so a shared
time-of-day shape cannot manufacture anything. Control: panel A's day d against panel B's day d+1.

| | same moment | day-shifted control |
|---|---|---|
| **true flow level** | mean **+0.5847**, median +0.5801, range +0.43 .. +0.74 | +0.1857 |
| **residual** | mean **+0.0025**, median +0.0005, range -0.013 .. +0.018 | +0.0016 |

Leave-one-out linear fit of one panel's residual anomaly on the other nine: **R^2 = 0.0007** over 16,989
moments.

## Why this negative is strong

The measurement has a **positive control built in**. The same statistic finds +0.585 in the truth, so a shared
driver leaking into the residual would be visible by a wide margin. It is not there: the residual's cross-panel
correlation is indistinguishable from the day-shifted control.

The corridors genuinely move together - there is a common demand signal - and **the model already extracts all of
it from its own panel's visible cells**. Other corridors add nothing.

## Where that leaves the input side

Of the release, the Task 1 model now provably gains nothing from: other panels (here), Task 4 (no time axis,
note 112), occupancy at target cells (blanked, note 112), the official conservation flux (organizer-held, note
112), upstream mainline flow at any lag (note 107), a count lattice or a CTM (note 103).

**One input remains untested: the ramp flows.** `M3_V2` is off in production, and note 05 - the only run that
ever switched it on - states in its own text that "a clean V2-only ablation was not run": it bundled the ramp
channels with the dN loss, used 2 panels, and predates the noise floor. `r_on` and `r_off` appear directly in
the conservation equation that S_LWR scores, and the model has never been given them on their own.

## The ramps, measured two ways (2026-09-25)

**As inputs.** `M3_V2=1`, no dN loss, width 64, 10 panels, seed 0, against the same-seed control and the
nine-seed distribution:

| run | S_state | S_FD | S_LWR | total |
|---|---|---|---|---|
| control, seed 0 | 0.95091 | 0.98749 | 0.63641 | 0.44583 |
| **ramp channels, seed 0** | 0.95061 | 0.98745 | **0.63195** | 0.44528 |

S_LWR **-0.00446 (-1.7 sd)**, total -1.7 sd. The ramp channels make it slightly *worse*. Note 05's +0.004 was
therefore the dN loss, not the inputs - and that +0.004 was itself about 1.5 sd on two panels.

**Directly, with no model in the way.** The residual at target cells on ramp-attached links, against the ramp
flows and their time differences (2,537,436 cells, 10 panels), with a week-shifted residual as the control:

| regressor | corr with e | day-shifted |
|---|---|---|
| r_on | +0.0020 | +0.0032 |
| r_off | +0.0054 | +0.0058 |
| r_net | -0.0018 | -0.0013 |
| d r_on / d r_off / d r_net | +0.0006 / -0.0010 / +0.0030 | +0.0011 / +0.0004 / +0.0019 |

Linear fit on all six: **R^2 = 0.00023** on the level and **0.00029** on the increment. Every correlation is at
the control's own magnitude, so even those are not real.

`r_on` and `r_off` appear in the conservation equation S_LWR scores, and they carry **0.02% of the residual**.
The network was not ignoring ramp information; there is none to ignore, and adding seven signal-free channels to
a capacity-limited model is a pure dilution, which is exactly what the ablation shows.

## The input inventory is complete, and it is empty

| input | result |
|---|---|
| other corridors at the same moment | R^2 0.0007 (positive control +0.585 in the truth) |
| upstream mainline flow, any lag to +-160 links | \|corr\| <= 0.008 (note 107) |
| **ramp flows** | **R^2 0.0002** |
| Task 4 OD and path flows | no time axis (note 112) |
| occupancy at target cells | blanked with speed and flow (note 112) |
| the official conservation flux | organizer-held, never published (note 112) |
| count lattice, CTM, kNN retrieval on the CNN's own features | nothing (note 103) |

Nothing released, and reachable, predicts the Task 1 residual. What remains on the model side is capacity, which
is still paying at width 256 and is worth perhaps +0.002 to +0.004 in total (note 115).
