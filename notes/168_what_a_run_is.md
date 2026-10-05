# 168 What differs between runs: the train months are ONE run; runs differ in per-ramp demand

Date 2026-09-27. Code `analysis/run_diversity_eda.py`, `analysis/run_factor_eda.py`. Purpose: define what a
generator of new runs must vary (the user's direction: generate runs, not synthesise look-alike data).
The scored months' released layer is used only to derive the method (no model is fitted on it).

## The nine train months are the same run

Per-link queued share (v <= 0.6 vf, eligible observed cells, 06-20 h), correlation between months: 0.93-1.00
among the train months on every panel (0.99-1.00 on 6 of 8). Against the scored runs:

| panel | train vs val | train vs pri | val vs pri |
|---|---|---|---|
| D7_I10_E | 0.86-0.89 | 0.94-0.96 | 0.73 |
| D7_I10_W | 0.93-0.98 | 0.95-0.99 | 0.95 |
| D7_I210_E | 1.00 | 1.00 | 1.00 |
| D7_I210_W | 0.97-0.98 | 0.90-0.93 | 0.98 |
| D7_I405_N | 0.98-1.00 | 0.97-0.99 | 0.99 |
| D7_I405_S | 0.95-0.97 | 0.89-0.94 | 0.92 |
| D12_I5_N | 0.81-0.85 | 0.63-0.71 | 0.91 |
| D12_I5_S | 0.61-0.65 | 0.84-0.86 | 0.54 |

## The run factor is ramp demand, not mainline capacity or demand

Scored run / train ratio per link (median, q10-q90 over links / ramps), 06-20 h:

- mainline free-flow flow (demand proxy): 0.96-1.02, q10-q90 within about +-5 %
- mainline p98 flow (capacity proxy): 0.97-1.01 (D12_I5_S private q10 0.80, D12_I5_N q10 0.89)
- **on-ramp mean flow: medians 0.29-0.99, q10 0.06-0.63, q90 0.48-3.29**
- **off-ramp mean flow: medians 0.41-1.06, q10 0.09-0.57, q90 0.72-2.31**

Each run draws its own per-ramp demand (factors of 0.1 to 3 per ramp); the mainline links barely change. With a
single training run, every model sees one fixed ramp-demand pattern, which is the mechanism behind note 108 (H does
not travel between runs, -0.13) and note 41 (ramp features learn run-specific associations and hurt onset).
A generator of new runs therefore has to redraw per-ramp demand (plus the incident schedule: 5 / 7 incidents per
scored month per the release docs) and let the physics move the bottlenecks.

## Correction (same day): the released ramps are not the generator's ramps

Checked right after: along the corridor, mainline flow changes between adjacent links do not follow the released
ramp flows (corr of q[s_down] - q[s] with the attached on-ramp flow within +-0.1 on every panel, either direction).
The network manifest lists the ramp attachment map as "replaced_with_synthetic". The per-ramp ratios above are
therefore properties of the synthetic ramp layer, not evidence that runs differ in the demand that moves
bottlenecks. The first section (one train run; scored runs differ) stands. The generator's real entries and exits
are the OD zones of path_set.csv (note 169).
