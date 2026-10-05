// filepath: notes/62_t2_ongoing_band_specialist.md
# 62 — T2 ongoing: the ±4-link band specialist loses to R everywhere → track ① is closed

Date: 2026-09-16 · Script: `analysis/t2_ongoing_band_specialist.py` · Log: `work/t2_ongoing_band_specialist.log`
(6 min)

Note 61 measured a real ceiling: perfect prediction of the true cells within 4 links of the queue visible at T is
worth **+0.0121 total**, and that band holds essentially all of the remaining ongoing loss. This note tests whether
a specialist can reach any of it. Training rows = only the eligible cells of that band (150k per panel, 40–83%
positive), features = the production I set + the one-sided shockwave block (note 54 v1) + the two-sided block
(v2) + the band geometry (distance to the T queue, inside/outside, horizon index), 2 seeds, blended into R's
probability **on the band only**.

| R's weight inside the band | validation (τ 0.45) | private (τ 0.45) | band error rate (val / priv) |
|---|---|---|---|
| **1.0 (R unchanged)** | **0.8501** | **0.8490** | **0.0431 / 0.0455** |
| 0.7 | −0.0013 [−0.003, +0.001] | −0.0034 [−0.005, −0.002] | 0.0429 / 0.0468 |
| 0.5 | −0.0071 | −0.0106 | 0.0451 / 0.0496 |
| 0.3 | −0.0149 | −0.0190 | 0.0483 / 0.0532 |
| 0.0 (specialist alone) | **−0.0246** | **−0.0282** | 0.0523 / 0.0574 |

Monotone: every step away from R is worse, in both months, at τ 0.40/0.45/0.50 alike. **The specialist is rejected.**

**It is not that the boundary information was ignored.** The boundary blocks take **50.8% of the model's gain**,
led by `s2_m1_h` (the projected crossing margin of one side) at 34.4%, ahead of `vr_last` at 25.5%. The specialist
genuinely learned boundary-driven structure on the band it owns — and is still worse there than the ensemble.

## Synthesis: three independent attempts, one result

| attempt | what it did | outcome |
|---|---|---|
| note 54 v1 | shockwave features (measured edge speed + Rankine–Hugoniot) added to the cell model | +0.009 alone, +0.004…+0.006 in M's old blend, **0 inside the R ensemble** |
| note 54 v2 | both sides of every link tracked separately | doubles v1 alone, **still 0 inside the R ensemble** |
| **note 62** | a specialist trained only on the band that holds the loss, blended only there | **strictly negative** |

Hand-built queue-boundary structure has now lost to the transformer ensemble three times, in three different
framings (extra features, better features, separate model on the residual region). The ensemble already extracts
what the 60-minute history contains about boundary motion; the ±0.012 the oracle sees is prediction uncertainty
that this information does not resolve, not modelling slack.

## Where that leaves the 0.9 target

The user's three tracks are now all measured and closed:

| track | verdict | note |
|---|---|---|
| ① ongoing as queue-object / boundary prediction | closed — oracle real (+0.0121), unreachable by every framing tried | 61, 62 |
| ② onset run-specific prior from past days | closed — the rare bottleneck is active, but so is the habitual one, twice as much | 59 |
| ③ onset demand / path-load inference | auxiliary only — ceiling 0.45 top-1 on the missed windows, and the oracle does not beat the estimate | 60 |

The only measured lever left is more transformer seeds: +0.001…+0.002 locally, **≈ +0.0005…+0.001 on the LB** after
the note-58 calibration. From 0.87687 the target needs +0.023. **It is not reachable with the information the
release exposes**; the measured ceilings (perfect onset bottleneck → 0.897 total, perfect ongoing → +0.025) bound
even oracle-level work below it.
