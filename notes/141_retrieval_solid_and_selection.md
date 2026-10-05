# 141 - Retrieval done right: SOLID v3 + per-window member selection lift AU's ongoing on every window set

Two ingredients that each moved AU by ~0.0005-0.001 (notes 138, 140) rebuilt so that they reach the early-month
windows the official set holds:
1. the neighbour bank includes the SAME day's windows whose horizon ends before the origin (labels released by T),
   not only earlier days - the first days of a month have almost no earlier days;
2. similarity on the target's queue neighbourhood (links queued in its last 15 min +- 8), and for SOLID on the adapted
   backbone's own collapsed features, minus a time-of-day distance; SOLID's head update is similarity-weighted.

- SOLID v2 (`adapt_variants.py --method solid2`): pool up to the day's first origin; v3 (`--method solid3`): each
  window's own pool = earlier days + same-day windows released by its origin. Applied to all four p_PS seeds.
- selection v2 (`occ_lab/select_v2.py`): seven members (gs1, adapted tfm8, cnn3, p_PS, occ_cnn, occ_tfm, stid),
  per-horizon Brier on the k nearest released windows, w ~ prior exp(-beta (B / B_min - 1)), shrunk to AU's weights
  by n / (n + kappa); then the production corrector.

| ongoing vs AU (val / priv) | all windows | days 1-5 |
|---|---|---|
| p_PS -> SOLID v2 | -0.0002 / +0.0011 | +0.0017 / +0.0006 |
| p_PS -> SOLID v3 | +0.0001 / +0.0008 | +0.0013 / -0.0003 |
| selection (earlier days only), k40 b10 | +0.0017 / +0.0022 | +0.0003 / -0.0005 |
| selection (+ same day), shipped p_PS, 6 configs | +0.0015..+0.0021 / +0.0021..+0.0024 | -0.0001..+0.0014 / +0.0004..+0.0027 |
| **SOLID v3 + selection, 6 configs** | **+0.0020..+0.0025 / +0.0023..+0.0031** | **+0.0008..+0.0031 / +0.0009..+0.0013** |
| SOLID v3 + selection k60 b5 kappa30 | +0.0024 / +0.0031 | +0.0027 / +0.0013 |

Every configuration is positive on both months and both window sets. Production path: the neighbour bank is the
official-like lab pool of the same run (causal member predictions cached, labels from the released layer), so the
official windows use exactly what the lab evaluates.

## Candidate AX

`work/run_AX.sh`: AU (`run_AU_frozen.sh`) with p_PS -> official SOLID v3 (`scripts/t2_pseudo_adapt_official.py --solid3`,
four seeds averaged into `work/sub_t2pseudo_solid3_4`) and `--select` (bank = the lab pool with SOLID v3 p_PS,
k 60, beta 5, kappa 30). The selection check (AU weights reproduce p0) passes on every panel. Patched into AU
(`t2_patch_ongoing.py`, `swap_task_rows.py`): `work/sub_AX/submission.csv`, 88 ongoing cells differ from AU
(43 validation, 45 private).
