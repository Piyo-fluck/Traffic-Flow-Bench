# 138 - Other ways to adapt to the scored run (WiSE-FT, surgical, SOLID, TAFAS-style GCM): none beats the shipped recipe

`occ_lab/adapt_variants.py` on p_PS's SSL w192 CNN (s20, s21), each run next to an in-script re-run of note 106's
recipe ("base", which reproduces AU in the stack: -0.0003 / -0.0001). `occ_lab/av_stack_eval.py` replaces p_PS s20/s21.

| ongoing (val / priv) | single, all | single, days 1-5 | stack vs base, all / days 1-5 |
|---|---|---|---|
| base (inp + blocks + rec, 30 steps) | 0.850 / 0.829 | 0.876 / 0.877 | 0 / 0 |
| WiSE-FT, predict with a = 0.25 / 0.5 / 0.75 | 0.825-0.846 / 0.804-0.826 | 0.873-0.875 / 0.870-0.875 | -0.008..-0.002 / -0.002..+0.000 |
| WiSE-FT, interpolated weights carried (a 0.5 / 0.75) | 0.827 / 0.837 (val) | ~0.873 | -0.006 / -0.004 |
| surgical: inp / inp+blocks 0-1 / blocks 6-7+head / GroupNorm | 0.824 / 0.842 / 0.833 / 0.823 (val) | 0.871-0.876 | -0.007 / -0.002 / -0.003 / -0.007 |
| SOLID (per-window head on the 32 closest pseudo windows) | 0.850 / 0.829 | 0.877 / 0.877 | **+0.0005 / +0.0005** |
| GCM only (TAFAS) | 0.810 / 0.790 | 0.870 / 0.866 | -0.012 / -0.002 |
| GCM + full | 0.849 / 0.828 | 0.876 / 0.876 | -0.0001 / -0.0003 |

Every restriction of the adaptation (fewer layers, interpolation back towards the source weights, calibration
modules only) loses in proportion to how much it restricts: the run shift wants the whole backbone moved, and the
shipped recipe is not over-adapting. SOLID's per-window head is the only non-negative variant, at noise level.
