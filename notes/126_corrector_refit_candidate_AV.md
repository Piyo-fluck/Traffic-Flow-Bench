# 126 - Every ongoing family adapted; the residual corrector refitted on the adapted stack; candidate AV

`occ_lab/adapt_family.py` generalises note 106's supervised adaptation to every ongoing family (tfm, cnn3,
occ_cnn, occ_tfm, stid) and three window sets (official-like pool, the 2031-01/02 OOF windows of the train run
along the same causal trajectory, the official windows). Ada queue: `work/logs_fam/`.

Single families, ongoing val / priv, frozen -> adapted: cnn3 0.78 / 0.77 -> 0.83 / 0.81; occ_cnn 0.76-0.78 /
0.75-0.76 -> 0.83 / 0.81; occ_tfm 0.79-0.80 / 0.76-0.78 -> 0.84 / 0.81; stid 0.67 / 0.66 -> 0.73 / 0.70.

The production corrector (`residual_lgb.txt`) was fitted on p0 = old (no p_PS) with frozen feature columns.
`occ_lab/refit_eval.py` refits it on the OOF cache with the exact p0 and columns it is applied to
(`occ_lab/models/residual_lgb_R{0,1,2}.txt`; the saved files needed NAMES as feature names and LF line endings).

| ongoing stack (official-like pool) | all windows | vs AU | early days | vs AU |
|---|---|---|---|---|
| AT | 0.8606 | | 0.8919 | |
| AU (adapted tfm8 in p0, old corrector) | 0.8661 | | 0.8929 | |
| AUc (AU + adapted cnn3 in p0) | 0.8670 | +0.0009 | 0.8916 | -0.0013 |
| R0 refit, frozen columns | 0.8679 | +0.0019 | 0.8961 | +0.0032 |
| **R1 refit, adapted tfm8 / stid / occ_cnn / occ_tfm columns** | **0.8686** | **+0.0025** | **0.8969** | **+0.0040** |
| R2 R1 + adapted cnn3 | 0.8684 | +0.0023 | 0.8945 | +0.0016 |

Candidate AV = R1: `work/run_AV.sh` (`--tfm-dir work/sub_t2tfmps8 --tfm-feat adapted --stid-dir work/sub_t2stid4
--occ-cnn-dir work/sub_t2occ_cnn4 --occ-tfm-dir work/sub_t2occ_tfm4 --corrector occ_lab/models/residual_lgb_R1.txt`),
patched into AU: 186 ongoing cells differ, Task 1/4 identical. Submitted 2026-09-26 ~00:45 UTC.

## LB

**AV 0.88573 against AU 0.88635: -0.00062.** The local +0.0025 (both months, early days +0.0040) did not transfer;
the sign flipped. Same pattern as the corrector variants of note 110 (all lost on the LB): changes to the corrector
layer are not predicted by the official-like pool, while new adapted MEMBERS in the blend (AU) were. AU stays best.
