# 163 The history values at score-INELIGIBLE cells: free information every member discards

Facts: score eligibility is a random ~24 % per (link, 5-min slot) mask (every link 0.76 eligible, ~91 flips per
link-day). 29 % of the ineligible cells still carry speed / flow (pct_observed ~60) in the released layer AND in the
official window_history (speed finite where not eligible 0.291 / 0.294); on the training months those values equal
the truth exactly (|obs - true| median and p90 0.0). Every member builds its history mask as eligible & finite, so
they are dropped: observed share 76 % -> 83 % if used.

Test (`analysis/t2_cnn_pca.py`, T2CNN_OKALL=1: history mask = finite, in training (truth histories) and scoring;
same recipe as the control, Lovasz 0.5, seeds 30 / 31, Ada occ_lab/logs/pca). Lab pool, cnn alone, ongoing tau 0.65:

| | ES bce | val d1-14 | val all | priv d1-14 | priv all |
|---|---|---|---|---|---|
| control s30 | 0.0311 | 0.757 | 0.760 | 0.757 | 0.738 |
| okall s30 | 0.0304 | 0.771 | 0.776 | 0.773 | 0.756 |
| control s31 | 0.0316 | 0.762 | 0.765 | 0.769 | 0.744 |
| okall s31 | 0.0315 | 0.772 | 0.778 | 0.782 | 0.759 |

+0.010..+0.018 ongoing IoU for a single CNN, both seeds, both months; onset alone also up in 6 of 8 cells. (The lab
histories are 52 % observed from Task 1 masking, so the official gain may be smaller.) Applies to every member (the
official histories carry the values: scripts' hist_from_release builds ok = eligible & finite).

## In production: p_PS rebuilt (w192 SSL 30k + 30k supervised + pseudo adaptation sup+rec, seeds 20-23, T2_OKALL=1)

Ada: work/logs_ssl/run_ssl_okall.sh (analysis/t2_cnn_okall.py; env-gated patches in tta_lab/tta/common.py and
occ_lab/pseudo_adapt.py, originals kept as *.pre_okall). Final ES bce okall / prod: 0.0229/0.0227, 0.0211/0.0206,
0.0213/0.0220, 0.0214/0.0220. `occ_lab/okall_stack_eval.py`:

| | all val | all priv | d1-5 val | d1-5 priv |
|---|---|---|---|---|
| single p_PS, 4-seed mean of per-seed IoU: prod | 0.8507 | 0.8284 | 0.8752 | 0.8769 |
| single p_PS okall | 0.8531 | 0.8304 | 0.8772 | 0.8784 |
| AU stack | 0.8743 | 0.8579 | 0.8907 | 0.8950 |
| AU with p_PS -> okall (4 seeds) | +0.0022 | -0.0007 | **+0.0042** | **+0.0049** |

Much smaller than the plain CNN of the first test (+0.010..+0.018): the SSL pretraining and the pseudo adaptation
already recover part of the missing cells. Positive on days 1-5 in both months. The transformer family follows
(work/logs_ssl/run_tfm_okall.sh).

## + the transformer family (tfm8, seeds 3-10, T2_OKALL=1, pseudo adaptation sup 1e-4; run_tfm_okall.sh)

| | all val | all priv | d1-5 val | d1-5 priv |
|---|---|---|---|---|
| single tfm8 prod / okall | 0.8575 / 0.8606 | 0.8350 / 0.8376 | 0.8682 / 0.8776 | 0.8824 / 0.8854 |
| AU, tfm8 -> okall | +0.0012 | +0.0012 | +0.0031 | +0.0013 |
| AU, p_PS -> okall | +0.0022 | -0.0007 | +0.0042 | +0.0049 |
| **AU, p_PS + tfm8 -> okall** | **+0.0024** | **+0.0010** | **+0.0065** | **+0.0043** |

Both members rebuilt: positive in all four conditions; days 1-5 +0.0043..+0.0065 ongoing (~+0.0008 total). Next:
the official build (hist_from_release and t2_pseudo_adapt_official need the same switch).

## + cnn3 (t2_cnn_lov seeds 0-2, okall) and the note 160 cell stacker on top

cnn3 okall alone mixed (all 0.7989/0.7813 -> 0.7967/0.7749; d1-5 0.8540/0.8160 -> 0.8454/0.8416); in the stack with
p_PS + tfm8 okall: +0.0028 / +0.0012 all, +0.0061 / +0.0042 d1-5 (cnn3 adds nothing). The cross-month cell stacker
(`og_cellstack.py --okall`, deltas vs the p_PS + tfm8 okall stack): a0.5 +0.0066 / +0.0064 all, +0.0038 / +0.0022
d1-5; with the away features a0.5 +0.0075 / +0.0057, +0.0060 / +0.0018. Implied vs AU (p_PS + tfm8 okall + stacker
a0.5): ~+0.009 / +0.007 all, ~+0.010 / +0.0065 d1-5.

## Candidate BD (built, not submitted)

Official: `scripts/t2_pseudo_adapt_official_okall.py` (Ada; the three history masks = has a value, labels
eligible-only) for p_PS s20-23 and tfm s3-10 -> work/sub_t2pseudo_ssl4_okall, work/sub_t2tfmps8_okall (means);
`work/build_BD.sh` -> `work/run_BD.sh` = run_AU_frozen.sh with only those two dirs swapped (corrector and its frozen
feature members unchanged) -> patched into AU -> `work/sub_BD/submission.csv`. vs AU: 186 ongoing cells in 51
windows (114 validation, 72 private; 106 added, 80 removed), onset / Task 1 / Task 4 identical.

## Corrector refit and the LB

`occ_lab/refit_okall.py` (note 126's R0 recipe on the okall p0; residual_lgb_R0okall.txt), vs AU:
BD +0.0024 / +0.0010 all, +0.0065 / +0.0043 d1-5; BD + R0ok +0.0046 / +0.0052 all, +0.0057 / +0.0076 d1-5 (worse than BD
on d1-5 validation) -> not adopted (AV precedent). Submitted **BE** = BB (AU + onset set decoder) + BD's ongoing
(205 cells vs AU): public **0.88617** (AU 0.88635; BA 0.88621; the onset part was +0.00062 AX->BA). If the onset part
is additive, the okall ongoing change is about -0.0008 on public - the lab +0.0024 (val all) did not transfer.
