# 125 - Pseudo-supervised adaptation of the transformer and STID families; candidate AU

Note 106's recipe (causal supervised adaptation on real Task 2 windows of the scored run built before the day's
first origin, labels from the released layer, weights carried forward) had only ever been applied to the CNN
(p_PS). The transformer family (tfm8, 0.60 of `old`) and the STID family (corrector features only) were frozen.
The transformers' one earlier adaptation used the reconstruction loss alone and hurt (note 75); here the loss is
the supervised one only.

## Single models, official-like ongoing pool (`occ_lab/pseudo_adapt_tfm.py`, `occ_lab/pseudo_adapt_stid.py`)

| family | frozen val / priv | adapted val / priv |
|---|---|---|
| transformer s3 | 0.8068 / 0.7838 | 0.8408 / 0.8147 (12 of 16 panel-splits up) |
| transformer s4 | 0.8038 / 0.7885 | 0.8371 / 0.8205 |
| STID s0..s3 | 0.665-0.676 / 0.653-0.664 | 0.723-0.729 / 0.698-0.704 |

lr 1e-4 (3e-4 worse on the smoke panel), 30 steps per day, bs 16.

## In AT's stack (`occ_lab/tfmps_stack_eval.py`, ongoing family IoU, mean of both months)

| stack | all windows | early days (<= 4 earlier) |
|---|---|---|
| AT | 0.8606 | 0.8919 |
| **tfm8 -> 8 adapted transformers, corrector fed the frozen tfm8 column** | **0.8661 (+0.0055)** | 0.8929 (+0.0010) |
| same, corrector fed the adapted tfm8 | 0.8639 (+0.0033) | 0.8937 (+0.0018) |
| + STID adapted in the corrector columns | 0.8654 (+0.0048) | 0.8933 (+0.0014) |
| reference only: 0.5 p_PS + 0.5 adapted tfm8 (a weight change) | 0.8675 (+0.0069) | 0.8934 (+0.0015) |

Adapted STID gains +0.05 alone but nothing in the stack: the corrector was fitted on the frozen STID's column.

## Candidate AU

`scripts/t2_pseudo_adapt_official.py --arch tfm` (sup only) for tfm s3..s10 on Ada -> `work/sub_t2tfmps_s*`,
mean `work/sub_t2tfmps8`; `work/run_AU_frozen.sh` = `work/run_AT.sh` + `--tfm-dir work/sub_t2tfmps8 --tfm-feat
frozen` -> `work/sub_t2AU_frozen`; patched into AT -> `work/sub_AU/submission.csv`. 120 ongoing cells differ from
AT; Task 1/4 identical. Submitted 2026-09-25 23:22 UTC (the day's last slot).

Also measured the same day and negative: same-day adaptation for p_PS (AT stack +0.0007), same-run analog
forecasts for ongoing (+0.0003), eligibility-aware onset decision (-0.0006).

## LB

**AU 0.88635 against AT 0.88457: +0.00178**, a new best, from 120 ongoing cells. The local call was about +0.0008
(ongoing family +0.0055 x 0.15), so the realisation is about 2.2x - the first ongoing change that realised MORE than
its local estimate. Consistent with note 110: a new KIND of member (an adapted transformer family) moved the LB.
