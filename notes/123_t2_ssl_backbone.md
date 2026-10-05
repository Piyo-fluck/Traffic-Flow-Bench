# 123 - Note 119's mechanism on Task 2: an SSL-pretrained backbone makes the pseudo-supervised adaptation pay more

Note 122 left one untried axis: on Task 1, SSL pretraining is what made test-time adaptation pay. Task 2's
adaptation (note 106, p_PS = 0.60 of the ongoing blend) ran on a width-64 backbone that had never been pretrained.

## Recipe

`analysis/t2_cnn.py` (Ada copy; the pre-patch file is kept as `analysis/t2_cnn_pre_ssl_backup.py`) gains
`T2CNN_SSL=n`: before the supervised stage, n steps of masked reconstruction with the existing rec head, on
12-slot history windows drawn from **every fit-day slot of the train truth** (not only labelled windows), 64-link
crops, hidden share per window from U(0.10, 0.70). Train split only; nothing from the scored months. Then the
production supervised stage unchanged (BCE + 0.5 Lovasz + 0.3 rec, 30k steps). Runner:
`work/logs_ssl/run_ssl.sh <gpu> <width> <ssl steps> <seed> <tag>` (train, then `occ_lab/pseudo_adapt.py
--modes none rec sup+rec --save ssl_<tag>`).

## Single model after the production adaptation (sup+rec), mean of validation / private

| backbone | seeds | ongoing | onset |
|---|---|---|---|
| production w64 (no SSL) | s20, s21, s22 | 0.8286, 0.8243, 0.8290 -> **0.8273** | 0.741, 0.691, 0.723 -> 0.718 |
| w64 + SSL 30k | s20 | 0.8370 | 0.731 |
| w192, no SSL | s20 | 0.8310 | 0.729 |
| **w192 + SSL 30k** | s20, s21, s22, s23 | 0.8364, 0.8423, 0.8400, 0.8395 -> **0.8395** | 0.733, 0.737, 0.766, 0.737 -> 0.743 |

Ongoing +0.0122, with the two seed sets not overlapping (production sd 0.0026). Width alone buys little
(+0.0024); SSL alone buys +0.0084 at w64. The onset lab surface cannot rank mixtures (note 108), so its +0.025
is recorded, not relied on.

One reversal against Task 1: on an SSL-pretrained backbone, adaptation by reconstruction ALONE ("rec", idea 9b)
collapses onset (w64: 0.67 -> 0.49; w192: 0.67 -> 0.60), while sup+rec is fine. The ad9b onset member must stay on
the old backbone.

## In the shipped ongoing stack (`occ_lab/ongoing_ablation.py --tag ...`, p_PS replaced, corrector unchanged)

| p_PS | blend + corrector, val / priv | mean | vs AS |
|---|---|---|---|
| production s20 (= AS) | 0.8582 / 0.8430 | 0.8506 | - |
| production recipe, 3-seed mean | 0.8600 / 0.8471 | 0.8536 | +0.0030 |
| **w192 + SSL, 3-seed mean** | 0.8664 / 0.8544 | **0.8604** | **+0.0098** |
| w192 + SSL, 4-seed mean | 0.8676 / 0.8535 | 0.8606 | +0.0100 |

Seed-matched, the backbone is worth **+0.0068** on the stack (+0.0064 / +0.0073, both months); seed averaging the
member adds +0.0030 on top. Against AS: ongoing family +0.0100 -> about +0.0015 total locally, about +0.0012 at
note 106's realisation of 0.8.

## Candidate AT (built, not submitted)

AS with only the ongoing p_PS replaced by the 4-seed mean of the w192 + SSL backbones, each adapted on the
official windows by `scripts/t2_pseudo_adapt_official.py` (T2CNN_WIDTH=192, same recipe as AP; outputs
`work/sub_t2pseudo_ssl_s{20..23}`, mean in `work/sub_t2pseudo_ssl4`). `work/run_AT.sh` = `work/run_AR.sh` with only
`--pseudo-dir` changed -> `work/sub_t2AT`; then `t2_patch_ongoing.py` + `swap_task_rows.py` on AS ->
`work/sub_AT/submission.csv`. Onset, Task 1 and Task 4 are byte-identical to AS; **337 ongoing cells differ**
(validation 201: 66 up / 135 down; private 136: 76 up / 60 down). Local call about +0.0012 total.

## LB

**0.88457 against AS 0.88405: +0.00052**, a new best. The local call was about +0.0012 (ongoing family +0.0100
x 0.15 x 0.8), so the realisation is about 0.43 - inside the 0.44-0.6 band of the earlier ongoing candidates
(note 89), below AP's 0.8. Note: the label "AT" had already been used on 2026-09-25 for the Task 4 L1 submission
(0.84635); this one is the SSL-backbone AT.
