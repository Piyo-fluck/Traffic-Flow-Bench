# 90. Stability-selected corrector

Date 2026-09-22. Code `occ_lab/stability.py`, log `occ_lab/logs/stability.log`, models `occ_lab/models/stab_*.txt`
on Ada. The user's proposal after note 89: AJ and AK were ahead locally and lost on the LB, which points at
variance in WHICH cells get corrected; so allow AI's correction only where correctors fitted on different time
blocks of the tail reproduce it - stability of the estimator, not confidence of the model.

## Setup

AI's recipe throughout (15 features, the tail's masked-layer windows, L2 on y - p0, 400 trees, eta 0.75,
gate 0.10). The 59 held-out days (2031-01-01 .. 02-28) split into contiguous blocks:

    lobo8   8 blocks, leave one out    -> 8 correctors, 51 of 59 days each
    half6   4 two-week blocks, keep 2  -> 6 correctors, 30 of 59 days each (half-subsamples, as in the original
                                          stability selection)

Filters on the cells AI corrects: `sign >= k` (k block correctors give r_hat AI's sign), `flip >= k` (k of them,
applied alone, flip the cell the same way), `median >= k` (median r_hat of the blocks, kept where k agree in sign).

## Results (pseudo windows; precision = share of changed cells whose new decision is right)

AI corrects 8,546 cells in March and 9,436 in April; **only 0.506 / 0.525 of them end up right.**

| variant | IoU gain Mar / Apr | kept of AI | precision Mar / Apr | rescue | spoil |
|---|---|---|---|---|---|
| AI | +0.0046 / +0.0042 | 1.00 | 0.506 / 0.525 | 0.152 / 0.163 | 0.0022 / 0.0023 |
| lobo8 sign >= 7 | +0.0046 / +0.0042 | 0.99 / 1.00 | 0.506 / 0.525 | 0.151 / 0.162 | 0.0022 / 0.0023 |
| lobo8 flip >= 7 | +0.0046 / +0.0043 | 0.77 / 0.79 | 0.526 / 0.537 | 0.122 / 0.131 | 0.0016 / 0.0018 |
| lobo8 flip >= 8 | +0.0040 / +0.0035 | 0.67 / 0.68 | 0.537 / 0.543 | 0.107 / 0.115 | 0.0014 / 0.0015 |
| **half6 sign >= 6** | **+0.0050 / +0.0046** | 0.91 / 0.92 | 0.512 / 0.533 | 0.139 / 0.151 | 0.0020 / 0.0021 |
| half6 sign >= 5 | +0.0049 / +0.0046 | 0.98 | 0.507 / 0.527 | 0.148 / 0.160 | 0.0022 / 0.0023 |
| half6 flip >= 6 | +0.0031 / +0.0033 | 0.53 / 0.54 | **0.548 / 0.564** | 0.087 / 0.094 | 0.0011 / 0.0012 |
| half6 median >= 4 | +0.0051 / +0.0040 | 0.89 | 0.509 / 0.529 | 0.151 / 0.162 | 0.0022 / 0.0023 |

## Reading

- **Sign agreement over 7/8-day blocks filters nothing** (99-100% kept): the lobo8 correctors are near copies.
  Half-subsamples are needed for the filter to bite.
- **The user's headline metric moves the right way in both months**: requiring the flip to reproduce raises the
  precision of the corrections monotonically (0.506 -> 0.548 in March, 0.525 -> 0.564 in April at half6 flip 6/6).
  But the IoU gain falls with it (+0.0031 / +0.0033): the corrections the blocks disagree on carry more rescue
  than spoil. At lobo8 flip >= 7 the user's target shape appears - 80% of the rescue kept - but spoil falls by a
  quarter, not by half, and IoU is flat.
- The best IoU is half6 sign >= 6: +0.0004 / +0.0004 over AI in both months, 8-9% of AI's corrections dropped,
  precision +0.006 / +0.008. That is the size of the local leads that did not hold on the LB (notes 87-89).
- **AI's corrections are right only about half the time by count** (0.51-0.53), yet the IoU rises. A correction's
  worth to the IoU therefore depends on which way it flips and where, not only on whether it is right - the
  precision of the changed cells is not the quantity the score rewards.

## LB: AL = 0.87880 (submission 56453846), -0.00022 against AI

- AL = AI with the half6 sign >= 6 filter: AI's production corrector (`occ_lab/models/residual_lgb.txt`)
  unchanged, its correction dropped wherever fewer than 6 of the 6 half-subsample correctors
  (`occ_lab/models/stab_half6_*.txt`) share its sign (`scripts/t2_residual_predict.py --stab-models ...
  --stab-min 6`). On the official windows 322 corrected cells became 302; **AL is a strict subset of AI: 20
  queue rows differ, all of them AI corrections withdrawn.**
- Withdrawing those 20 rows cost 0.00022 - about 70% of AI's whole gain over Z (+0.00031 from 322 rows). The
  corrections the time blocks disagree on are, on the LB, among AI's most valuable ones - the same direction as
  the local finding above that the disputed corrections carry more rescue than spoil, only much stronger.
- AI stays the best.
