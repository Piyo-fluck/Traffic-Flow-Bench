# 77 - Task 2: training on 4.5x more ongoing windows helps the CNN alone, but not the blend, and costs onset

Date: 2026-09-20. Code: `data_lab/train_more_og.py`, driver `data_lab/run_dataexp.sh` (Ada).

## What was changed

Only the training set. Everything else (network, inputs, labels, losses, 30k steps, seed 20, rec head) is the
production recipe of rec_s20.

| | ongoing origins | cap per panel | ongoing windows | onset windows |
|---|---|---|---|---|
| production (control `ctrl_prod`) | every 30 min | 3,000 | 21,663 | 2,352 |
| `og_step2_nocap` | every 10 min | none | 98,243 | 2,352 |

The control reproduces `verify_00_rep9b` to four decimals, so the comparison is clean.

## Results (official-like windows, family delta vs candidate T, both models adapted by 9b)

| | val onset alone | val onset T-blend | val ongoing alone | val ongoing T-blend | priv ongoing alone | priv ongoing T-blend |
|---|---|---|---|---|---|---|
| control | -0.0118 | +0.0000 | +0.0005 | -0.0014 | +0.0003 | -0.0001 |
| more ongoing | **-0.1745** | **-0.0248** | **+0.0044** | -0.0009 | +0.0016 | +0.0006 |

- Ongoing alone improves, and the window CI is positive on validation (+0.0036 [+0.0003, +0.0068]).
- **Inside the ongoing blend the gain disappears** (-0.0009 / +0.0006): the transformers and the global-state
  LightGBM already carry it.
- **Onset collapses** (-0.17 alone on validation): the onset windows are now 2.4% of the training set instead of
  9.8%, and the same network serves both conditions.

## Reading

The ongoing capacity of the CNN was indeed limited by the data cap, but the blend does not need it, and the shared
network makes the extra ongoing data expensive for onset. A version that keeps the onset share (oversampling) or a
separate ongoing-only model would be the next thing to try if this direction is continued.

## With the onset share restored (`--onset-share 0.098`, onset windows repeated 2-4x per panel)

100,595 training windows, ES BCE 0.0215 (control 0.0246, more-ongoing 0.0223 - the best of the three).

| family delta vs candidate T | val onset alone | val onset T-blend | val ongoing alone | val ongoing T-blend | priv onset T-blend | priv ongoing alone | priv ongoing T-blend |
|---|---|---|---|---|---|---|---|
| control | -0.0118 | +0.0000 | +0.0005 | -0.0014 | -0.0012 | +0.0003 | -0.0001 |
| more ongoing | -0.1745 | -0.0248 | +0.0044 | -0.0009 | +0.0024 | +0.0016 | +0.0006 |
| **+ onset share kept** | -0.0760 | **-0.0014** | **+0.0082** | -0.0004 | -0.0039 | **+0.0048** | -0.0007 |

- Repeating the onset windows brings the onset blend back to the control (-0.0014 validation), so the collapse of
  the previous run was indeed the share.
- **The CNN's ongoing improves the most of the three** (alone +0.0082 validation, +0.0048 private, both window CIs
  positive: +0.0063 [+0.0028, +0.0098] and +0.0070 [+0.0036, +0.0103]).
- **The ongoing blend is still flat** (-0.0004 / -0.0007). The 0.20 CNN slot sits next to 0.60 transformers and
  0.20 global-state LightGBM, which already cover what the extra data adds.
- Onset alone is still below the control on validation (-0.076 vs -0.012) while private onset T-blend is -0.0039.
- So: better model, no gain in the blend as it stands.

## Three seeds in the ongoing slot, and the LB

- Two more seeds (21, 22) of the same recipe: ES BCE 0.0215 / 0.0218 / 0.0215, all better than the control 0.0246.
- Ongoing slot at the production tau 0.50, family delta vs candidate T:

| CNN slot | validation | private |
|---|---|---|
| new x1 | -0.0006 | -0.0008 |
| new x3 | +0.0004 | +0.0011 |
| production cnn3 + new x3, 1:1 (submitted) | +0.0007 | +0.0003 |

  All window CIs cross 0. At tau 0.55 the deltas look bigger, but the base is 0.0037 lower, so 0.50 stays.
- **Candidate AC** = Z with the ongoing CNN slot as the 6-model average
  (`work/sub_AC_ttt3600_ogdata_t4ref/submission.csv`), 29 queue rows differ from Z, all ongoing.
- **LB: AC = 0.87838 (submission 56394206), -0.00033 against Z (0.87871).**
  - The local estimate was about +0.0001 total, so this is within the noise of a single measurement, on the
    negative side.
- Z remains the best.
