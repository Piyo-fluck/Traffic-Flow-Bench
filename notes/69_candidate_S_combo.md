# 69 - Building the candidate: idea 9b holds, the Task 1 residual LightGBM does not (and my comparison was not paired)

Date: 2026-09-16 · Scripts: `scripts/t2_onset_t1res_predict.py`, `scripts/t2_adapt_official.py`,
`analysis/t2_onset_combo_eval.py` · The user asked for a candidate carrying all three measured gains.

The three: ongoing as the 8-seed transformer ensemble (3:1:1, tau 0.50), onset with the Task 1 residual features in
the LightGBM (ideas 2/3), onset with the past-day adapted CNN (idea 9b).

## The combination table (blank-aware official-like onset windows)

| onset | validation IoU / family | private IoU / family |
|---|---|---|
| M production: H + CNN x3 | 0.8101 / 0.7797 | 0.8327 / 0.8181 |
| **H + adapted CNN** | **0.8181 / 0.7862 (+0.0079)** | **0.8516 / 0.8227 (+0.0189)** |
| H+t1res + CNN x3 | 0.8083 / 0.7779 (**-0.0018**) | 0.8247 / 0.8173 (**-0.0079**) |
| H+t1res + adapted CNN | 0.8181 / 0.7859 (+0.0079) | 0.8418 / 0.8385 (+0.0092) |

## The flaw in my own comparison

Note 64 measured the Task 1 residual features **paired**: a base model and a variant trained identically, three
seeds each. Here the production baseline is the **cached** H probabilities, trained earlier with its own bagging
and seeds, while `H+t1res` is a model I trained today. The difference therefore contains training-run variation as
well as the features, so the negative numbers in the third row do **not** establish that the features hurt - and
the positive numbers of note 64 do not carry over either. **The feature block is unverified in the production
configuration**, which is the only configuration that matters for a submission.

The two measures also disagree on private for the combination (window-mean IoU 0.8418, below the adapted-only
0.8516, but family mean 0.8385, above its 0.8227). With 162 windows spread over 8 panels the family mean is driven
by a few small panels; when two summaries of the same experiment point in opposite directions, neither is evidence.

## Decision

Ship only what is clean:

| change | evidence | in the candidate |
|---|---|---|
| ongoing 8-seed transformer ensemble, tau 0.50 | family +0.0030 / +0.0017 vs R, both months | **yes** |
| onset past-day adapted CNN (idea 9b) | +0.0079 / +0.0189 IoU and +0.0065 / +0.0046 family, private CI excludes 0, attributed to the adaptation | **yes** |
| onset Task 1 residual features | paired evidence in note 64, unpaired and ambiguous here | **no** |

`work/sub_t2S` (all three) was built and verified first - 196 differing rows against R, all queue, 25 ongoing and
6-7 onset windows per split - and is kept for reference. The shipping candidate is the two-change build.

## Candidate T, built and verified (`work/sub_T_ttt_t2adapt_t4ref/submission.csv`)

    onset     0.7 * LightGBM H (unchanged) + 0.3 * the past-day adapted CNN, expected-IoU ratio rule
    ongoing   0.20 * gs1 + 0.60 * transformer x8 (s3-s10) + 0.20 * CNN x3, tau 0.50
    Tasks 1, 3, 4   unchanged

Build:

```bash
.venv/Scripts/python.exe scripts/t2_adapt_official.py --out-dir work/sub_t2adaptcnn      # idea 9b, per-day causal
.venv/Scripts/python.exe scripts/t2_cnn_blend_predict.py   --cnn <cnn x3> --og-nets <cnn x3> --og-arch cnn   --og-nets2 <tfm s3..s10> --og-arch2 tfm --w-og2 0.60   --onset-probs-dir work/sub_t2robust_bag5 --onset-cnn-probs-dir work/sub_t2adaptcnn   --ongoing-probs-dir work/sub_t2gs --w-on 0.7 --w-og 0.20 --tau 0.50 --out-dir work/sub_t2S2
.venv/Scripts/python.exe scripts/swap_task_rows.py --base work/sub_R_ttt_t2ens_t4ref/submission.csv   --task queue --files work/sub_t2S2/queue_*.csv --out work/sub_T_ttt_t2adapt_t4ref/submission.csv
```

Verification: 87,000 rows per split with the keys aligned to R; 109 (val) and 83 (priv) queue predictions differ,
over 25 ongoing and 6 / 4 onset windows; at the upload level **192 differing rows out of 6,980,503, all
`task == queue`**, with state and odme bit-identical.

Expected: family-mean onset +0.0065 / +0.0046 and ongoing +0.0030 / +0.0017 over R, i.e. **+0.0012…+0.0015 total
locally**, about +0.0007 after the note-58 halving - below what the public LB can resolve, with the larger share of
the effect sitting on the private month.

`scripts/t2_cnn_blend_predict.py` gained `--onset-cnn-probs-dir` for this: the adapted model has different weights
on every day, so its onset probabilities cannot be recomputed from a single checkpoint.

## The paired re-test, and why the Task 1 residual block still cannot ship

Re-run with a properly paired baseline (`analysis/t2_onset_combo_eval.py`: identical bags, identical prior-dropout
draws, the only difference being the four extra columns), with the cached production H kept as a reference row:

| onset | val IoU / family | priv IoU / family |
|---|---|---|
| H **cached** (production) + CNN x3 | 0.8101 / 0.7797 | 0.8327 / 0.8181 |
| H **retrained** + CNN x3 | 0.8052 / 0.7754 | 0.8237 / 0.8104 |
| H retrained **+ t1res** + CNN x3 | 0.8083 / 0.7779 | 0.8247 / 0.8173 |
| H retrained + adapted CNN | 0.8136 / 0.7815 | 0.8485 / 0.8323 |
| H retrained + t1res + adapted CNN | 0.8181 / 0.7859 | 0.8418 / 0.8385 |

**Paired, the features help**: family +0.0025, +0.0044 (validation) and +0.0069, +0.0062 (private), positive in all
four conditions.

**But the retrained H is itself 0.0043-0.0077 family below the production H**, and the production hyperparameters
match mine exactly (`--rounds-on 600`, 63 leaves, 5 bags, prior dropout). The gap is the draw: note 22 measured
that the onset score moves by up to 0.08 with the prior-dropout draw alone, and five bags do not average that out.

So the feature gain (+0.005 family, about +0.0008 total) is **smaller than the noise of retraining the model that
carries it**. Shipping it means swapping the production draw for a new one and hoping the new draw is not worse.

**Rule this establishes: an onset LightGBM change worth less than about 0.01 family cannot be shipped without
averaging over many prior-dropout draws** - the retraining noise eats it. Idea 9b escapes this entirely because it
replaces the CNN half and leaves the cached LightGBM probabilities untouched.
