# 93. The onset STID: a third onset member, and the largest gain since candidate T

Date 2026-09-23. Code `occ_lab/train_onset_stid.py`, `predict_onset_stid.py`, `onset_weight_sweep.py`,
`scripts/t2_onset_blend_predict.py`. Follows note 92: training the onset CNN on T+30 alone made a much better
single model that the blend did not want. The user's suggestion was to bring in a heterogeneous architecture, as
STID was for ongoing (note 85).

## The model

Onset is now, by the organizers' own announcement, "which link breaks down first" at T+30 - a per-link propensity
problem. A link-identity embedding is exactly the bias for it, and no production onset member has one.

    input   per link the 15 x 12 history (occupancy included), flattened and projected to 96, concatenated with
            link-identity (1,450 global ids), time-of-day (288) and day-of-week embeddings, 32 each
    body    3 residual MLP blocks, no mixing across links
    head    ONE map: the T+30 logit
    windows onset windows only (`earliest_onset`), BCE + 0.5 Lovasz, 8k steps, 4 seeds, 296k parameters

## In the blend

Onset family IoU on the official-like windows, production = 0.7 H + 0.3 adapted CNN (0.7862 / 0.8227):

| STID weight | best H / ad9b | validation | private |
|---|---|---|---|
| 0.00 (re-weighting only) | 0.65 / 0.35 | +0.0044 | +0.0034 |
| 0.05 | 0.55 / 0.40 | +0.0205 | +0.0199 |
| 0.15 | 0.45 / 0.40 | +0.0233 | +0.0222 |
| **0.30** | **0.35 / 0.35** | **+0.0230** | **+0.0337** |
| 0.45 | 0.15 / 0.40 | +0.0356 | +0.0260 |
| 0.60 | 0.05 / 0.35 | +0.0237 | +0.0287 |

86 of the 195 mixtures beat production on both months. **The gain is the STID, not the re-weighting**: without
it the best re-weighting is +0.0044 / +0.0034, and one twentieth of STID already buys +0.020. The plateau from
0.10 to 0.60 is broad, so the chosen point is not a knife edge. Alone the STID is 0.7854 / 0.8176 - about the
production blend's own level, and unlike note 92's onset30 (better alone, useless in the mixture) it is
complementary.

Chosen: **0.35 H + 0.35 ad9b + 0.30 STID**, the middle of the plateau, decided as before with the expected-IoU
ratio rule (c = 1.0) at T+30.

## Candidate AN, LB = 0.88153 (submission 56478202): +0.00251 over AI, a new best

AN = AI with the onset rows rebuilt (`scripts/t2_onset_blend_predict.py` -> `scripts/t2_patch_ongoing.py`);
the ongoing half, and everything else, is AI's. **22 queue rows differ from AI.** Rebuilding production's own
weights (0.7 / 0.3 / 0) with the same script reproduced AI's onset rows exactly - 0 rows changed - so the
comparison is clean.

Local estimate was +0.023 / +0.034 onset family, i.e. +0.0035 .. +0.0050 total; the LB gave +0.00251, a
realisation factor of about 0.6, inside note 58's band. This is the largest LB gain since candidate T and the
first time the onset half has moved since idea 9b.

## The same search on ongoing finds nothing (`occ_lab/ongoing_weight_sweep.py`)

Note 85 only ever added the ongoing STID to the frozen blend (0.05-0.30, taken proportionally) or swapped it into
a slot. The onset win came from a region that kind of search cannot reach - the strongest member cut from 0.70 to
0.35 - so the full simplex over gs1 / tfm8 / cnn3 / stid was run at 0.05 (1,771 mixtures, tau 0.50, both months).

**0 of 1,771 beat production on both months.** The production weights 0.20 / 0.60 / 0.20 sit at the optimum, and
the best mixture at each STID weight degrades monotonically with it:

| stid weight | best mixture | validation | private |
|---|---|---|---|
| 0.05 | 0.20 / 0.60 / 0.15 | -0.0004 | -0.0004 |
| 0.25 | 0.15 / 0.50 / 0.10 | -0.0068 | -0.0098 |
| 0.50 | 0.00 / 0.50 / 0.00 | -0.0464 | -0.0484 |

So the weight search was not the blind spot on ongoing: **the identity member adds on onset and does not add on
ongoing.** A plausible reading is that ongoing already has the queue in its history, so which links are prone to
break carries little beyond what the observed queue already says, whereas onset asks exactly that question.

## Seeds 4 -> 8 do not help (`occ_lab/onset_seed_check.py`, `run_onset_stid2.sh`)

Four more STID seeds were trained and the sweep rerun on the 8-seed average.

| mixture | 4 seeds (AN) val / priv | 8 seeds val / priv |
|---|---|---|
| H 0.35 / ad9b 0.35 / **STID 0.30** (AN) | **+0.0230 / +0.0337** | +0.0083 / +0.0199 |
| H 0.30 / ad9b 0.35 / STID 0.35 | +0.0298 / +0.0337 | +0.0117 / +0.0155 |
| H 0.45 / ad9b 0.40 / STID 0.15 | +0.0233 / +0.0222 | +0.0233 / +0.0213 |
| H 0.50 / ad9b 0.40 / STID 0.10 | +0.0219 / +0.0222 | +0.0219 / +0.0213 |

Mixtures beating production on both months: 86 of 195 with four seeds, 55 with eight. At AN's operating point
the 8-seed average is worse on both months; only at small STID weights are they equal.

The likely mechanism is the decision rule, not accuracy: onset is decided by the expected-IoU **top-k set**, so
the SCALE of the probabilities sets k directly. Averaging more seeds flattens the map and moves k. AN keeps its
four seeds. (Re-tuning the ratio c, or a temperature on the STID probabilities, could recover it - but onset
decision-rule changes have lost on the LB three times: U 0.87641, X 0.87609, AB 0.87764.)
