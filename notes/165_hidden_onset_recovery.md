# 165 Idea (1): recover the official onset the blank hides, and use it as the run's onset supervision

Scripts: `occ_lab/hidden_onset_recover.py` (recovery), `occ_lab/onset_film_recov.py` (note 120's FiLM with
recovered labels). An official onset window hides its horizon + 60 min (18 slots); a bottleneck queue keeps its
downstream front at the bottleneck, so the head of the queue visible 65 min after the breakdown points at it.

Recovery of the hidden bottleneck (hit = within +-2 links of a true T+30 queued cell):

| | top-1 |
|---|---|
| training months (1,722 events; OOF H): H arg-max alone | 0.912 |
| post-blank heads, longest run / head x H / learned ranker (events with a visible queue, 88 %) | 0.578 / 0.865 / 0.895 (top-3 0.940) |
| scored months, lab onset events: validation, blend arg-max -> recovery | 0.828 -> **0.864** (+6 net) |
| private | 0.858 -> 0.852 (-1 net) |

Note 120's FiLM (onset STID, history-branch FiLM, adapted per day on earlier days' first-morning windows, past-day
cap 4 = an official window's budget) on the training tail, AP blend onset IoU delta:

| supervision | steps | AP blend | STID alone |
|---|---|---|---|
| true labels | 40 | +0.0060 | +0.0137 |
| recovered, head +-1 | 40 | +0.0053 | -0.150 |
| recovered, head only | 40 | +0.0068 | -0.288 |
| recovered, soft (1 head, 0.5 neighbours) | 40 | +0.0070 | -0.192 |
| **recovered, head only** | **20** | **+0.0085** | -0.239 |

Recovered labels overlap a true T+30 cell in 0.762 of the tail's supervision windows (1.0 cell per label; ranker fitted
on Jun-Dec only). The blend gains at least as much as with the true labels while the STID alone collapses (the
single-cell target sharpens it: the ratio rule alone then picks one cell) - it helps as a ranking member only.
Open: the scored-month lab check (recovered labels from the earlier days' lab onset events) and how it interacts with
the set decoder's ranker, which was fitted on the plain STID column.

### Scored months (occ_lab/film_scored_eval.py, occ_lab/setdec_film_eval.py)

Per lab onset window on day d, supervision = the lab onset events of the 4 most recent earlier days labelled by the
recovered head (width 0; ranker fitted on all training months); FiLM series, 20 steps, carried forward. Recovered labels
overlap a true T+30 cell in 0.496 of 274 (lower than the tail's 0.762).

| onset family IoU | validation | private |
|---|---|---|
| STID alone / FiLM-STID alone | 0.7854 / 0.6382 | 0.8176 / 0.6321 |
| ratio rule, plain blend (= AP) | 0.8160 | 0.8623 |
| ratio rule, FiLM blend | 0.8244 | 0.8682 |

Inside BB's set decoder (stage 1 LambdaRank + swap, stage 2 add; cross-month, 6 seeds):

| STID member of the base / extra ranker features | validation | private |
|---|---|---|
| AP (= BB) | 0.8315 | 0.9020 |
| FiLM | 0.8465 | 0.8908 |
| AP + FiLM candidate features (5 cols) | 0.8350 | 0.9020 |
| 0.5 AP + 0.5 FiLM | 0.8356 | 0.9051 |
| **0.5 AP + 0.5 FiLM + FiLM candidate features** | **0.8402** | **0.9051** |

The last is +0.0087 / +0.0031 over BB's decoder in both months (switches ONSET_ST=filmhalf, ONSET_FILMFEAT=1).

### Official regime and the decoder's noise floor

Official onset windows: one per panel per day on days 1-5 of the month, so FiLM adapts only for days 3-5 (<= 3
rounds, <= 4 labels). `scripts/t2_onset_film_official.py` -> work/sub_t2film: mean |FiLM - plain| on the official
onset windows 0.000-0.005 (lab, carried forward over the month: 0.005-0.036). Lab re-run in that regime (fresh FiLM
per day, R rounds; FILM_FRESH_ROUNDS, FILM_TAG):

| set decoder, stage 2 | validation | private |
|---|---|---|
| BB (AP STID) | 0.8315 | 0.9020 |
| R=3 filmhalf / + FiLM features | 0.8347 / 0.8332 | 0.9091 / 0.8986 |
| R=1 filmhalf / + FiLM features | 0.8336 / 0.8325 | 0.9077 / 0.9020 |
| placebo: filmhalf with FiLM = plain x (1 + 0.02 N(0,1)), 2 draws | 0.8292 / 0.8189 | 0.9009 / 0.9105 |

The placebo moves the refitted decoder by -0.013..+0.009, the same size as every FiLM delta above (and as the
carried-forward +0.0087 / +0.0031). The FiLM effect is not separable from the decoder's refit noise.
