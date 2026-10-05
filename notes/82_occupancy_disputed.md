# 82 - Occupancy carries residual signal on the disputed ongoing cells, but re-deciding them still loses on the LB

Date: 2026-09-20/21. Scripts: `analysis/t2_disputed_rules.py`, `analysis/t2_disputed_occ_signal.py`,
`scripts/t2_stack_occ.py`.

## 1. No decision rule helps (measurements, not search)

On the disputed cells (note 80), keeping the blend everywhere else, ongoing family delta vs the production rule:

| rule on the disputed cells | train (coverage 0.76) | pseudo val | pseudo priv |
|---|---|---|---|
| blend at tau 0.50 (production) | 0 | 0 | 0 |
| tau 0.40 / 0.45 / 0.55 / 0.60 | -0.005 / -0.002 / +0.0003 / -0.002 | all negative | all negative |
| majority of the three members | -0.0005 | -0.0062 | -0.0096 |
| follow tfm8 | -0.0107 | -0.0103 | -0.0105 |
| max / min member | -0.030 / -0.010 | -0.032 / -0.044 | -0.040 / -0.044 |
| fill holes / drop isolated (space) | -0.0041 | -0.0079 | -0.0068 |
| fill across horizons (time) | -0.0000 | -0.0000 | -0.0001 |
| queue-front wave, c = 0.5...2.0 | -0.020...-0.024 | -0.031...-0.038 | -0.034...-0.047 |

**Calibration on the disputed cells** (train coverage): p 0.50 -> 0.44 actual, p 0.60 -> 0.57, p 0.72 -> 0.76.
Near the threshold the probability is honest, so thresholds and monotone recalibration cannot win; the cells are
genuinely near coin flips given the members' information.

## 2. Occupancy does carry new information

Occupancy is the one released mainline channel the Task 2 models never see. Leave-one-panel-out on the train
month's 38,000 disputed cells, LightGBM on logit(p) with and without occupancy features (own link, neighbours,
change over the history, ratio to the weekday x tod profile):

| input | log-loss | AUC | accuracy |
|---|---|---|---|
| blend probability only | 0.5685 | 0.7722 | 0.7077 |
| + occupancy | **0.5437** | **0.7853** | 0.7189 |

## 3. Candidate AF and the LB

`scripts/t2_stack_occ.py`: stacker trained on train-month windows (official coverage) with the occupancy
features, applied to the official windows' disputed cells, **with the number of queued cells held fixed** (the
threshold per split is set so the count matches the blend's) - the fix for candidate AE, which had deleted queue
and lost. Result: 10,629 queued rows against Z's 10,627, 414 cells swapped (233 validation, 181 private).

**LB: AF = 0.87794 (submission 56399322), -0.00077 against Z (0.87871).**

| candidate | what changed on the disputed cells | queued rows | LB |
|---|---|---|---|
| Z | - | 10,627 | 0.87871 |
| AD | stacker on the month's past pseudo windows | 10,551 | 0.87848 |
| AE | stacker on train-month windows | 10,342 | 0.87755 |
| AF | + occupancy, queued count held fixed | 10,629 | 0.87794 |

Three attempts, three losses. The disputed cells are 1,667 of the official ongoing cells; on the public month the
interventions touch 138-265 rows, and every version has landed below Z.
