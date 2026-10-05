# 78 - Same-day data before T (allowed since 2026-09-20): no gain, in either use

The organizers confirmed that a Task 2 window may use any released data timestamped at or before its origin T,
including earlier hours of the same day. Two uses were tested.

## 1. Today's demand as a selector (`analysis/t2_onset_today_demand.py`)

Share of windows (truth inside candidate T's top 3) where a score picks the true bottleneck:

| score | T-missed (n=29) | other (n=297) |
|---|---|---|
| candidate T itself | 0.14 | 0.95 |
| today's flow / weekday profile, 3h | 0.48 | 0.52 |
| the same over 6h | 0.41 | 0.49 |
| today's on-ramp flow / profile, 3h | 0.48 | 0.58 |
| today's flow / capacity, 3h | **0.86** | 0.68 |
| control: profile flow / capacity (no same-day data) | 0.72 | 0.71 |
| control: history-only flow / capacity (already in the models) | 0.76 | 0.70 |

The only strong-looking score is today's flow over capacity, and its two controls reach the same level, so what
carries it is the structural margin, not the same-day part. The purely same-day ratios sit at chance. **Negative.**

## 2. Same-day windows added to the 9b adaptation (`run_adapt.py --today-steps`)

Extra adaptation steps on windows drawn from earlier the same day, before that day's first window history.

| run | val onset T-blend | priv onset T-blend | val ongoing T-blend | priv ongoing T-blend |
|---|---|---|---|---|
| 9b (control) | +0.0000 | -0.0012 | -0.0014 | -0.0001 |
| + 10 steps today | +0.0000 | -0.0034 | -0.0020 | -0.0002 |
| + 30 steps today | +0.0000 | -0.0022 | -0.0020 | -0.0004 |
| today only, 30 steps, no past days | -0.0053 | -0.0134 | -0.0021 | -0.0016 |

- Onset alone on private looks better (+0.0125 / +0.0126 family against T, against +0.0036 for 9b), but inside the
  blend it is not: the onset T-blend is unchanged on validation and worse on private.
- Dropping the past days and adapting only on today is clearly worse, which says the run-level signal matters more
  than the day-level one.
- **Negative.** Same-day adaptation does not beat 9b.

## 3. Cross-panel adaptation pool (`tta/run_cross_adapt.py`)

The clarification also allows the other corridors' data before T. Idea 9b draws its 16 windows from the panel's
own earlier days; here each of the 30 steps per day picks a panel (own with probability `--own-share`) and draws
from that panel's earlier dates. `cross_own` (own-share 1.0) is the matched control.

| family delta vs candidate T | val onset alone | val onset T-blend | val ongoing alone | val ongoing T-blend | priv onset T-blend | priv ongoing alone |
|---|---|---|---|---|---|---|
| 9b (verify_00_rep9b) | -0.0118 | +0.0000 | +0.0005 | -0.0014 | -0.0012 | +0.0003 |
| cross_own (control) | -0.0101 | +0.0000 | +0.0012 | -0.0018 | -0.0018 | -0.0004 |
| cross50 | -0.0385 | -0.0007 | **+0.0020** | -0.0008 | -0.0019 | -0.0011 |
| cross_all (uniform panels) | -0.0795 | -0.0090 | +0.0014 | -0.0009 | -0.0063 | -0.0019 |

- **Onset gets worse the more foreign panels enter the pool** (-0.010 -> -0.039 -> -0.080 alone on validation).
  Adapting on other corridors pulls the trunk away from the panel whose window is being predicted.
- Ongoing alone improves slightly with cross50 (+0.0020 validation, window CI +0.0023 [+0.0010, +0.0037]), but
  the ongoing blend is flat again (-0.0008), and private ongoing is not positive.
- **Negative.** The adaptation wants data from the corridor it is predicting, not more data in general.
