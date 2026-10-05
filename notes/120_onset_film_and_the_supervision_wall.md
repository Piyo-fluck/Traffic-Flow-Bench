# 120 - Onset FiLM works, and onset has no per-run supervision to run it on

Two Task 2 items were measured this week, both of the same shape: adapt on strictly earlier days of the same
run. This note records what the onset one bought on the lab surface, why it cannot ship, and what that implies
for the ongoing one.

## The FiLM, on the lab surface

A frozen `OnsetSTID` with a single FiLM pair as the only trainable tensors,

    h' = gamma * h + beta,    gamma = 1 and beta = 0 on day 1 of a run

adapted per day on strictly earlier tail days' first-morning onset windows with BCE on eligible cells, evaluated
on the `onset_hicov` tail windows (history coverage 0.764 against the official 0.761). The comparison is one
member against an adapted version of itself at fixed blend weights, so note 108's H-inflation confound does not
apply.

The insertion point is not a detail:

| FiLM on | STID alone | delta | AP blend | delta |
|---|---|---|---|---|
| `series` - the history branch only | 0.8060 | **+0.0165** | 0.8675 | **+0.0060** |
| `pre` - the concatenated vector before the head | 0.8032 | +0.0136 | 0.8648 | +0.0032 |
| `both` | 0.7994 | +0.0098 | 0.8651 | +0.0035 |

`pre` scales the vector that already holds the frozen identity, so the FiLM reaches the identity through the
back door; `both` is worse still. The ordering is the design intent reading out: **freeze what the link is, adapt
how the evidence is read.**

The day budget is not the constraint either - capping the adaptation to the 6 most recent past days gives
+0.0145 / +0.0063, statistically the same as the uncapped +0.0165 / +0.0060. Two to six examples are enough.

## The wall

`analysis/t2_run_supervision_budget.py` counts, for every official window, how many release-side pool windows
lie on strictly earlier days of its own month.

| pool | official windows | with at least one earlier-day window |
|---|---|---|
| first-morning onset -> **onset** windows | 80 | **0 (0.0%)** |
| ongoing selector -> ongoing windows | 80 | 68 (85.0%) |
| ongoing selector -> onset windows | 80 | 62 (77.5%) |

The official onset windows sit on the first five days of each month, and a day that holds one has **its own first
morning queue hidden inside that window's horizon** - which is the event the FiLM learns from. So the supervision
the measurement used does not exist where it would have to be used. Not scarce: absent.

## The substitute loses, and loses by distribution, not by fit

The one pool that does reach 77.5% of the onset windows is the broad one - every origin whose history and slot T
hold no visible queue and whose T+30 does.

| broad pool, `series` | STID alone | AP blend |
|---|---|---|
| 40 steps, lr 3e-3, past-day cap 6 | -0.0288 | -0.0028 |
| 40 steps, lr 3e-3, no cap | -0.0194 | -0.0014 |
| 10 steps, lr 3e-3, cap 6 | -0.0090 | -0.0011 |

Weakening the fit walks the loss monotonically toward zero and never through it. That is the signature of a
distribution mismatch, not of overfitting: the best a broad-pool FiLM can do is nothing. **What the FiLM was
learning is the first-morning-onset selector itself, not the run's volume of queue data.**

So the onset FiLM is closed. With note 108 (the onset tail cannot rank mixtures because H is a same-run memory
inflated there) this hardens into a structural constraint: **onset admits no per-run adaptation at all.** What is
left for onset is the train-month prior (the STID's static identity), the window's own released history, and the
decision rule.

## What it means for the ongoing adapter

The same question, asked of item 1's online residual adapter, does not kill it but shrinks it by a factor of
three. The lab walked the whole month, so its windows had 10-20 earlier days behind them; the official ongoing
windows have one to four.

| ongoing family IoU | validation | private | mean | vs global |
|---|---|---|---|---|
| global corrector, whole month | 0.8586 | 0.8424 | 0.8505 | - |
| + adapter, lam 100 step 1.0 | 0.8618 | 0.8484 | 0.8551 | **+0.0046** |
| global corrector, <= 4 earlier days | 0.8452 | 0.8573 | 0.8513 | - |
| + adapter, lam 100 step 1.0 | 0.8465 | 0.8589 | 0.8527 | **+0.0014** |

Positive in both months and monotone in lam and step, so it is real - but +0.0014 on the ongoing family is about
**+0.0002 of the total**, inside the total's seed noise (sd 0.00032). The ramp length was chosen for the
whole-month regime (`lam_d = min(1, past_days/5) * step`), which costs the official windows a further 20-80% of
the step they could take; shortening the ramp is the only lever left on it.

**The general lesson, and it is the second time (note 108 was the first): a per-run adaptation must be measured
in the regime the official windows are in - the number of earlier days they actually have - not on a surface
that walks a whole month.**
