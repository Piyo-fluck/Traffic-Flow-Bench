# 121 - The ongoing selector is real evidence, and the decision layer is closed anyway

The user's line: the official ongoing windows are not arbitrary. They are chosen with

    S = 1  iff  IoU(Y, Y_persist) <= 0.9      (`maximum_persistence_iou`, notes 24-25)

against the TRUE future, so a window's presence in the file is evidence about its label, and the quantity to
report is P(Y | X, S = 1), not P(Y | X). We had been using the selector only to build official-like evaluation
windows, never as a condition at prediction time. `occ_lab/selector_posterior.py`.

Y_persist is the last observed queued state per link held over the horizon, and the scored cell set is the
sample submission's own rows, so both sides of the statistic are known at test time.

## The premise is right

| | true future | our prediction |
|---|---|---|
| mean IoU with Y_persist | 0.62 - 0.70 | 0.65 - 0.74 |
| share above 0.9 | **0.000**, by the selector | **4 - 21%** |

We put 10% of our answers in a region the organizers removed, and those windows score far below the rest
(D7_I210_E private: 0.455 against 0.822). The posterior is persistence-shrunk exactly as predicted.

## Conditioning on it is worth +0.0013, and it is genuine

Under an independent-Bernoulli posterior the constrained MAP is reached by flipping the cells nearest the
threshold until the constraint holds - only two moves lower the statistic, extension and retraction.

| ongoing family IoU, whole month | mean | vs AS |
|---|---|---|
| AS | 0.8505 | - |
| constrained MAP at 0.9 | 0.8517 | **+0.0013** |
| extension flips only | 0.8492 | -0.0013 |
| retraction flips only | 0.8488 | -0.0017 |
| **control: the same flips against a window-permuted Y_persist** | 0.8462 | **-0.0043** |
| pushing the violating windows to 0.85 / 0.8 / 0.75 / 0.7 | 0.8512 / 0.8496 / 0.8471 / 0.8444 | +0.0007 .. -0.0061 |
| marginal tilt, logit p + lam (1 - 2 Y_persist), lam 0.1 / 0.2 / 0.4 | 0.8505 / 0.8501 / 0.8477 | 0.0000 .. -0.0027 |

The control separates it: spending the same flips for a reason that carries no selection information costs
0.0043, so the +0.0013 is the selector's own. But nothing stronger works. Pushing past the boundary loses
monotonically, and the marginal tilt reproduces note 118's class-conditional threshold (+0.0003 scale) exactly.
**The exact Bayesian answer IS the boundary projection; everything stronger is mis-specification.** The
indicator carries about 0.15 bits per window - it says WHICH windows are wrong, never HOW.

## The ceiling of the whole decision layer, and it is huge

Keeping the posterior's per-cell ORDER and changing only how many cells each window takes:

| ongoing family IoU, whole month | mean | vs AS |
|---|---|---|
| AS, flat tau = 0.50 | 0.8505 | - |
| knowing each window's true IoU(Y, Y_persist) | 0.8746 | +0.0241 |
| knowing each window's true \|Y\| | 0.9006 | **+0.0501** |
| knowing each window's best prefix length k* | 0.9267 | **+0.0762** |

S_queue's whole remaining gap is 0.063 and ongoing is half of it, so this is the largest oracle in Task 2.

## But the budget is unbiased, and its residual is unpredictable

On the OOF tail (6,969 windows), **the flat threshold takes 83.8 cells and k* wants 84.3** - log-ratio mean
-0.002, sd 0.194. The budget is not systematically wrong. It is right on average and noisy per window, and that
noise is what the oracle is buying.

A LightGBM fitted on the OOF to predict log(k*/k_tau):

| features | 5-fold CV R2 | corr |
|---|---|---|
| posterior only (sorted profile, mass by horizon, counts, spread) | +0.018 | +0.250 |
| + released-history observables (queue count, trend, width, components, hour) | +0.028 | +0.257 |

and applied to the scored months it loses at every strength: -0.0010 / -0.0050 / -0.0079 / -0.0118 at shrink
0.25 / 0.5 / 0.75 / 1.0. Two to three percent of in-domain variance, and it does not transfer.

**Verdict: the ongoing decision layer holds a +0.076 oracle that is essentially irreducible. The remaining
S_queue gap cannot be taken by any rule that reorders or re-budgets the cells; it has to come from the
probabilities themselves.** Note 31 reached the same wall from the other side - the expected-IoU ratio rule that
gains +0.05..+0.09 on onset loses on ongoing, because ongoing cells are correlated in space and time.

One correction to my own reading during the measurement: the oracle does not show the budget is wrong. It shows
that knowing the per-window realisation is worth a lot, which is true of an oracle threshold on any classifier.

## Two more Task 2 items closed the same day

**Item 2, a horizon-dependent w_PS** (`occ_lab/w_ps_by_horizon.py`). The ongoing blend uses one scalar 0.60 for
every horizon step, while p_PS is supervised on the run's own released queues (which pins the near horizon) and
the frozen ensemble carries the train months' dynamics (which is what T+30 needs). Fitted on one month and read
on the other, with the corrector's features regenerated from each new base:

| fit -> test | best (early, late) | transfer | sign |
|---|---|---|---|
| validation -> private | (0.6, 0.7) | +0.0003 | late > early |
| private -> validation | (0.6, 0.5) | -0.0035 | late < early |
| validation -> private, early regime | (0.5, 0.6) | +0.0002 | late > early |
| private -> validation, early regime | (0.4, 0.4) | -0.0007 | flat |

The direction does not reproduce and no transfer exceeds +0.0003. The surface is flat around 0.60 and the two
months disagree about which side. My own prediction (late lower, because the near horizon is what run
supervision pins) was wrong on validation. **w_PS has no horizon structure.**

**Item 3, selector-matched pseudo-supervision** (`occ_lab/pseudo_adapt.py --sel official`). The density-ratio
weighting the plan called for is unnecessary because the numerator's condition is known: the organizers' ongoing
selector is 07-21 h, history coverage >= 0.7, and persistence IoU <= 0.9, while `sup_batch` accepts anything
with two queued history cells and a queued horizon - so the adaptation had been spending its steps on exactly
the trivially persistent windows the official set excludes. Applying the official selector to the supervision:

| ongoing family IoU, global corrector | whole month | at most 4 earlier days |
|---|---|---|
| loose selector (what AR and AS ship) | 0.8505 | 0.8513 |
| official selector | 0.8500 | **0.8489** |

It loses, by -0.0005 and -0.0024. The rejection rate leaves far less supervision per day, and the persistent
windows evidently still teach the dynamics. The run adapter reproduces on this cache (+0.0052 / +0.0031),
which is the same as on the shipped one, so the comparison is clean.
