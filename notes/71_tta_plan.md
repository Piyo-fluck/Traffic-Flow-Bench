# 71 - The direction the 10 ideas point to: traffic-specific test-time adaptation (plan, not yet run)

Date: 2026-09-16 · Source: the user reading of notes 63-70. **No experiment has been run from this note yet.**

## The diagnosis it rests on

| what was changed | ideas | result |
|---|---|---|
| learn a different representation | 1 (Task 1 trunk), 5 (contrastive), 10 (joint Task 1/2/4) | all worse |
| keep the representation, nudge its parameters towards the scored run | **9b (past-day self-supervised adaptation)** | **onset +0.0485 alone, +0.0079/+0.0189 in the blend, LB +0.00091** |

So the train-to-test gap is **not** "the model needs a different representation of traffic"; it is "the same traffic
representation needs its parameters tuned a little to this run". And that shift is estimable from the unlabeled
past days of the scored month - the first onset-side hint in this project that reproduced on private AND moved the
public board (note 70).

Ideas 6 and 7 are demoted for the opposite reason: they improve the ability to FIND the rare bottleneck
(missed-window top-1 0.05 -> 0.30) but that never converts into score, so building yet another rare-onset
classifier is not where the next hour goes.

## The plan, in the order to run it

**A. Adapted Task 1 residual (idea 2 x idea 9b).** Highest priority, lowest cost. Idea 2 measures
`r = x_obs - x_hat(global Task 1)`; after 9b we can measure `r_adapt = x_obs - x_hat(adapted Task 1)`. The global
residual carries the whole run offset (if the run runs 1% hotter, every link shows +1%); the adapted residual
carries only what is anomalous *for this run*, which is exactly the "anomaly, not level" quantity that notes 59/60
kept failing to build by hand. Pipeline: adapted reconstructor -> adapted residual features -> onset LightGBM.

**B. Micro-adaptation on the current window.** After the past-day adaptation, mask part of the visible cells of the
window itself (T-60…T-5, no future) and take 1-5 reconstruction steps before predicting. Legal, and it tunes to the
domain state at the moment of prediction. Only safe together with C.

**C. Parameter-efficient adaptation.** Update a small module instead of the whole trunk: per-block scale/bias
(`h' = gamma h + beta`), a low-rank adapter (`W' = W + AB`), or only the input 1x1 and the head. Compare against the
current full-trunk 9b under identical conditions. If a run-specific calibration is all that is needed (speed scale,
flow scale, temporal responsiveness, spatial coupling), the small module should match or beat the full update, and
it makes B safe on a 12-slot window.

**D. A better adaptation loss.** Masked reconstruction restores the state; Task 2 needs its evolution. Add
next-step and delta terms on past days: `L = L_mask + a*L_nextstep + b*L_delta`, with
`L_delta = ||(x_hat_t - x_{t-1}) - (x_t - x_{t-1})||`. Onset depends on dv/dt and dq/dt more than on v itself.

**E. Meta-learn the adaptation.** Split train into per-day tasks: inner loop = self-supervised adaptation on the
previous day, outer loop = onset loss on the next day. `theta_e = theta - alpha * grad L_recon(past)`, minimise
`sum_e L_onset(future; theta_e)`. Task 2 labels appear only in the outer loop, so test time is unchanged. This turns
9b from an adaptation that happens to help into one designed to help. The big bet, and the first large idea in this
project with a measured reason to fund it.

**F. Ensemble the adaptation trajectory, gated by the self-supervised loss.** Build several adapted models (last day
only / all past days / speed-only / flow-only / small lr / large lr) and weight them by
`w_k ∝ exp(-beta * L_recon_current(M_k))` - a quantity we can observe at prediction time. This differs from the
gating that failed in notes 51/65, which tried to guess which Task 2 prediction was right from 169 windows of
evidence.

**G. Separate slow shift from fast anomaly.** Write `X = B_run + R`, with `B_run` the run-specific baseline the past
days give us and `R` the residual of the current window, and feed Task 2 mainly `R`. The generalisation of A.

## References the user supplied

TAFAS (AAAI 2025) updates a light calibration module from partially observed test data; an IEEE T-ITS 2025 paper
does online test-time adaptation for traffic forecasting by correcting with recent observations rather than
retraining; PETSA (arXiv 2506.23424) updates only small input/output calibration modules and low-rank adapters;
ACM TIST 3474837 meta-learns fast adaptation to new traffic domains from unlabeled data.

## Status

Resuming in about two hours. Candidate U (the three-seed version of the adapted onset CNN) was left building in the
background; its completion needs no reaction.
