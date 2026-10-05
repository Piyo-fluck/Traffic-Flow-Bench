# 119. Candidate AS: the first Task 1 change that moved the leaderboard (0.88405)

Date 2026-09-25. LB **0.88405** against AR's 0.88318, **+0.00087**.

## What it is

AR with Task 1 replaced. Everything else - the queue half (note 110) and the ODME half - is byte-identical.

    width 192, receptive field "all1" (every dilation 1, note 115), so 7.98M parameters
    stage 1: masks we generate ourselves, rate drawn U(0.10, 0.70), 25 epochs  (analysis/t1_ssl.py)
    stage 2: the real Task 1 task, 25 epochs at lr 7e-4
    five seeds, averaged
    each member TTT-adapted per scored split, 1200 steps (note 32: same as 3600)
    production month-deviation base, --month-kappa 2 --month-kappa-q 10

6,735,795 state rows changed, 13,455,776 values. No non-positive speeds, no NaN, min speed 4.54 km/h.

## The local call, and it held

Matched conditions - same 1200 TTT steps, same in-month pseudo-eval, validation split:

| | S_state~ | S_LWR~ |
|---|---|---|
| production trio + TTT | 0.9494 -> 0.9518 | 0.7709 -> 0.7752 |
| **AS, five members + TTT** | 0.9501 -> **0.9521** | 0.7783 -> **0.7819** |
| difference after TTT | **+0.0003** | **+0.0067** |

Predicted total +0.00078; **realised +0.00087**. The Task 1 local surface predicts the leaderboard, which the
Task 2 surfaces never did (note 110).

## Why it works: TTT was being wasted

The single most useful measurement of the whole day. TTT on a model that never saw a self-supervised masking task
**damages S_LWR**; on an SSL-pretrained model it helps:

| | S_state~ | S_LWR~ |
|---|---|---|
| w192 all1, no SSL, + TTT | +0.0012 | **-0.0024** |
| w192 all1, SSL wide, + TTT | +0.0020 | **+0.0026** |

TTT is masked reconstruction, which is stage 1's task. A model that has never solved it gets pulled off
distribution; one that has gets refined. The pre-TTT gap of +0.0035 S_LWR becomes +0.0085 after TTT.

This is also why experiment 3 looked marginal at first: on the local test block without TTT the SSL arms were
+0.0030 S_LWR over their control (1.1 sd) and the three mask distributions - `iso`, `prod`, `wide` - were
indistinguishable (0.65716 / 0.65708 / 0.65729, inside 8% of the noise floor). **The mask distribution does not
matter; being compatible with test-time adaptation does.** The user proposed the two-stage design; it paid for a
reason neither of us predicted.

## The ingredients, each measured

| step | S_LWR | note |
|---|---|---|
| production width 64, 9-seed mean | 0.63635 | 115 |
| width 192 | 0.65135 | 115, +5.7 sd |
| + receptive field all1 (same parameters) | 0.65428 | 115 |
| + SSL pretraining | 0.65729 | this |
| five seeds instead of one | +0.0017 total on the seed curve | 115 |
| + TTT | the sign of TTT's effect on S_LWR flips | this |

Width was dismissed in note 07 on a single run; that was the costliest error in the Task 1 record.

## A bug found on the way

`scripts/t1_ttt_train.py` unpacked `m3.sample_batch` as four values after note 101 gave it five, so the
production TTT pipeline could not run at all on current code. Fixed.
