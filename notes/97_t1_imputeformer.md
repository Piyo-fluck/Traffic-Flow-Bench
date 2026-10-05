# 97. An ImputeFormer-style second family for Task 1

Date 2026-09-24. Code `analysis/t1_m4_imputeformer.py`, `analysis/t1_blend_eval.py`; logs `work/t1_m4.log`,
`work/t1_m4b.log` on Ada. Motivation from the user: Task 1 is almost entirely one family (three dilated CNNs),
and the offline/bidirectional nature of the task is not exploited at the architecture level.

Note 96 had already removed the structural argument (the mask is i.i.d.; long gaps carry error in proportion),
so this was run as a pure function-class bet on the free-flow flow bulk.

## The model

Taken from ImputeFormer (KDD 2024), reduced to its two mechanisms:

    projected temporal attention   the day's 288 slots are summarised onto k learnable prototypes and read back,
                                   which induces low-rank temporal structure instead of free T x T attention
    embedded spatial attention     attention across links whose queries carry a learned link-identity embedding

Three blocks of (temporal, spatial, FFN). Inputs, targets, loss and the fit/ES/test day split are exactly
`t1_m3_cnn`'s, so only the function class changes. The self-supervised hiding was also corrected to draw the
official regime rates (0.20 / 0.30 / 0.50, MASK_SPEC) at (slot, link) level rather than a generic 0-30%.

Two budgets, on the eight panels Ada carries (the local machine has all ten; Ada has the Task 2 eight):

| run | size | schedule | ES S_state | TEST S_state |
|---|---|---|---|---|
| m4 | d 96, 16 prototypes, 0.48M params | 25 epochs x 400 steps, batch 6 | 0.9369 (still rising) | 0.9242 |
| **m4b** | d 128, 24 prototypes, batch 12 | 60 epochs x 400 steps | 0.9474 (flat for 3 epochs) | **0.9353** |
| production CNN (m3all), same panels | | | | **0.9508** |

## Blended with the CNN

| alpha (ImputeFormer weight) | S_state | speed RMSE | flow RMSE | flow, free flow |
|---|---|---|---|---|
| 0.00 (CNN) | 0.9508 | 1.336 | 29.87 | 29.02 |
| **0.10** | **0.9510** | 1.332 | 29.83 | 29.00 |
| 0.20 | 0.9507 | 1.350 | 29.93 | 29.09 |
| 0.50 | 0.9471 | 1.528 | 30.99 | 30.04 |
| 1.00 | 0.9353 | 2.097 | 35.03 | 33.62 |

**The best blend is +0.0002 S_state**, i.e. about +0.00007 on the total, and it helps on five of the eight panels
and hurts on three. Scaling the model closed a third of the standalone gap (0.9242 -> 0.9353 against 0.9508) and
converged, but the complementarity did not appear: at the weight where a complementary member normally shows
itself (0.10-0.20, where the onset STID jumped +0.020) this one is flat.

## Reading

What is measured is this reduced implementation at a ~40 minute budget, not the published model. But the number
that matters for us is the blend, and it says the errors are not complementary: the transformer is worse
everywhere in the same places, including the free-flow flow bulk that note 96 identified (33.6 against 29.0).
Branch closed; the Task 1 lever that has actually moved the LB remains test-time training.
