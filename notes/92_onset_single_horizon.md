# 92. Onset as a single-horizon problem: a much better model, and no better blend

Date 2026-09-23. Code `occ_lab/train_onset.py`, `adapt_onset.py`, `eval_onset.py`; logs `occ_lab/logs/onset30_*`,
`adapt_onset_*`. Trigger: the organizers' announcement (`notes/onset_discussion_2026-09-23.txt`) that every
queue_onset window is selected as the earliest origin whose horizon contains a queue, so the queue is at T+30 in
all 120 windows and T+5..T+25 are empty.

## What was already right

Our pipeline places onset predictions only at T+30 (`pred[j, F - 1]` in `scripts/t2_cnn_blend_predict.py`, the
rest of the map stays zero) and decides with the expected-IoU ratio rule there; the training windows come from
`earliest_onset`, the same selector condition. So the announcement's main implication - do not spread a
prediction over six steps - was already in place, and the IoU-cost trap it warns about was never paid.

## What was left: the loss

The onset windows entered training with the loss spread over all six horizons, five of which are empty by
construction and never read. `train_onset.py` fine-tunes on onset windows only with BCE + 0.5 Lovasz **at T+30
alone** (8k steps, lr 2e-4, from the production checkpoints; the listwise term stays off, note 46).

### Single-model onset IoU (official-like windows, ratio rule c = 1.0, family mean)

| model | validation | private |
|---|---|---|
| **onset30 (single-horizon, 4 seeds)** | **0.7504** | **0.7347** |
| adapted CNN (9b) - the slot it would replace | 0.7412 | 0.6787 |
| production cnn3 | 0.6835 | 0.6585 |
| LightGBM H | 0.7652 | 0.7981 |

**Training on T+30 alone is worth +0.067 / +0.076 against cnn3 and +0.009 / +0.056 against the adapted model.**
Adding idea 9b's past-day adaptation on top (`adapt_onset.py`, same 30 steps / lr 1e-4) gives the best standalone
onset model measured anywhere: **0.7912 on validation**, above the production blend itself (0.7862) - but on
private it is 0.7337, no better than unadapted.

### In the production blend (0.7 LightGBM H + 0.3 CNN), it loses everywhere

| variant | validation | private |
|---|---|---|
| **production (T/AI): .7H + .3 ad9b** | **0.7862** | **0.8227** |
| .7H + .3 base ADAPTED (our rerun of 9b, harness check) | 0.7862 | 0.8180 |
| .7H + .15 ad9b + .15 onset30 | 0.7841 | 0.8206 |
| .7H + .3 onset30 | 0.7783 | 0.8163 |
| .7H + .3 onset30 ADAPTED | 0.7823 | 0.8122 |
| .6H + .4 onset30 ADAPTED | 0.7897 | 0.8071 |

The harness reproduces production (our 9b rerun matches on validation exactly and is 0.0047 low on private), so
the comparison is fair. Every mixture containing the single-horizon model is below the production blend on both
months.

## Reading

The same shape as the ongoing work (notes 84, 85): **a better member is not a better blend.** The adapted CNN
earns its 0.3 weight by being wrong where LightGBM H is wrong in a different way, not by its own accuracy;
onset30, fine-tuned from the same CNNs on the same windows, is more accurate and less complementary. Nothing
here is worth a submission.
