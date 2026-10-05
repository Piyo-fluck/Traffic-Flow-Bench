# 83 - A fourth ongoing family (DCRNN-style diffusion-GRU + autoregressive decoder): not orthogonal enough

Date: 2026-09-21. Code: `dcrnn_lab/dcrnn.py`, `dcrnn_lab/eval_dcrnn.py`. The user's proposal: the three ongoing
members all emit the six horizons at once, so add a model that propagates the queue state forward horizon by
horizon, and judge it by error orthogonality rather than by its own IoU.

## The model

- Encoder: per-link GRU over the 12 history slots; after every step the link states are mixed by a bidirectional
  diffusion convolution (shifts of 1, 2, 3 links up- and downstream, learned weights).
- Decoder: a GRU cell unrolled six times; each step takes the previous horizon's predicted queue probability
  (scheduled sampling, teacher share 1 -> 0 over the first half of training) and emits the next horizon's logit.
- Inputs and loss identical to production (BCE + 0.5 Lovasz), 30k steps, two seeds. ES BCE 0.0252 / 0.0229,
  against about 0.020 for the transformers - a healthy model.
- Caveat noted before building: the corridor is a path graph, so the diffusion convolution largely repeats the
  CNN's dilated link convolutions. The decoder is the genuinely new part.

## Results (official-like ongoing windows, family IoU)

| | validation | private |
|---|---|---|
| blend (candidate T) | 0.8357 | 0.8233 |
| dcrnn alone | 0.7565 | 0.7314 |
| gs1 / tfm8 / cnn3 alone | 0.7958 / 0.8253 / 0.7989 | 0.7735 / 0.8128 / 0.7813 |
| blend + dcrnn 0.10 | -0.0007 | -0.0010 |
| blend + dcrnn 0.15 / 0.20 | -0.0021 / -0.0031 | -0.0022 / -0.0047 |
| blend + dcrnn 0.15 at T+25/T+30 only | -0.0008 | -0.0004 |

## Why it does not add (the orthogonality check)

- Cell-error correlation with the members: gs1 0.41, tfm8 0.52, cnn3 0.57, blend 0.54 - **the same range as the
  existing members have with each other** (0.41-0.57, note 80). It is not more orthogonal, only weaker.
- Cells where all three members are wrong and the DCRNN is right: **0.13%**. Cells where it is wrong and they are
  not: **1.9%**, fifteen times more.
- By horizon its error grows faster than the blend's (T+5 0.015 vs 0.009; T+30 0.033 vs 0.019), so the
  autoregressive decoder does not help where the error mass is. Feeding the predicted state forward accumulates
  its own error instead.
- Worst on persistence cells (0.076 vs the blend's 0.041), i.e. it loses queues that are already visible.

**Not adopted.** The recurrence was the plausible missing bias, but as built it is dominated: it repeats the same
errors and adds its own. A stronger version (more capacity, longer training, or the decoder on top of the
transformer encoder) would have to beat the orthogonality bar, not the IoU bar.

## Second attempt: a bidirectional spatial selective scan (Mamba-style), `dcrnn_lab/spatial_ssm.py`

The user's second proposal: put the state space over LINKS, not time (the history is only 12 slots), scanning
upstream -> downstream and back with an input-dependent decay, and keep the one-shot 6-horizon head so nothing is
fed back. Implementation notes: the naive log-space cumulative scan overflows over 350 links (NaN); it is solved
in chunks of 8 links with only the chunk carries passed sequentially.

Training is healthier than any previous attempt: **ES BCE 0.0182 / 0.0177**, against 0.020 for the transformers
and 0.023-0.025 for the DCRNN.

| ongoing family | validation | private |
|---|---|---|
| blend (candidate T) | 0.8357 | 0.8233 |
| spatial SSM alone | 0.7447 | 0.7310 |
| blend + SSM 0.10 | -0.0002 | -0.0007 |
| blend + SSM 0.15 / 0.20 | -0.0010 / -0.0022 | -0.0008 / -0.0026 |
| blend + SSM 0.15 at T+25/T+30 | -0.0012 | -0.0004 |

Orthogonality, the criterion that matters:

| | SSM | DCRNN | members with each other (note 80) |
|---|---|---|---|
| error correlation with gs1 / tfm8 / cnn3 | 0.43 / 0.54 / 0.55 | 0.41 / 0.52 / 0.57 | 0.41 / 0.50 / 0.53 |
| all three wrong, the new model right | 0.126% | 0.133% | - |
| new model wrong where they are right | 1.758% | 1.908% | - |
| error rate at T+30 (blend / new) | 0.019 / 0.030 | 0.019 / 0.033 | - |
| persistence cells (blend / new) | 0.041 / 0.072 | 0.041 / 0.076 | - |

**Also not adopted.** It trains better than the DCRNN and its errors overlap slightly less, but the correlations
sit in the same 0.43-0.55 band as the existing members, it rescues 0.13% of cells and spoils 1.8%, and the error
profile by horizon and by cell type is the same shape as everything else, only worse. Two different inductive
biases (recurrent decoder, directed spatial scan) landed in the same place: **what these models get wrong is not a
property of their architecture.**

## LB: AG = 0.87859 (submission 56401712), -0.00012 against Z

- AG = Z with the spatial SSM (mean of two seeds) added to the ongoing blend at 0.10, via
  `scripts/t2_ssm_blend_predict.py` and `--og-pred-dir`. 58 queue rows differ from Z.
- Local estimate: -0.0002 / -0.0007 family, i.e. roughly flat; the LB agrees (-0.00012), the closest
  local-to-LB match of the week.
- Z stays the best.
