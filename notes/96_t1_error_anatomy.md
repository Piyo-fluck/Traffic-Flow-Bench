# 96. Where the Task 1 error is, and two structural shortcuts that are not there

Date 2026-09-24. Code `analysis/t1_gap_anatomy.py`, `analysis/t1_flow_conservation_oracle.py`. Measured on the
real official mask over the local test block (train days 2031-01/02, 59 days x 10 panels, 7,089,289 targets),
using the saved production CNN predictions (`work/preds/t1_m3all_*_test.npz`).

## The mask has no exploitable structure

`trafficflowbench-public/docs/MASK_SPEC.md` settles the geometry: a target is drawn **i.i.d. per (day, timestamp,
link)** by hashing `panel|regime|date|timestamp|link_id` with blake2b against the regime's rate (R1 0.20, R2 0.30,
R3 0.50), and one day carries one regime. There are no blocks by construction; several stations on one link are
blanked together because masking happens at link level.

By how isolated a target is in time:

| group | share | speed RMSE | flow RMSE | speed err mass | flow err mass |
|---|---|---|---|---|---|
| both temporal neighbours observed | 48.3% | 1.283 | 29.65 | 46.6% | 48.3% |
| one side observed | 41.1% | 1.317 | 29.66 | 41.7% | 41.1% |
| **both gone** | 10.6% | 1.377 | 29.77 | 11.8% | 10.7% |

**The cells with no temporal neighbour carry error in proportion to their number.** Speed degrades 7%, flow
0.4%. At R3 alone it is 1.406 against 1.344. So a model whose advantage is reconstructing long gaps has nothing
here to be better at - which removes the stated reason for a low-rank / bidirectional imputer.

## The error is the free-flow bulk, and it is flow

| traffic state | share | speed RMSE | flow RMSE | speed err mass | flow err mass |
|---|---|---|---|---|---|
| **free flow** | **95.6%** | 1.269 | **28.87** | **90.0%** | **90.5%** |
| transition (0.6-0.85 vf) | 0.2% | 3.773 | 61.99 | 1.6% | 0.8% |
| queued | 4.2% | 1.854 | 42.66 | 8.4% | 8.6% |

S_state = 1 - (0.54 x RMSE_v / 25 + 0.46 x RMSE_q / 600) = 1 - (0.0282 + 0.0227) = 0.949. Speed at 1.27 km/h is
close to note 01's noise floor of 0.9; **flow at 28.9 is 1.6 times its floor of 18**, and 90% of that sits in the
easiest cells. Nine tenths of the remaining Task 1 score is an ordinary accuracy problem on ordinary cells.

## Conservation from observed neighbours does not help

Masking is independent per link, so at a masked cell the neighbouring links at the same timestamp are observed
89% of the time. Corridor conservation q(i) = q(i-1) + on - off, with the released ramp flows:

| estimator (free flow) | coverage | flow RMSE |
|---|---|---|
| time interpolation on the link | 100% | 51.18 |
| nearest observed upstream link + ramps | 89.5% | 43.47 |
| nearest observed downstream link + ramps | 89.4% | 46.79 |
| mean of the two | 98.6% | 39.10 |
| **production CNN** | 100% | **28.87** |
| 0.5 CNN + 0.5 conservation | 100% | 31.41 |
| per-cell oracle min(CNN, conservation) | 100% | 24.43 |

The conservation estimate is 10 worse than the CNN and mixing it in makes things worse; on queued cells it is far
worse (80-113), as a steady-state balance should be where vehicles are storing. The oracle's 24.43 is the
min-of-two artefact, not evidence of accessible information. The CNN already sees the whole masked map with its
mask channel, so it is presumably using the neighbours better than the balance does.

## Reading

Both structural shortcuts are closed. What is left for Task 1 is ordinary accuracy on the free-flow flow bulk -
either a better function class, or more of the test-time training that has already produced three consecutive LB
gains. Unlike onset (note 95), **the Task 1 local evaluation can be trusted**: 7M targets, an RMSE metric, no
set-size rule, so a new family can be judged without spending submissions.
