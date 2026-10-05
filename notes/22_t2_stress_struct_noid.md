# 22 — Task 2: demand-stress helps ongoing a little; dropping link identity hurts; onset is dominated by draw noise

Date: 2026-09-15 · Scripts: `analysis/t2_demand_stress.py`, `analysis/t2_struct_noid.py` · Logs: `work/t2_demand_stress.log`, `work/t2_struct_noid.log`

All numbers are in-month pseudo windows with days < 14, family mean. The format is onset / ongoing / S_queue.

## Demand stress (projected flow / capacity from the train profile × today's level)

| | validation | private |
|---|---|---|
| robust (onset prior-dropout, ongoing no-prior) | 0.286 / 0.817 / 0.552 | 0.391 / 0.827 / 0.609 |
| + stress | 0.325 / 0.827 / 0.576 | 0.396 / 0.841 / 0.619 |

## Link identity vs structure and capacity margins (paired: identical samples, column subsets)

| Variant | validation | private |
|---|---|---|
| base (production robust set) | **0.366 / 0.819 / 0.593** | **0.402 / 0.833 / 0.618** |
| noid (− pos, panel) | 0.292 / 0.806 / 0.549 | 0.381 / 0.830 / 0.605 |
| noid + structure | 0.333 / 0.808 / 0.571 | 0.409 / 0.825 / 0.617 |
| noid + structure + margins | 0.310 / 0.807 / 0.559 | 0.399 / 0.828 / 0.613 |
| noid + all (+ stress) | 0.246 / **0.822** / 0.534 | 0.368 / **0.840** / 0.604 |

- **Removing `pos` and `panel` loses in both months.** Structure features recover part of the loss. Capacity-margin features do not help onset, even though `mg_tight_proj` ranks third in onset gain. The hypothesis that link identity is what fails to transfer is rejected.
- The demand-stress features raise the ongoing IoU in every comparison: +0.010 and +0.014 in the stress run, +0.003 and +0.007 here. That is small but consistent.
- **Onset is dominated by noise.** The same base configuration scored onset 0.286 (stress run) and 0.366 (this run) on validation; the only difference is the random prior-dropout draw. Onset differences below about 0.08 cannot be read from a single draw. This is measured next by `analysis/t2_bag_eval.py` (averaging 5 draws).
- **Bagging** (`analysis/t2_bag_eval.py base 5`, log `work/t2_bag_eval.log`). Five onset models, each trained on
  its own dropout draw and seed:

  | | single draw, mean ± sd (min–max) | averaged probability |
  |---|---|---|
  | validation | 0.347 ± 0.027 (0.302–0.385) | 0.360 |
  | private | 0.421 ± 0.018 (0.392–0.447) | 0.430 |

  Averaging adds about +0.01 onset over a typical single draw and removes draw luck. In production this is
  `scripts/t2_m1_predict.py --onset-bags 5`.
- Consequence for the 0.85 target: the T2 gains that can be demonstrated so far (robust setup plus stress on ongoing) come to about +0.01–0.015 total. T2 alone cannot reach 0.85.
