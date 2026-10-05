# 63 — T2 onset: the Task 1 reconstruction residual is the strongest alternative ranker found (0.48 all / 0.72 top-3 on the misses)

Date: 2026-09-16 · Script: `analysis/t2_onset_t1residual.py` · Log/data: `work/t2_onset_t1residual.csv`
· The user's ideas ② (reconstruction residual) and ③ (ensemble spread).

Notes 59 and 60 failed the same way: absolute past activity and absolute demand are highest at the **habitual**
bottleneck, so ranking by them reproduces M's error. Residual and spread are a different kind of quantity — "this
place is not behaving the way the traffic model expects today" — and Task 2 has never been given them.

Task 1 already reconstructs the whole scored month, cached per model, so no inference was needed:
`work/sub_m3ttt` (the production test-time-trained reconstruction used by L/M/R) and the three individual CNNs
`sub_m3v1`, `sub_m3v1_s1`, `sub_m3w128` for the spread. **The residual at an observed cell is not zero** — mean
|Δv| 1.5–2.7 km/h, 50–60% of cells above 1 km/h — so the network smooths rather than copies and the cached arrays
carry signal.

| ranking score (candidate Voronoi cells, history only) | all: top-1 / top-3 | **missed (40): top-1 / top-3** |
|---|---|---|
| M's own probability | **0.76** | 0.05 |
| signed speed residual, more negative = "slower than expected" | 0.07 / 0.18 | **0.00 / 0.10** |
| signed flow residual | 0.11 / 0.35 | 0.03 / 0.33 |
| **\|speed residual\|** | 0.47 / 0.76 | 0.30 / 0.68 |
| residual standard deviation | 0.39 / 0.65 | 0.30 / 0.55 |
| Task 1 ensemble spread | 0.38 / 0.67 | 0.23 / 0.45 |
| **\|residual\| × spread** | **0.48 / 0.73** | **0.35 / 0.72** |
| (note 60 for comparison: local flow / path load / ORACLE demand) | 0.21 / 0.25 / 0.24 | 0.35 / 0.42 / 0.45 |

## Reading

1. **The sign does not work.** The hypothesis was "today this link is slower than the model expects". That score
   ranks the true bottleneck **first in 0% of the missed windows** (median rank 8 of ~9). What works is the
   **magnitude** — how badly Task 1 fails to explain the place, in either direction.
2. **`|residual| × spread` is the best non-M ranker measured anywhere in this project**: 0.48 top-1 over all
   windows, against 0.21–0.28 for every demand- or activity-based score in notes 59/60. It is genuinely new
   information: Task 2 has only ever seen the observations, never the traffic model's failure to fit them.
3. On the missed windows it reaches 0.35 top-1 and **0.72 top-3** — below the 0.5 top-1 bar fixed in advance for a
   standalone ranker, but a strong candidate-narrowing signal.

## Next

Not a standalone ranker but **features**: per link, the signed and absolute residuals over the last 15 and 60 min,
their dispersion, the ensemble spread, the product, and their within-window ranks, added to H's onset feature set
and measured paired (3 seeds × 5 bags, official-like windows, alone and in M's blend), exactly as notes 44/50 did.

Caveat to carry into that test: the training rows come from the **train** split, and the cached Task 1
reconstructions cover the scored months only. Building the same features for training needs a Task 1 inference
pass over train (or the features must be built from the train-run models' own out-of-fold reconstructions, which
is the cleaner choice because the scored-month reconstructions come from a test-time-adapted model).
