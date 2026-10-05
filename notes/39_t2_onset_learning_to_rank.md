# 39 — Task 2 onset: learning-to-rank (LightGBM lambdarank) does not beat the binary model

Date: 2026-09-15 · Script: `analysis/t2_onset_rank.py` · Log: `work/t2_onset_rank.log`

Onset is a per-window top-k selection, so the ranking objective of the "select top-k per group" problems was
tried. The comparison is paired: H's features, the same prior-dropout draws, 5 bags each.
- **binary**: H's model plus the expected-IoU ratio rule.
- **rank**: lambdarank (one query group per window, NDCG@1/3/5). k comes from the binary rule and the links are
  ordered by the rank score.
- **hybrid**: links ordered by the mean of the two rank positions.

It is scored on the blank-aware first-morning onset windows.

| onset IoU (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| binary | 0.793 | 0.765 | 0.824 | 0.798 |
| rank | 0.790 (1/1) | 0.770 (3/1) | 0.819 (0/1) | 0.794 (0/2) |
| hybrid | 0.793 | 0.765 | 0.824 | 0.798 |

- Top-1 hit is 0.49–0.54 and top-3 is 0.77–0.87 for every variant. Ranking does not change which link comes
  first.
- The hybrid gives exactly the binary IoU. k is small (1–3), and both models agree on the top links. The misses
  are windows where the queue appears at a link neither model ranks near the top. The problem is the information,
  not the objective.
- Conclusion: not adopted. None of the researched top-competitor techniques gives the +0.007 the user asked for.
  - Learning-to-rank: nothing.
  - Stacking: estimated +0.002…+0.005.
  - Temporal experts averaging: unknown.
  - Test-time adaptation: weak under our data rules.
