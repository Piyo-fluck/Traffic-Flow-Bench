// filepath: notes/61_t2_ongoing_boundary_oracle.md
# 61 — T2 ongoing: the queue-boundary band is worth +0.012 total, and it is the largest measured headroom left

Date: 2026-09-16 · Script: `analysis/t2_ongoing_boundary_oracle.py` · Log: `work/t2_ongoing_boundary_oracle.log`
· The user's track ①: stop predicting per-cell occupancy, predict the queue's ENDS instead. Oracle measured first.

## A correction I had to make mid-way (recorded because it nearly killed the track)

The first implementation built the oracle by matching each queue run visible at T to the truth run it overlaps most
and predicting that truth run exactly. It returned **0.596**, i.e. *below persistence* (0.666, freezing the T
queue). An oracle with no false positives cannot score below persistence, so the matching was broken. Had the
sanity line not been printed next to it, the conclusion would have been "track ① is refuted" and a **+0.012**
direction would have been discarded. Every oracle from here on is printed next to a known lower bound.

The rebuilt oracle needs no run matching: predict every TRUE cell within `k` links of the queue visible at T, and
nothing beyond. It has no false positives by construction, so it is a proper ceiling.

## Where the true queue cells actually are

| distance from the queue visible at T | validation | private |
|---|---|---|
| **d = 0 (already a queued link at T)** | **0.838** | **0.839** |
| d = 1–2 | 0.068 | 0.070 |
| d = 3–4 | 0.025 | 0.025 |
| d ≥ 5 (a new queue elsewhere) | 0.070 | 0.067 |

93% of the truth sits within 4 links of the existing queue. Only 7% is out of reach of any boundary formulation.

## The ceiling

| | validation | private | Δtotal |
|---|---|---|---|
| R (production ensemble, τ 0.45) | 0.8501 | 0.8490 | — |
| ORACLE, true cells within **0** links | 0.8154 | 0.8162 | **−0.005** |
| ORACLE, true cells within **2** links | 0.9012 | 0.8991 | **+0.0077** |
| ORACLE, true cells within **4** links | **0.9305** | **0.9281** | **+0.0121** |
| persistence (sanity lower bound) | 0.6660 | 0.6644 | |

Per horizon step (validation, R vs the ±4 oracle): +5 0.908/0.971, +10 0.888/0.958, +15 0.868/0.946,
+20 0.850/0.933, +25 0.832/0.917, +30 0.821/0.906. **The gap widens with the horizon** (0.063 → 0.085), as note 52
predicted.

## Reading

1. **Perfect d0 alone is worse than R** (0.815 vs 0.850). R already predicts a good part of the growth beyond the
   current queue, so "track the existing object" is not by itself an upgrade — the formulation must not *replace*
   the cell model.
2. **The reachable headroom is the 1–4 link band**: +0.0077 (±2) to +0.0121 (±4) total, concentrated at T+20…T+30.
   This is the largest headroom measured anywhere in T2 that is not blocked by an information limit, and it sits on
   the ongoing side, which transfers to the LB at 0.44–0.76 (note 58).
3. Displacement targets for an ordinal model, from the matched runs: tail median +0/+1 links (p90 +7/+8), head
   median −1 (p10 −8/−9), about 60% of runs move at all, and 33–41% of runs clear entirely by the step.

## What to build

Not a replacement, but a **band specialist**: a model whose training rows are only the cells within ±4 links of the
queue visible at T (about 16% of the cells, carrying the whole remaining loss), with the boundary-displacement
framing as its target or its features, blended into R's ongoing probability on that band alone. R keeps the d0
interior and the far field.

Expected value: capturing a third to a half of the ±4 oracle gives +0.004…+0.006 ongoing-side locally, which the
note-58 calibration halves to **≈ +0.002…+0.003 on the LB** — the best remaining prospect, and roughly twice what
four more transformer seeds would buy.
