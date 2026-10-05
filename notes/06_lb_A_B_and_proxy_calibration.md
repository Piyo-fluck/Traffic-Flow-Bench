# 06 — Leaderboard: M3 for Task 1 → 0.691; Task 4 on the split's own counts → 0.722. The proxy overstates S_LWR.

Date: 2026-09-14 · Rebuild scripts: `scripts/run_submission_A.ps1` (A and B)

## Submissions (Public LB = validation month)
| ref | Content | Public LB | Δ |
|---|---|---|---|
| 56225746 | Official baselines | 0.55343 | — |
| 56228085 | **A**: T1 = M3 v1 CNN (10 panels, 40 epochs, seed 0), T2 persistence, T4 baseline | **0.69091** | +0.1375 |
| 56228092 | **B**: A + T4 = `public_reference.csv` (NNLS on the split's *own* counts + its own prior) | **0.72249** | +0.0316 |

## Decomposition
- **T4**: +0.0316 / 0.20 = +0.158 in S_ODME, so S_ODME is about 0.836 → **0.99**.
  Task 4 is essentially solved by fitting the split's released counts. It is
  no longer worth effort.
- **T1 + T3** (A vs baseline, +0.1375):
  - If S_state on validation ≈ the holdout value 0.952, Task 1 gives
    0.35 × (0.952 − 0.693) ≈ +0.091.
  - The remaining +0.047 is Task 3, so S_physics ≈ 0.355 + 0.047/0.15 ≈
    **0.67**. The local proxy said 0.76.
  - With S_FD ≈ 0.985 that puts the real **S_LWR ≈ 0.51**, against a proxy
    of 0.65.

## Finding
- **F19**: The S_LWR proxy is biased upward for good reconstructions, by
  about 0.14 here, much more than the 0.04 ceiling offset. It was exact for
  the baseline (0 = 0). The likely cause is the proxy's denominator
  Σ|ΔN_true|, built from noisy observations, which is larger than the
  organizer's Σ|rhs|, built from the projected boundary flows. Rankings
  between models should still hold, since the same denominator is used for
  every model, but read absolute proxy values as optimistic. The Task 3
  headroom is therefore larger than it looked: the real S_LWR is about 0.5,
  not 0.65.

## Where the remaining score is (weights × headroom, rough)
| Task | Now (est.) | Headroom to near-perfect | Total-score headroom |
|---|---|---|---|
| T2 Queue | 0.30 (persistence) | → 1.0 | **≈ +0.21** |
| T3 Physics | ≈ 0.67 | → ≈ 0.96 | ≈ +0.04 |
| T1 State | ≈ 0.95 | → ≈ 0.97 | ≈ +0.007 |
| T4 ODME | ≈ 0.99 | → 1.0 | ≈ +0.002 |

**Task 2 is now by far the largest lever.** Task 3 comes second and moves
only through Task 1 accuracy (see [05](05_t1_v2_ramps_dnloss_negative.md)).
