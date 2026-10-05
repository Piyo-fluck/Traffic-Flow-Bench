# 48 — LB: P = 0.87429 (−0.0008 vs M); the pseudo-eval estimate (≈ 0.879) did not materialise

Date: 2026-09-15 (14:00 UTC) · ref 56256035 · file `work/sub_P_ttt_t2gsmap_t4ref/submission.csv`

- **P = M's T1/T4 + T2 from `work/sub_t2P`:**
  - onset 0.6·(0.7 H + 0.3 CNN×3) + 0.4·analog map (K40 pca; note 47);
  - ongoing 0.5·global-state LightGBM + 0.5·CNN×3 at τ 0.55 (candidate N; note 43).
- **Public LB 0.87429** vs M 0.87505: **−0.0008.** The estimate was ≈ M + 0.004 (N's ongoing ≈ +0.002, map
  ≈ +0.002).
- Against M, only `queue_pred` differs (about 840 cells). Against N, only 19 onset cells in 8 windows.
- **Why this is inconclusive:**
  - The public LB scores only 5 onset + 5 ongoing official windows per panel of the validation split. One window's
    IoU moving by 0.2 shifts the total by about ±0.001–0.002.
  - The pseudo-eval estimates rest on about 170 (onset) and about 370 per panel (ongoing) official-like windows,
    which are not the LB's windows.
  - So a −0.0008 result does not refute the pseudo-eval gains, but it does not confirm them either.
- N alone was not submitted, so the LB cannot separate the ongoing change from the onset map.
- **Best remains M = 0.87505.** One team submission was left on UTC 9/15 after P.

## Q = M + the analog onset map only (ref 56256172, 14:07 UTC): **0.87345**

- Q = M's ongoing, T1 and T4, with P's onset. It differs from M in only 19 onset cells: validation 13 cells in 5
  windows, private 6 in 3. Files `work/sub_t2Q`, `work/sub_Q_ttt_t2map_t4ref/submission.csv`, log `work/build_Q.log`.
- **Decomposition on the public LB (validation official windows):**

  | | public LB | vs M |
  |---|---|---|
  | M | 0.87505 | — |
  | P (N's ongoing + map) | 0.87429 | −0.0008 |
  | Q (map only) | 0.87345 | **−0.0016 = the map's effect** |
  | N's ongoing alone, implied by P − Q | ≈ 0.8758 | **≈ +0.0008** |

- **The map hurt on the public LB.** Its 13 validation cell changes cost about −0.011 onset IoU on the 40 validation
  onset windows.
  - The pseudo-eval had the map at +0.013 (filled) / +0.008 (masked) on validation-like windows. That estimate
    rested on about 170 windows; the LB has 40 onset windows.
  - The map's strongest pseudo-eval gains were on PRIVATE (the final leaderboard), which the public LB cannot show.
- **N's ongoing change (global-state LightGBM, τ 0.55) looks positive (+0.0008),** consistent with its pseudo-eval.
- For the final selection, the best public evidence favours M, or M + N's ongoing (not submitted; implied ≈ 0.8758).
