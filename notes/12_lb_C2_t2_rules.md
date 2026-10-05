# 12 — Leaderboard: Task 2 rules give 0.801 (+0.079), short of the +0.12 expected

Date: 2026-09-14 · Rebuild script: `scripts/run_submission_C2.ps1`

## Submission
| ref | Content | Public LB | Δ vs B |
|---|---|---|---|
| 56228092 | B: T1 M3 v1, T2 persistence, T4 public_reference | 0.72249 | — |
| **56229889** | **C'**: B + T2 rules v2 (onset: T+30, top-3 of P_on + 0.5·P_q; ongoing: min v/vf over the last 3 steps < 0.8) | **0.80099** | **+0.0785** |

## What it implies
- Only Task 2 changed, so ΔS_queue = 0.0785 / 0.30 = **+0.262**. Starting
  from the official persistence value of 0.3017 on validation, that gives
  **S_queue ≈ 0.56**.
- Local estimates on train windows were onset 0.60–0.65, ongoing 0.77, so
  S_queue ≈ 0.69–0.71. The shortfall is about **0.13 in S_queue**.
- One submission cannot tell which condition fell short. Both of these
  decompositions give 0.56:
  - onset ≈ 0.35, with ongoing at the expected 0.77;
  - ongoing ≈ 0.53, with onset at the expected 0.60 (unlikely, since
    ongoing persistence alone is already ≈ 0.57–0.60).

## Hypotheses for the gap (to test)
- **H11**: *Label mismatch at onset.* The organizer labels come from the
  noise-free state, while ours come from the noisy observation. For
  near-threshold onsets, the truth cells at T+30 could be different or
  more numerous links, which hurts a tiny top-3 target most.
- **H12**: *Validation onsets are less recurrent.* The March demand draws,
  or the incident schedule (5 incidents in validation), put onsets at
  non-recurrent places or times. Incident-driven onsets are not in the
  priors.
- **H13**: *The window-selection rule differs from our reading* (for
  example, onset defined on the noise-free history). Then "only T+30"
  might not always hold on validation.
- A diagnostic submission separates the two conditions. Keep the onset
  rule, put ongoing back to persistence; the score then gives onset directly.
  The file is prepared in `work/sub_diag_onset_only/`.
