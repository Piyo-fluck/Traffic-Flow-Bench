# 33 — Leaderboard: candidate L scores 0.86893 (+0.0435 over E); the official-like pseudo-eval predicted it

Date: 2026-09-15 · Submission ref 56242481 (01:31 UTC) · Build: `scripts/run_submission_L.ps1`

| Submission | Public LB | Δ vs E |
|---|---|---|
| E (T1 3 CNNs; T2 train-prior models; T4 public_reference) | 0.82542 | — |
| teammate (09:04 JST) | 0.84077 | +0.0154 |
| **L** | **0.86893** | **+0.0435** |

L's content:
- **T1**: the 3 production CNNs with the month-deviation base (κ 2 / 10) and self-supervised test-time training per scored month (note 32).
- **T2 onset**: prior-dropout model with 5 bags (notes 20, 22), decided per window by the expected-IoU rule (ratio,
  c 1.0; note 31).
- **T2 ongoing**: no-prior model with demand-stress features, τ 0.65 (notes 22, 29).
- **T4**: public_reference (note 06).

The pre-submission estimate was ≈ 0.864 (range 0.853–0.870), built from the blank-aware official-like pseudo-eval
(notes 26, 28, 31) plus T1 in-month pseudo-eval gains. **The LB landed at the top of that range**, which confirms
that the official-like windows (first-morning onsets, official-blank-aware) track the Task 2 leaderboard. The
earlier generic in-month pseudo windows did not. The largest single gain came from matching the evaluation to the
official window selection and then optimising the decision for the metric (expected IoU), not from new features.

Experiments were stopped after this score, at the user's instruction.
