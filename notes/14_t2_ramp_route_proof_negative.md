# 14 — Proof test: horizon ramp flows do NOT raise S_queue (hypothesis from note 13 withdrawn)

Date: 2026-09-14 · Script: `analysis/t2_ramp_leak_proof.py` · Log: `work/t2_ramp_leak_proof.log`
Measurement only. Nothing here was used in a submission.

## Question
Note 13 claimed that the unblanked horizon ramp flows leak the Task 2 target
and "plausibly explain" the ≈ 0.92 top scores. The user asked for proof
without submitting.

## Test
Three predictors on the same windows, scored with local labels on train.
The local labels are validated: persistence gives 0.307 against the official
0.302.
1. **rules**: our legitimate rules v2.
2. **+ramps**: rules, plus links whose attached on-ramp flow at the horizon
   step is below 0.2 × its weekday-type/tod mean. Gaps of ≤ 6 stations
   between flagged links are filled. Onset uses T+30 only.
3. **placebo**: identical to +ramps, but the ramp flows come from another
   day of the same weekday type. If +ramps gained and placebo did not, the
   gain would be same-time horizon information.

## Result (S_queue, family mean over 8 panels)
| Predictor | Official windows (40+40) | Candidates (2,352 onset / 32,701 ongoing) |
|---|---|---|
| rules | **0.707** (onset 0.646, ongoing 0.768) | **0.700** |
| +ramps | 0.575 (−0.132) | 0.607 (−0.093) |
| placebo | 0.575 (−0.131) | 0.597 (−0.103) |

## Why it fails: flag rates
Eligible cells of links with a valid attached on-ramp; flag = on-ramp flow < 0.2 × its tod mean.

| Panel | P(flag \| queued) | P(flag \| free) | Queued base rate | Precision P(queued \| flag) |
|---|---|---|---|---|
| D7_I10_E | 0.544 | 0.051 | 0.015 | **0.138** |
| D7_I10_W | 0.276 | 0.002 | 0.004 | 0.407 |
| D7_I210_E | 0.633 | 0.001 | 0.001 | 0.414 |
| D7_I210_W | 0.106 | 0.005 | 0.044 | 0.475 |
| D7_I405_N | 0.884 | 0.028 | 0.061 | 0.669 |
| D7_I405_S | 0.704 | 0.015 | 0.077 | 0.795 |
| D12_I5_N | 0.615 | 0.008 | 0.064 | 0.833 |
| D12_I5_S | 0.526 | 0.010 | 0.046 | 0.722 |

- Queued cells are rare (0.1–8%), so even a 0.1–5% false-flag rate gives
  a precision of only 0.14–0.83. It is poor on the D7_I10/I210 panels.
- Coverage is limited to links with an attached, valid on-ramp. Onset
  targets (1–8 cells) often sit elsewhere, so a spurious flag replaces a
  good prior pick.

## Conclusion
- **Not proven. Using the horizon ramp flows this way lowers the score, and
  by about as much as the placebo.** The on-ramp collapse signal is real
  (note 13: AUC up to 0.93), but it is not exploitable this way. It fires
  often when the mainline is free, so it adds far more false positives than
  it adds true cells, especially for the 1–8-cell onset targets.
- **The claim that the ramp route explains the top scores is withdrawn.**
  We have no evidence for it. The top 0.92 remains unexplained.
- The draft report is **on hold**, and should not be posted with a claim
  of target leakage.
- The policy stands: horizon ramp flows are still future information, and
  we do not feed them into submissions.
- A stronger test is possible: a learned model with vs without the horizon
  ramp features. Without it, "a smarter use could help" can be neither
  claimed nor excluded.
