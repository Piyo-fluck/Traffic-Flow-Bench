# 00 — Official baselines and the submission pipeline

Date: 2026-09-14 · Data: post-rebuild release (2026-09-11)

## What we did
- Ran the organizers' three baselines per split (validation, private), merged
  them with `scripts/assemble_submission.py`, and submitted. We can rebuild the
  file with `scripts/run_official_baselines.ps1`.

## Numbers
| Check | Result | Reference |
|---|---|---|
| Task 1 local, D12_I5_N train | S_state 0.6255 (R1 0.618 / R2 0.632 / R3 0.626) | repo doc 0.6252 |
| Task 1 error, D12_I5_N train | speed RMSE 13.4 km/h → S_v 0.46; flow RMSE 109 veh/h/lane → S_q 0.82 | |
| Public LB (ref 56225746) | **0.55343** | official naive 0.5534 |
| Public LB top on 2026-09-14 | 0.924 (post-rebuild) | |

## Takeaways
- The pipeline is sound end to end. Row counts match `submission_key.csv` exactly:
  state 3,451,376 + 3,284,419; queue 87,000 ×2; odme 35,354 ×2.
- Headroom vs perfect, in total-score units: Queue +0.21, State +0.11,
  Physics +0.09 (only moved through Task 1), ODME +0.03.
- In Task 1, speed is the weak channel.

## Gotchas
- The Task 2 builder's `--split all` covers train+validation, so private is
  missing. Build per split.
- The Task 4 official baseline solves against train counts. The split's own
  counts (`public_reference.csv`) are legitimate input.
