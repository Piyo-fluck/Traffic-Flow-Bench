# 173 Task 1 per-panel TTT (candidate CH)

## Why

- Task 2 note 134: adapting the SSL w192 CNN per corridor beats pooled adaptation, even with 4x the steps. The run
  shift is corridor-specific.
- Task 1's TTT was pooled: one adaptation per (member, split) over all 10 panels. Note 102 listed per-panel TTT as
  untried.

## What

AS's five SSL w192 all1 members (`work/models/t1_ssl_wide{,_s1..s4}_all.pt` on Ada), each adapted separately on
every (split, panel):

- Recipe as AS: masked reconstruction, lr 2e-4, bs 16 crops of 64 links, rate U(0.10, 0.50).
- 600 steps per panel (pooled AS: 1200 steps over all 10).
- 100 checkpoints in Ada `work/sub_ppttt/ttt_<split>_<i>_<panel>.pt`.

Code changes:

| file | change | backup |
|---|---|---|
| `scripts/t1_ttt_train.py` | `TTT_PER_PANEL=1`, `TTT_ONLY` (subset of members) | `*.pre_pp` |
| `scripts/t1_predict_ckpt.py` | `{panel}` in a `--ckpt` path loads per-panel checkpoints | `*.pre_pp` |

Run script `work/run_ada_ppttt.sh`; 10 jobs on 4 GPUs, 1 h 47 min.

## Build

1. Inference as AS (5-member mean, month kappa 2 / 10) -> `work/sub_ppens5/pred_*.npz`.
2. `t1_npz_to_state.py` (run locally; Ada lacks the templates).
3. `swap_task_rows.py --base work/sub_BB --task state` -> `work/sub_CH/submission.csv`. 6,735,795 state rows,
   13,389,439 values changed; queue and ODME rows are BB's.

Mean |change| vs AS at target cells:

- speed 0.12-0.26 km/h
- flow 2.6-6.1 veh/h/lane

All finite, min speed 5.9 km/h. No local evaluation was run (the user asked for a candidate directly; the in-month
surface could not separate 1200 from 3600 steps either).

## LB

CH public **0.88625** vs BB 0.88697: **-0.00072**. Only Task 1 differs from BB, so per-panel TTT (600 steps per
panel) is worse than AS's pooled TTT on the leaderboard. The in-month evaluation (`work/run_ada_ppeval.sh`,
target-geometry hide, both months) was still running when CH was scored.

## Early-stopped per-panel TTT (candidate CI)

Code: `TTT_ES=1` in `scripts/t1_ttt_train.py`; run script `work/run_ada_esttt.sh`, 1 h 55 min on 4 GPUs.

Setup, per (member, panel):

- 50 % of the panel's visible cells, in another day's target geometry, are held out of the TTT input.
- Every 25 steps up to 600, the held-out cells are scored with J = 0.35 S_state~ + 0.10 S_LWR~.
- The best step is saved; step 0 (no TTT) is a candidate.

Best steps over the 100 (member, panel, split) runs:

- 0 steps: 4 (D12_I405_S, both months)
- 1-100 steps: 9
- more than 100 steps: 87, many at 400-600

Gain over step 0, by panel: +0.0002 (D12_I405_S) to +0.0026 (D12_I5_S); positive everywhere, similar in both
months. The curves rise in the first ~50-75 steps and then stay flat with noise of about +-0.0005. The held-out
surface therefore does not explain CH's LB loss at a fixed 600 steps.

CI = BB with Task 1 from these checkpoints (`work/sub_esens5` -> `work/sub_CI/submission.csv`), 13,370,481
values changed. Mean |change| at target cells: vs CH 0.07-0.18 km/h and 2-5 veh/h/lane; vs AS 0.10-0.22 km/h and
2-4 veh/h/lane.
