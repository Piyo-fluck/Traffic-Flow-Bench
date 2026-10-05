# 18 — Pseudo-evaluation inside the scored months: the T1 month base helps (+0.006), while the train holdout said it hurt

Date: 2026-09-14 · Scripts: `analysis/t1_pseudo_eval.py`, `analysis/t2_pseudo_eval.py`
Follows [17](17_run_shift_between_splits.md). Train is one simulation run, so no train holdout can show a
cross-run gain. The validation/private masked layers still show most cells, so labelled pseudo-tasks can be
built *inside* the scored months.

## Task 1: hide 20% of the visible eligible cells, reconstruct, score against the hidden observations
Log `work/t1_pseudo_eval_validation.log`. CNN: w64 seed 0 production checkpoint. The base is the train
table plus the month deviation, computed after hiding and shrunk by κ pseudo-days (same κ for speed and flow).

| κ | Pseudo S (family mean, validation) |
|---|---|
| none (production E) | 0.9413 |
| 10 | 0.9464 |
| **4** | **0.9477** |
| **2** | **0.9478** |
| 1 | 0.9472 |

- **F36**: Inside the shifted month the month base helps on every panel. Speed RMSE falls by about 15%
  (for example D7_I10_E 1.73 → 1.40 km/h, D7_I210_E 2.05 → 1.52). Flow improves at κ = 10 but degrades
  slowly as κ gets small, so speed and flow want different shrinkage.
- **F37**: Pseudo S without adaptation (0.941) sits below the train holdout (0.952). Task 1, and through
  it Task 3, also pays the run shift. The within-train check (note 17) showed a small *loss*, which
  confirms that only an in-run evaluation can judge this.
- Next: separate κ for speed and flow, on validation and on **private** (the final ranking).

### Separate κ for speed and flow (logs `work/t1_pseudo_eval_{validation,private}_pairs.log`)
| (κ_speed, κ_flow) | validation | private |
|---|---|---|
| (none, none): production E | 0.9413 | 0.9407 |
| (2, none) | 0.9479 | 0.9476 |
| **(2, 10)** | **0.9482** | **0.9480** |
| (2, 20) | 0.9482 | 0.9480 |
| (4, 10) | 0.9479 | 0.9477 |
| (1, 10) | 0.9479 | 0.9478 |
| (2, 4) | 0.9480 | 0.9479 |

- The two scored months agree: **+0.0069 / +0.0073 pseudo S** at (2, 10). That is about +0.0025 total
  through T1, plus whatever the better density brings to T3 (S_LWR is very sensitive to density error).
- **Adopted for T1 production: κ_speed = 2, κ_flow = 10** (`scripts/t1_predict_ckpt.py --month-kappa 2 --month-kappa-q 10`).

### Production with the month base (log `work/t1_ens3_month_pipeline.log`)
- Output: `work/sub_m3ens3_month/state_{validation,private}.csv`. The 3-CNN mean uses κ_speed = 2 and
  κ_flow = 10, and every panel matched its template.
- Mean month deviation per panel is 0.8–3.1 km/h in speed and 8–21 veh/h/lane in flow. The shift is
  largest on the D7_I405 and D12_I5 panels and smallest on the D12_I405 panels.
- The pipeline runs in three steps:
  1. `scripts/t1_tables_cache.py` (torch-free);
  2. one `scripts/t1_predict_ckpt.py --npz-only` process per panel, with retries;
  3. `scripts/t1_npz_to_state.py`.
- It is split this way because a single long torch+numpy process crashed natively on Windows
  (`0xc000070a` in ntdll) twice.

## Task 2: pseudo windows from the month's visible cells
Log `work/t2_pseudo_eval_validation.log`. Pseudo origins run every 15 min. Onset windows are few
(2–13 per panel), because Task 1 masking breaks the "first queued at T+30" pattern, so the onset numbers
are noisy. Ongoing windows number 164–1337 per panel.

| T2 method (validation month) | onset | ongoing | **pseudo S_queue** | LB-implied S_queue |
|---|---|---|---|---|
| Rules v2 (C') | 0.454 | 0.697 | **0.575** | ≈ 0.56 |
| Model, train priors (E) | 0.424 | 0.791 | **0.607** | ≈ 0.635 |
| Model, month-past prior κ = 8 | 0.430 | 0.808 | 0.619 | |
| **Model, month-past prior κ = 4** | 0.440 | 0.808 | **0.624** | |
| Model, month-past prior κ = 2 | 0.427 | 0.808 | 0.618 | |
| Model, κ = 4, + month P_on | 0.426 | 0.808 | 0.617 | |
| Model, κ = 2, + month P_on | 0.411 | 0.808 | 0.610 | |

- **F38**: The in-month pseudo-evaluation tracks the leaderboard. Its levels and its rules → model
  ordering match the LB-implied values (0.575 vs 0.56; 0.607 vs 0.635), unlike the train holdout (0.688 and
  0.837). **Use it as the primary offline validation for any run-shift question from now on.**
- **F39**: The month-past queue prior (κ = 4, past days only) adds +0.017 pseudo S_queue, mostly on ongoing
  (0.791 → 0.808). That is about +0.005 total. Adapting the onset-start prior P_on too does not help.
- The **private** T2 pseudo-evaluation was stopped on user request at 7/8 panels (2026-09-14 ~22:40),
  before it printed scores. Rerun `py analysis/t2_pseudo_eval.py private` to confirm κ = 4 on private.
- Candidate uploads prepared for 2026-09-15, both verified against the template:
  - **F1** = E + T1 month base: `work/sub_F1_ens3month_t2m1_t4ref/submission.csv`.
  - **F2** = F1 + T2 month-past prior κ = 4: `work/sub_F2_ens3month_t2m1month_t4ref/submission.csv`.
    It differs from F1 in 297 queue cells only.
  - Rebuild both with `scripts/run_submission_F.ps1`.
- The run shift also hurts ongoing (pseudo 0.79 against a holdout of 0.90), so models trained on one run
  under-adapt. Next idea: learn from the scored month's own past-day pseudo windows (same user rule, past
  days only).
