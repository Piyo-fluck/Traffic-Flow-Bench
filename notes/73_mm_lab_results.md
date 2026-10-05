# 73 - Onset multimodality hypothesis (mm_lab, run by the user on Ada, 2026-09-17)

Sources: Ada `mm_lab/reports/01-03*.md`, raw data in `mm_lab/out/verify_*`. Kit: `mm_lab/EXPERIMENTS_MM.md`.
Scope: official-like onset windows, 169 validation / 162 private. "Missed" means candidate T's IoU is 0
(34 windows).

## Results

| item | result | verdict |
|---|---|---|
| 1. neighbour multimodality | Missed windows, k=50: truth share >= 0.1 in 56%, truth mode in the top 3 in 62% (k=100: 82%). Top-mode share about 0.85 | information partly present, but strongly unimodal |
| 2. adaptation delta, as a rank | Missed windows: delta~ top-1 41% / top-3 56%. **T's own ranking: top-1 12% / top-3 85%** | a selection signal |
| 2. direct correction `logit p_T + l * delta~` | Cross-split onset family gain (validation / private) for adaptation RNG seeds 3/17/31: +0.0167/+0.0044, +0.0130/+0.0003, +0.0130/+0.0011. Best lambda 0.1-0.25; worse from 0.5 up | only survivor, small on private |
| 3. counterfactual margin | Missed windows: top-1 29% / top-3 65%, no better than T | no |
| 4. multi-hypothesis, K=1 / 3 / 5 / 5 with frozen trunk | Any-head top-1 hit on missed windows goes 0.12 -> 0.41-0.71. Selector and pi-mixture are much worse on private (-0.04 to -0.15 alone, T-blend -0.005 to -0.014). The winning head takes 86-88% of windows | no |

## Readings

- **T's top 3 already contains the truth in 85% of the missed windows.** Candidate generation is not the
  bottleneck; K heads (0.41-0.71) generate candidates *worse* than T's own top 3. The whole problem is
  **selection among T's top 2-3 peaks**. Notes 51/65 gating failed on exactly this, and the multi-head selectors
  fail here too.
- Delta~ is the one selection signal that transfers across months. Structurally, the correction extrapolates
  9b's direction ("adapt a bit more along what adaptation changed"). It may mean 9b is under-applied in the blend.
- Expected value of the delta correction: private family gain of about +0.002 on average, so total
  +0.0003 before the usual LB halving. It needs the unadapted rec_s20 probabilities on the official windows,
  which is cheap. Use lambda = 0.10 (conservative).
- MDN (5), precursors (6) and the cross-panel audit (7) were not run. The user's report closes the exploration.

## Candidate V (T + delta correction, lambda 0.10), built 2026-09-17: identical to T, not submitted

- Official-form check (`analysis/t2_delta_correction_check.py`). The adapted model is the cached 9b that T ships;
  the base is rec_s20 without adaptation.
  - Taking the median over all links or over the eligible links makes no difference.
  - Lambda 0.10 gives family +0.0146 on validation and +0.0187 on private. The private window CI includes 0.
  - Lambda 0.10 changes the selection in 11/169 (validation) and 11/162 (private) official-like windows.
- Build:
  - `scripts/t2_adapt_official.py --steps 0` wrote `work/sub_t2base_rec20`.
  - `scripts/t2_cnn_blend_predict.py` has two new options, `--onset-delta-base-dir` and
    `--onset-delta-lambda`. Backup: `work/t2_cnn_blend_predict.py.bak_delta`.
  - Output: `work/sub_V_ttt_t2delta_t4ref/submission.csv`.
- **On the 80 official onset windows, lambda 0.10 and 0.25 change 0 selections, 0.5 changes 1, 1.0 changes 4.**
  The upload is therefore identical to T. At the official-like rate of 6.5%, about 5 changes were expected, and
  0 has a probability of about 0.005. So on the official windows the adaptation delta is too small relative
  to T's margins.
- Lambda values of 0.5 and above were negative locally, so there is no variant worth a submission.

## Next-best candidates after V (2026-09-17): neither is worth a submission

- **U, the mean of 3 adapted seeds (s20/s21/s22).** Official-like onset:
  - CNN alone: family +0.0210 (validation), +0.0173 (private).
  - T-blend: **-0.0026 / -0.0109**, private window CI [-0.0255, -0.0008].
  - s21 and s22 on their own are worse than s20 in both months, alone and in the blend. Averaging adaptation
    seeds shows the same pattern as tta F: it helps the CNN alone and hurts inside the blend. Rejected.
- **W, 9b with the next-step loss (mask + 1.0 x next, k=1; note 72 D).**
  - Build: `scripts/t2_adapt_official.py --next-w 1.0` -> `work/sub_W_ttt_t2adaptnext_t4ref/submission.csv`.
  - **Only 1 queue row differs from T: private, one onset window.** The public score would be identical.
  - Not submitted.
- Also: `T2CNN_REC` must be unset when running `t2_cnn_blend_predict.py`. The production CNN checkpoints have no
  rec head, so loading them fails if it is set.

## LB: U = 0.87641 (submission 56285293, 2026-09-17), -0.00137 against T (0.87778)

- The submission was made on the user's request; the local T-blend had already come out negative
  (-0.0026 validation / -0.0109 private).
- U changes only 5 validation queue rows (4 onset windows) from T, yet the public score dropped by 0.00137.
- **The negative local estimate was right in sign**: averaging the adaptation seeds hurts inside the blend. On the
  local-to-LB calibration, this is the first negative onset prediction that the board confirmed.
- T remains the best.

## LB: W = 0.87778 (submission 56299671, 2026-09-17), identical to T

- W was submitted on the user's request ("submit one of the built but unsubmitted candidates").
- **S was excluded on purpose**: its t1res features come from the whole-day Task 1 reconstruction, which on the
  official windows reaches the post-buffer route.
- V is identical to T.
- As expected, the public score equals T: W differs only in one private cell.

## Incident / supply-side hypothesis: closed after a 10-minute check (2026-09-17)

- The release documents say it directly (`trafficflowbench-public/docs/DATA.md`): validation and private have
  "their own incident schedules: **five incidents in validation, seven in private**, at different times and
  places".
- `config/synthetic_release_v1.json` holds only `incident_severity: 1.0` and
  `capacity_multiplier: 1.2`. It gives no time or location rule, and the scenario seed is held by the organizers.
- **Upper bound.** There are 5 and 7 incidents per month over all panels and all hours. T misses 34 onset
  windows (17 per month over 8 panels), and an incident only matters if it lands on the right panel, day and
  morning window. So incidents can explain at most 5+7 of the 34 misses, and realistically a few.
  For the public LB, 5 incidents against 40 official onset windows is a small overlap.
- Even where an incident is the cause, its time and place are drawn from an unpublished seed. It is Bayes error
  for a history-only model unless there is a pre-T signature, which a few events cannot establish either.
- **Not pursued.** The incident oracle and the hidden-capacity filter were not built.

## Occupancy / dynamic ramp flow / local accumulation as the selector among T's top 3: closed (5-minute check)

- Setup: official-like onset windows whose truth lies in T's top 3 NMS peaks. For each window, the candidate with
  the largest score within +-1 link is picked, using the last 15 minutes of history.
- Score definitions:
  - `occ`: mean occupancy over the last 15 minutes.
  - `d_occ`: that mean minus the mean of the 15 minutes before.
  - `r_on/cap`: on-ramp flow over capacity.
  - `d_r_on`: the change in on-ramp flow over capacity.
  - `accum/cap`: (q_up + r_on - q_down - r_off) over capacity.

| pick correct among the top 3 | T | occ | d_occ | r_on/cap | d_r_on | accum/cap |
|---|---|---|---|---|---|---|
| T-missed windows (n=29) | 0.17 | 0.34 | 0.45 | 0.48 | 0.24 | 0.41 |
| other windows (n=297) | 0.94 | 0.54 | 0.34 | 0.38 | 0.40 | 0.41 |

- Every new signal sits at about the chance level of 2-3 candidates (0.35-0.5) in **both** groups.
- On the missed windows they only look better than T because T's pick is wrong there by construction.
- A usable selector would have to beat chance on the windows T gets right as well, and none does. They are not
  a switching signal. This is consistent with note 41, where history ramp flows as LGBM features made onset
  worse by 0.03-0.07. **Not pursued.**

## Delta-core hedge and the official selector audit: both closed (2026-09-17)

**Selector audit.**
- The official train onset windows (40, with labels from the true layer) put the first queue **at T+30 in 39 of
  40 windows**, with zero queued cells at T+5..T+25. Ongoing windows have queues from T+5.
- This matches our pseudo-window definition (`first_onsets`: the first queue at T+30), our training windows
  (`ctx.eo`) and the T+30-only onset prediction. There is no selector mismatch to fix.
- The 5-windows-per-panel / 360-minute spacing only thins the sample.

**Delta-core hedge**, i.e. Q_T union {the delta-preferred alternate peak, 1-2 cells at T+30}:
- Any cell at T+20/T+25 is a guaranteed false positive, because onset horizons are empty there.
- The alternate peak was taken either as the top delta~ peak away from T's pick, or as the top-3 candidate of T
  with the largest delta~.

| variant | IoU recovered on T-missed windows (sum) | IoU lost on the other windows (sum) | onset family |
|---|---|---|---|
| alternate from all peaks, +1 cell | +1.2 (val) / +1.4 (priv) | -45.9 / -45.9 | -0.26 / -0.27 |
| alternate from T's top 3, +1 cell | +3.1 / +1.4 | -43.2 / -43.5 | -0.23 / -0.26 |
| +2 cells | slightly more recovered | about -50 to -55 | -0.28 to -0.33 |

- Onset truth sets are only 1-5 links, so one extra wrong cell costs about 0.3 IoU on a correct window.
- A hedge would pay only if it were applied where T misses with precision of about 0.65 or better. No such miss
  detector exists: the failures are confident ones (note 55). **Closed.**

## Small re-tunes of T (2026-09-17): the onset weight and ratio c move; ongoing tau does not

- Grid on the official-like windows, cached probabilities only.
- **Ongoing tau**: 0.45 gives -0.0006/-0.0022 and 0.55 gives -0.0036/-0.0022, so the current 0.50 is optimal.
  No change.
- **Onset**: w_on (the LightGBM share) against ratio c, family delta vs T (w_on 0.7, c 1.0), validation / private:

| w_on \ c | 0.8 | 0.85 | 0.9 | 1.0 |
|---|---|---|---|---|
| 0.55 | +0.017 / -0.004 | +0.018 / -0.006 | +0.017 / -0.009 | +0.010 / -0.035 |
| **0.6** | +0.012 / +0.024 | **+0.010 / +0.021** | +0.011 / +0.022 | +0.016 / -0.001 |
| 0.65 | -0.004 / +0.022 | -0.002 / +0.020 | +0.001 / +0.002 | +0.004 / +0.003 |
| 0.5 / 0.4 | | | | clearly negative on private |

- w_on 0.6 with c 0.8-0.9 is a small plateau that is positive in both months.
- Window CIs at w_on 0.6, c 0.85: +0.014 [-0.003, +0.033] on validation and +0.004 [-0.009, +0.016] on private.
- **Caveat**: the setting was picked by looking at both months, and the surface is jagged (w_on 0.55 is negative on
  private). Treat it as optimistic.
- Expected gain: total about +0.0025 locally, about +0.0013 after halving.
- It fits the delta-correction reading: the adapted CNN was under-weighted.
- **Candidate X** = T with w_on 0.6 and c 0.85:
  - `scripts/t2_cnn_blend_predict.py --ratio-c` was added for this.
  - Upload: `work/sub_X_ttt_t2won06_t4ref/submission.csv`.
  - 9 queue rows differ from T: 3 validation and 4 private onset windows.

## LB: X = 0.87609 (submission 56300149, 2026-09-17), -0.00169 against T

- Locally +0.010 validation family (about +0.0015 total), yet the public score fell. X changes only 3 validation
  onset windows (15 official onset windows per... 40 in total), so this is 3 windows' worth of noise, and it
  went the wrong way.
- Together with U (-0.00137), this is the second onset-side re-tune that lost on the board. The note 58 rule
  holds again: **tuning the onset decision on 169 windows does not transfer**, even when both months look
  positive. Only 9b (a weight change of the model, not a decision change) ever transferred.
- T stays the best and the final choice. Onset decision re-tuning is closed.

## Candidate Y (2026-09-17): T with the Task 1 TTT extended to 1200 steps

- Build: `scripts/run_candidate_Y.sh` -> `work/sub_Y_ttt1200_t2adapt_t4ref/submission.csv`. Took about 20 minutes.
- 6,733,999 state rows differ from T. Queue and odme rows are identical.
- Expected (note 32 appendix, validation pseudo T1/T3): total +0.0013 against +0.0010 at 400 steps, so
  **+0.0003 over T**.
- This is a Task 1/3 change, not an onset decision change, and the validation measurement covered the whole
  month, not 169 windows.

## LB: Y = 0.87845 (submission 56300633, 2026-09-17), a new best, +0.00067 against T

- The local estimate was +0.0003. The LB gain is about twice that, so the Task 1/3 side transfers fully.
- **Y is the new best and the final-submission choice.** `run_candidate_Y.sh` rebuilds it.
- Follow-up: 3600 steps were flat locally (+0.0013, the same as 1200), so there is nothing more to take on the
  step count.

## LB: Z = 0.87871 (submission 56330720, 2026-09-18), a new best, +0.00026 over Y and +0.00093 over T

- Z is T with the Task 1 TTT at 3600 steps. Build: `scripts/run_candidate_Z.sh`, about 40 minutes.
- Locally, 3600 steps looked the same as 1200 (+0.0013 total), yet the LB still rose: 400 -> 1200 -> 3600 steps
  gave 0.87778 -> 0.87845 -> 0.87871.
- The local estimate here is a 20%-hidden pseudo T1/T3 on the validation month, and it under-reads the TTT
  gain. More steps may still pay; each 3x costs about 3x the time (about 2 hours for 10,800 steps).
- **Z is the new best and the final-submission choice.**
