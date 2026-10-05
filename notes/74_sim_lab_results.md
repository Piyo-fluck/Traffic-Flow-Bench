# 74 - sim_lab (plan 1, synthetic-run pretraining): the CTM branch failed calibration; empirical day-bootstrap is ambiguous

Source: Ada `sim_lab/reports/00-08*.md`, run by the user (2026-09-19). Kit: `sim_lab/SIM_EXPERIMENT.md`.

## CTM synthetic runs: stopped at the calibration gate

Eight approaches were tried, all on the public train month only. None passed on all 8 panels at the same time.
The gate asks for queued share within 0.5-2x of train, map correlation >= 0.5, top-8 overlap >= 3 and run
diversity 0.3-0.8.

- The eight approaches: discharge capacity, fd capacity, link-wise capacity correction (1 and 2 rounds), stronger
  run perturbation, a panel-level share correction, bootstrap of daily demand, an on-ramp capacity drop, and a
  differentiable inverse CTM (plus transferring its profile and combining it with the bootstrap).
- The inverse CTM helped most. D7_I10_E map correlation went 0.39 -> 0.66, D7_I405_S 0.07 -> 0.71.
- Structural failures remain:
  - I405_N loses its queues (0.097 -> 0.001-0.018).
  - The rare queues of I10_W cannot be placed.
  - Several panels have run diversity stuck at 0.8-1.0.
- The observed upstream flow is not exogenous demand; it is already capped by congestion. The report concludes
  that a state-space demand identification or a merge / capacity-drop / spillback model would be needed.
  **CTM pretraining was not started.**

## Empirical day-bootstrap mix (reports 07-08)

- Each "run" copies one or more whole public-train days (sampled without replacement).
- The model trains on real and empirical windows at 50:50, for 8k steps.
- Results against an 8k-step real-only control, onset T-blend window IoU, validation / private:
  - control .7955 / .8139
  - mix, seed 81: .8077 / .8345 (+0.012 / +0.021)
  - mix, seed 82: .8087 / .8429
- Candidate T (the production rec_s20 + 9b) is **.8181 / .8516**, and no variant beats it. Averaging the two
  seeds is worse than seed 82 alone.

## Reading (mine)

- The empirical runs are **the same train days again**, so they add no new run.
- The gain over the control more likely comes from the sampling change, not from run diversity:
  - Each empirical 1-day run keeps all its onset windows but only 500 ongoing ones, so the onset share goes up.
  - The mix also re-weights days.
  - The control (8k steps) is far shorter than the production 30k.
- **Possible early-stopping contamination.** Early stopping uses the train days of 2031-01/02. If the empirical
  pool can draw those days, the model trains on them, and the ES BCE (.0295, the best) and the checkpoint choice
  become biased. This is not a scored-month leak, but it invalidates the comparison.
- Checks before going further:
  1. Remove the ES days from the empirical pool.
  2. Add a control with the same onset/ongoing sampling ratio and no empirical runs.
  3. Compare at the production length (30k steps) against rec_s20, i.e. what T actually uses.

## T combined with the empirical-mix models: worse everywhere (2026-09-19)

- Script: Ada `tta_lab/_tmp/combo_T_mix.py`. Scope: official-like windows. Family delta vs T, validation / private.
- **Onset** (the 0.3 CNN slot of the T-blend):

| CNN slot | validation | private |
|---|---|---|
| (ad9b + mix82) / 2 | -0.0098 | -0.0017 |
| ad9b + mix81 + mix82, equal weights | -0.0103 | -0.0033 |
| 2/3 ad9b + 1/3 mix82 | -0.0090 | -0.0026 |
| mix82 alone | -0.0204 | -0.0016 |

  All window CIs are negative or around 0.
- **Ongoing** (the 0.2 cnn3 slot): cnn3 + mix gives -0.0004 to -0.0016, also worse.
- Same pattern as U and tta F: averaging a second CNN into T's CNN slot hurts the blend. The mix models add no
  information that T lacks. Branch closed.

## Candidate AA (2026-09-19): Z plus the Task 3 conservation loss in the Task 1 TTT (plan 3)

- `scripts/t1_ttt_train_dn.py` is the production TTT plus `TTT_DN` x `m3.dn_loss`, the |dN error| / |dN true|
  S_LWR proxy.
  - The link length rides along as an 11th data channel; the networks still get the 10 v1 channels.
  - Settings: TTT_DN = 0.05 and 3600 steps, as in Z.
- Build: `scripts/run_candidate_AA.sh` -> `work/sub_AA_ttt3600dn_t2adapt_t4ref/submission.csv`, about 40 minutes.
  Only state rows differ from T (6,734,646). Queue and odme are identical.
- Prior evidence (note 05, the loss in training rather than TTT): net 0. The local TTT metric under-reads, so the
  LB decides.

## LB: AA = 0.87858 (submission 56345276, 2026-09-19), -0.00013 against Z (0.87871)

- The conservation loss in the TTT does not help. It is slightly negative, which matches note 05 (net 0 or
  slightly worse).
- Plan 3 is closed. **Z stays the best and the final choice.**

## LB: AB = 0.87764 (submission 56349112, 2026-09-19), -0.00107 against Z

- AB is Z with the onset ratio-rule c = 1.30. Ada sim_lab report 22 chose c on the train time-series OOF, not on
  the scored months.
- Locally the onset family gain was +0.0106 (validation) / +0.0141 (private). AB changes 4 validation onset
  windows (8 links added in total).
- **Third onset decision-layer change to lose on the public board** (U -0.00137, X -0.00169, AB -0.00107), even
  with out-of-sample selection.
- The public LB scores only 5 onset windows per panel, and here 4 of the 40 validation onset windows changed.
  Adding links costs about 0.3 IoU per wrong link on 1-5 link truths (the delta-core hedge result), and the
  official windows punish that.
- Z stays the best.
