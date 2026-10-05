# 169 Generating runs with the organizers' physics (DLSim4GMNS): structure found, replication in progress

Date 2026-09-27. Code `analysis/dlsim_gen.py` (v1, released ramps), `analysis/dlsim_gen2.py` (v2/v3, OD zones).
Package: DLSim4GMNS 0.1.0 (Apache-2.0, pip; native C++ merge loader, ~0.25 s per simulated day and panel).

## The generator's structure, as far as the release shows it

- DLSim4GMNS: Newell cumulative-count loading, triangular FD, lane-proportional merge of the downstream inflow
  capacity, diverges with a fixed off_frac and no off-ramp spillback, incidents = capacity (qmax) drops, 6-s ticks.
- The released ramps are synthetic (note 168 correction): they do not balance the mainline flows.
- The real entries/exits are the OD zones of path_set.csv. The longest path is the full corridor (257-428 links);
  the panel's sensored links lie on it in order (1.0-4.2 corridor links per gap), and every zone is an interchange
  with an entry and an exit in the same gap (72-100 zones).
- Travel direction from path_set: z index increases downstream on D7_I10_E, D7_I210_E, D7_I405_N, D12_I5_N and
  decreases on the other four. The queue-end movement test is inconclusive (both ends move about equally), so this
  does not settle the upstream convention of note 159 / hidden_onset_recover.UP_HIGHER either way.

## Replication of train days (queued share sim / true, map corr, daily space-time IoU; 12 days)

| generator | D12_I5_S | D12_I5_N | D7_I405_N |
|---|---|---|---|
| v1: released ramps as demand, fd caps | 0.301/0.082, 0.31, 0.180 | - | - |
| v2: zones, the day's flows as demand, fd caps | 0.009/0.082, 0.34, 0.076 | 0.006/0.058, 0.26, 0.050 | - |
| v3: zones, day flows, caps = in-queue discharge p50 | **0.047/0.082, 0.66, 0.252** | 0.026/0.058, 0.15, 0.139 | 0.010/0.127, 0.12, 0.033 |

Free-flow flows along the corridor are reproduced closely (exit fractions from the weak prior are fine). The
failure is capacity: queues in the truth sit at flows well below the released capacity (D7_I405_N: v 22 at
4,346 vph on links with 7,675 vph), so the effective bottleneck capacities are not the released FD values
("FD capacity parameters" are listed as replaced_with_synthetic). Next: fit the bottleneck capacities by
matching simulated to true queue-head occupancy on train (iterative, the simulator is fast).

## Capacity fit on train (analysis/dlsim_fitcap.py) - D12_I5_S

Damped head-occupancy + queued-share updates, 40 rounds on 30 fit days, best round by fit-day IoU, evaluated on
20 held-out train days with each day's own flows as demand: queued share 0.109 / true 0.091, map corr **0.81**,
daily space-time IoU **0.414**, against 0.269 for "the same weekday a week later" (the run's day-to-day variation).

## Side test: CBI/QVDF arrival-vs-service crossing as onset candidate evidence (occ_lab/cbi_crossing_eda.py)

Per set-decoder candidate: arrival = upstream sensored neighbour's flow, quadratic extrapolation to T+30; service
mu = fitted effective capacity (above) or the released capacity. Pairwise AUC best candidate vs others, scored-month
lab windows, D12_I5_S + D7_I405_N (280 pairs):

| feature | fitted mu | released cap |
|---|---|---|
| E(10 min) cumulative excess | 0.671 | 0.808 |
| E(20) / E(30) | 0.657 / 0.614 | 0.783 / 0.744 |
| first crossing | 0.530 | 0.528 |
| max excess / mu | 0.586 | 0.673 |
| mu / C | 0.604 | 0.388 |
| production blend mass | 0.961 | |
| random placebo | 0.474 | |

The released-capacity version is the stronger one and is essentially note 145's headroom (already in BB's decoder);
the fitted service rate is worse (the per-link fit is not identified well enough at candidate level). Far below
the production evidence. Not pursued.

## All panels: capacity fit (held-out train days, own-day demand)

| panel | map corr | daily IoU | reference (same weekday next week) |
|---|---|---|---|
| D7_I210_E | 1.00 | 0.753 | 0.604 |
| D12_I5_S | 0.81 | 0.414 | 0.269 |
| D7_I405_S | 0.79 | 0.377 | 0.250 |
| D12_I5_N | 0.85 | 0.310 | 0.163 |
| D7_I10_E | 0.90 | 0.298 | 0.253 |
| D7_I210_W | 0.45 | 0.269 | 0.537 |
| D7_I405_N | 0.50 | 0.193 | 0.299 |
| D7_I10_W | 0.38 | 0.038 | 0.075 |

## Do demand differences reproduce run differences? (analysis/dlsim_runshift.py; evaluation use)

Train-fitted capacities; demand from each run's observed flows. Correlation of the CHANGES of the per-link queued
share (sim scored-demand - sim train-demand vs true scored - true train): positive in 12 of 16 panel-runs, median
about +0.5 (D12_I5_S val +0.74, D12_I5_N +0.60/+0.60, D7_I405_N +0.53/+0.66, D7_I210_W +0.41/+0.67, D7_I210_E
val +0.93); about 0 or negative on D7_I10_E (-0.20/+0.01) and D7_I405_S private (-0.01). Demand redraws explain a
good part of how bottleneck maps shift between runs; the rest is run-level capacity / incidents.

## How demand differs between runs (analysis/dlsim_demand_draws.py)

Per entry (OD-zone gap), log(run / train) of the 06-20 h free-flow demand: sd 0.025-0.134 by panel, lag-1
autocorrelation along the corridor about 0 (I210_E +0.55, I5_N +0.2), panel mean within +-4 %, upstream boundary
within +-2.5 %, and corr(val ratio, pri ratio) +0.29..+0.77 (mean about 0.5) - exactly what independent per-zone
draws around a common base give when both runs are compared with the same train draw. Generator for a new run:
train-day demand x exp(sigma_panel * eta_zone), eta iid N(0,1), plus a panel scale and incidents.

## Generated runs (analysis/dlsim_make_runs.py) and the first training test

8 runs x 60 days per panel (work/genruns, 855 MB; per-entry demand factor exp(sigma N), sigma = the panel's
run/train log sd; panel scale 2 %, upstream 1.5 %; Poisson incidents). Map corr between generated runs / with the
train truth: I10_E 0.98 / 0.88, I10_W 0.58 / 0.43, I210_E 1.00 / 1.00, I210_W 0.95 / 0.44, I405_N 0.80 / 0.54,
I405_S 0.61 / 0.63, I5_N 0.96 / 0.83, I5_S 0.91 / 0.72 (real runs vs train: 0.61-1.00). The sim-real gap is large on
I10_W, I210_W, I405_N.

Test (Ada, occ_lab/run_gen.sh): leap_t2 whole-panel fine-tune of the SSL w192 T2 CNN, 6k steps, with generated
parts drawn for half of the batches (occ_lab/genparts.py, leap_t2 --gen), seeds 20/21, then the lab stage (daily
pseudo-supervised adaptation). Control = leap ctrl_s20/s21 (same recipe, real only):

| ctrl (sup+rec) | onset val / pri | ongoing val / pri |
|---|---|---|
| s20 | 0.7986 / 0.7371 | 0.8562 / 0.8293 |
| s21 | 0.7943 / 0.7492 | 0.8572 / 0.8368 |

Result, round 1 (gen-frac 0.5 throughout), sup+rec:

| | onset val / pri | ongoing val / pri |
|---|---|---|
| gen s20 | 0.7950 / 0.8248 | 0.8537 / 0.8274 |
| gen s21 | 0.7823 / 0.7612 | 0.8498 / 0.8330 |
| mean delta vs ctrl | -0.008 / +0.050 | -0.005 / -0.003 |

Without adaptation onset drops hard (0.72-0.74 -> 0.57-0.66): the sim-real gap. Per panel
(occ_lab/gen_panel_breakdown.py) the private onset gain is D7_I10_W private (3 windows, 0.000 -> 0.667) and
D7_I405_S private (0.484 -> 0.670 for s20 only); elsewhere mixed or negative (D12_I5_N -0.03..-0.06 in all four,
D7_I405_N s20 -0.06), ongoing slightly down almost everywhere. Read: no effect to slightly negative.
Round 2 running: generated share decaying 0.5 -> 0 (tags gdec), and the same without the three poorly generated
panels' runs (gdecg).

Round 2 (occ_lab/gen_panel_breakdown2.py; onset sup+rec, mean of s20/s21; window-weighted / family mean without
D7_I10_W, whose private has 3 onset windows and produced the apparent +0.04..+0.09):

| | val | pri |
|---|---|---|
| ctrl | 0.8467 / 0.8715 | 0.8295 / 0.8414 |
| gen (0.5 throughout) | 0.8270 / 0.8451 | 0.8304 / 0.8329 |
| gdec (0.5 -> 0) | 0.8319 / 0.8538 | 0.8285 / 0.8325 |
| gdecg (0.5 -> 0, good panels only) | 0.8477 / 0.8681 | 0.8408 / 0.8471 |

Ongoing sup+rec: -0.002..-0.005 in every variant. Conclusion: at this fidelity, mixing generated runs into the
training does not help (best variant: val +0.001 / -0.003, pri +0.011 / +0.006). The generator itself holds up
(capacity fit beats the day-to-day reference on 5 panels; demand swaps reproduce run shifts in 12 / 16).

## Root causes of the poor fit (2026-09-28, after the user's direction to find causes rather than move on)

1. **The truth lies exactly on the released FD.** Congested cells: (q - released congested branch) / cap median
   0.00, IQR +-0.02..0.03 on every panel; free-flow v / vf 0.997-1.000. The released FD IS the generator's FD.
2. **Receiving capacity** (analysis/dlsim_breakdown_anatomy.py, ~3,000 train breakdown events): the head h of a new
   queue discharges at 0.96-0.99 x the capacity of the next DOWNSTREAM link (0.58-0.78 x its own; cap_down /
   cap_h ~0.8), pre-breakdown flow 0.85-0.96 x cap_down, no capacity drop. A Godunov/CTM node: inflow is capped by
   the receiving link's capacity, the queue sits upstream. DLSim4GMNS's Newell link model caps inflow only by
   storage, so its queue sits inside the bottleneck link. The v3 per-link capacity fit (caps down to 0.35-0.9 x)
   was absorbing this and the demand error. -> v5 (analysis/dlsim_gen4.py): a short gate link in front of every
   corridor link with that link's capacity; entries merge into the gate.
3. **Demand is censored in queues.** Observed flows cannot exceed capacity; the excess sits in storage (and in
   entry queues off-network). v5 demand: conservation (q_b - (1 - f) q_a + dN/dt, N = k L with k = q / v exact on
   the FD) and max with the weekday free-flow profile x the day's level.
4. **Unsensored links** take the DOWNSTREAM sensor's FD (the capacity change sits right after the true head).

v5 with released capacities and no fitting, held-out train days (share sim / true, map corr, daily IoU):
D12_I5_S 0.075/0.083, 0.26, 0.178; D7_I210_W 0.033/0.084, 0.30, 0.165; D7_I405_N 0.038/0.155, 0.19, 0.119;
D7_I10_W 0.026/0.007, -0.01, 0.018; D7_I10_E 0.027/0.042, 0.10, 0.073; D7_I210_E 0.000/0.011, 0.45, 0.022;
D7_I405_S 0.039/0.082, 0.08, 0.131; D12_I5_N 0.013/0.044, 0.29, 0.090. Main true heads are reproduced
(D12_I5_S 56/84/87, D7_I210_W 5/47/74); durations too long at those, spurious heads elsewhere, small ones missed.
Remaining cause: demand. Next: capacities fixed at the released values, per-entry x time-of-day demand
multipliers fitted by simulation (dynamic ODME, as DLSim's calibrate_odme).

## Further root causes (same day)

5. **Newell link model artifacts.** With the capacities raised 1.3x (static) half of v5's congestion stays: the
   storage-based inflow on short links (<= 0.2 km at 6-s ticks; bwtt rounding) creates queues unrelated to
   capacity. 2-s ticks reduce it (free-flow v/vf p10 0.854 -> 0.927) but do not remove it.
6. **Two capacities, tested wrongly first.** The truth carries flow above the released capacity without breaking
   down (D12_I5_S link 7: p99 1.025, max 1.089 x cap; link 57: p99 1.039) while a queue behind a head discharges
   at 0.96-0.99 x the downstream capacity - pre-breakdown capacity above the discharge rate. A whole-day fixed
   point of "drop while congested" is ill-posed (the drop creates the congestion that justifies it) and showed no
   effect; the drop has to be decided causally inside the simulation.
7. **Unsensored gap length is not the driver.** Tail-propagation estimates (analysis/dlsim_gap_length.py) are too
   noisy at 5-min resolution (sd / median 1.1-1.4) and suggest short connectors; scanning the assumed length 0.1-1.0x
   changes nothing.
-> analysis/ctm_gen.py: a Godunov CTM on the full corridor (numba, 2-s steps, ~0.1 s per day), sending
   min(vf k, a cap), receiving min(b cap, w (kj - k)) with a causal capacity drop, lane-proportional merges, free
   exits, origin queues. Released FD, no fitting, held-out train days: D12_I5_S share 0.055 / 0.083, map corr 0.41,
   daily IoU 0.232 (DLSim v5 0.178); D7_I210_W 0.174; D7_I405_N 0.076; D12_I5_N 0.070. Pre-breakdown factor 1.06
   does not help yet. Congestion is still too low everywhere: the censored demand. Demand multipliers are being
   fitted with the CTM (objective: IoU minus the share gap, so that overlap is not bought with extra queue).

## CTM generator with fitted censored demand (analysis/ctm_gen.py + dlsim_fitdemand.py CTM=1)

Released FD, per-entry x 5-block multipliers on the censored demand fill (objective IoU - |share gap|), 60 fit /
30 held-out train days. Held-out daily IoU: D7_I210_E 0.744, D12_I5_S 0.396, D12_I5_N 0.385 (best so far),
D7_I405_N 0.353, D7_I210_W 0.292, D7_I405_S 0.243, D7_I10_E 0.184 (too much queue, 0.116 / 0.037),
D7_I10_W 0.137 (best so far). Comparable to the v3 capacity fit with far fewer free parameters.

Run-shift test (dlsim_runshift.py CTM=1), corr of the map changes val / pri: D12_I5_S +0.83 / +0.56,
D7_I405_N +0.75 / +0.83, D12_I5_N +0.52 / +0.67, D7_I210_W +0.40 / +0.69, D7_I405_S +0.52 / +0.43,
D7_I10_E +0.81 / -0.30, D7_I210_E +0.93 / +0.08, D7_I10_W +0.11 / +0.13: positive in 15 / 16, median about +0.54
(v3 ~+0.50, v5 ~+0.48). The demand swap moves the simulated map toward the true run in 13 / 16.

Run diversity: with the v2-measured sigma the CTM runs are too alike (map corr between runs 0.93-0.99 vs 0.54-0.97
for real runs); per-panel draw scale (I5_S 2.0, I405_S 1.0, I5_N / I10_E 2.5, others 1.5). I5_N and I10_E stay at
0.92-0.97 even at 2.5x: their run differences come from something other than entry demand (capacity / incidents),
consistent with the negative I10_E private run-shift. Generated: work/genruns_ctm, 12 runs x 60 days per panel.
Pretraining re-run on Ada (tags gctm_s20/s21, generated share 0.5 -> 0).

## CTM runs in pretraining (tags gctm): worse; the cause is the onset precursor, not the maps

Onset sup+rec, s20/s21 mean, window-weighted / family w/o D7_I10_W: val 0.8423 / 0.8672 (ctrl 0.8467 / 0.8715),
pri **0.8125 / 0.8211 (ctrl 0.8295 / 0.8414)**; ongoing flat to slightly down. Better maps did not help.

analysis/gen_precursor_gap.py, earliest-onset windows, real train truth vs CTM runs: AUC of q/cap (last 3 history
slots) for picking the links queued at T+30, and their median q/cap:
D12_I5_N 0.20 -> 0.63 (0.46 -> 0.70), D7_I10_E 0.65 -> 0.85, D7_I210_E 0.86 -> 0.98, D7_I10_W 0.85 -> 0.67,
D7_I405_N 0.63 -> 0.59 (0.70 -> 0.49), D12_I5_S 0.61 -> 0.78, D7_I405_S 0.42 -> 0.42, D7_I210_W 0.78 -> 0.69.
In the CTM runs the onset link is, much more often than in the truth, the link loaded closest to its capacity (a
deterministic breakdown where demand crosses capacity); in the truth onsets often appear at links with LOW q/cap
(D12_I5_N AUC 0.20). Pretraining on the generated runs teaches that precursor, which the truth does not follow.
Next: the anatomy of where the truth first shows a queue relative to the bottleneck (unsensored bottlenecks
upstream-shifting the first observed queue, or link-specific breakdown capacities).

## The deeper cause: hidden (unsensored) bottlenecks

analysis/onset_precursor_anatomy.py: in the truth the onset links are picked by q / cap_down (next sensored link)
only at AUC 0.46-0.88 (D12_I5_N 0.46, D7_I405_S 0.52), in the CTM runs at 0.92-1.00: the CTM breaks down exactly
where the sensored load crosses a sensored capacity, the truth does not.
analysis/discharge_variability.py: day-to-day CV of the discharge at recurrent heads is small (0.6-3 %, D7_I10_W
6.7 %) - no random daily capacity - but its LEVEL is 0.64-0.79 x the next sensored link's capacity on six panels
(0.99 on D7_I210_E and D7_I405_N; CTM 1.01). Recurrent bottlenecks bind at capacities that sit in the unsensored
gaps, ~0.7 x the released sensored values. That is what the v3 capacity fit was approximating, and why the truth's
onsets look non-deterministic in sensored q/cap.
-> CTMGen._hidden_bottlenecks: for each head with >= 5 queue-days, the median daily discharge (a direct measurement
of capacity) becomes the capacity of the first cell after the head. No occupancy fitting. Held-out daily IoU
(off -> on): D7_I210_W 0.163 -> 0.434, D7_I405_S 0.098 -> 0.317, D12_I5_S 0.245 -> 0.365, D12_I5_N 0.104 -> 0.134,
D7_I10_E 0.086 -> 0.067 (too much queue), D7_I405_N none found.

With hidden bottlenecks + the censored-demand fit (CTM): held-out daily IoU D7_I210_W 0.468, D12_I5_N 0.332,
D7_I405_S 0.317, D7_I10_W 0.158 (best so far on all four), D12_I5_S 0.388, D7_I405_N 0.353; worse on D7_I10_E
(0.153) and D7_I210_E (0.652). Run shift: positive in 14 / 16, median about +0.56. **Onset precursor (AUC of
q / cap_down, truth / generated): D7_I405_S 0.52 / 0.45 (was 0.96), D12_I5_N 0.46 / 0.36 (0.96), D12_I5_S
0.80 / 0.67 (0.95), D7_I210_W 0.86 / 0.90** - the deterministic-breakdown artefact is gone there; D7_I10_E breaks
(0.07), D7_I405_N / D7_I210_E unchanged (0.97 / 1.00). Hidden bottlenecks kept on 5 panels (off on D7_I10_E and
D7_I210_E, chosen by train replication). Pretraining re-run: tags ghb, generated runs of D7_I10_E and D7_I10_W
excluded (precursor still far off).

## Pretraining with the hidden-bottleneck CTM runs (tags ghb): first positive in both months

Onset sup+rec, s20/s21 mean, window-weighted / family w/o D7_I10_W: val 0.8499 / 0.8737 (ctrl 0.8467 / 0.8715),
pri 0.8382 / 0.8477 (ctrl 0.8295 / 0.8414) -> +0.003 / +0.002, +0.009 / +0.006. The same CTM without hidden
bottlenecks (gctm) was -0.017 / -0.020 on private: fixing the onset precursor flipped the sign, which confirms the
diagnosis. Small and panel-mixed: D7_I405_S pri 0.519 -> 0.654, D12_I5_S val 0.752 -> 0.816, D7_I210_W pri +0.04;
D12_I5_S pri 0.816 -> 0.708, D7_I405_N pri -0.04, D12_I5_N -0.02. Ongoing -0.002..-0.003 (as before).
Open: the effect as an onset member (production onset = H/STID/ad9b blend + set decoder), and the panels whose
precursor still differs (D7_I405_N 0.97 vs 0.86, D7_I210_E, D7_I10_E).

## Why ghb loses on D12_I5_S private, D7_I405_N private, D12_I5_N (analysis/hidden_bn_across_runs.py)

Hidden-bottleneck capacities are identical across runs (discharge levels equal to two decimals in train / val /
pri). What changes between runs is WHICH bottlenecks activate: D12_I5_N head 114 (0.60 x cap) queues on 198 / 273
train days and on 0 days in both validation and private; D12_I5_S head 26: 195 train days, private 2; heads 70-79:
82-115 train days, validation 0; D7_I405_N (no hidden bottlenecks) heads 97-110 and 232/234: 57-130 train days,
private 0. The generator draws runs as train demand x independent noise, so every generated run keeps train's
active bottlenecks (I5_N generated runs correlate 0.96) and pretraining reinforces them. Real runs are base demand
x their own draw, and train is one of the draws: both scored runs deviate from train in the same direction
(corr of the ratios +0.3..+0.8), i.e. train's own draw is the outlier where it activates a bottleneck the other
runs do not. Fix needs train's own deviation from the base demand; open question (user decision) whether the
per-split Task 4 weak priors may be used for that.

## Weak priors do not carry the run's demand draw (analysis/weak_prior_vs_demand.py, evaluation only)

corr(log prior ratio, log measured entry-demand ratio), run / train, PM: -0.28..+0.15 on every panel; the train
prior's level even anti-correlates with the measured train demand (-0.54..-0.02). The priors' own val and pri ratios
correlate (+0.34..+0.62) through the shared train prior - independent noise per split. Not usable.

## Why the generated runs keep train's bottlenecks (analysis/head_activation_freq.py)

Share of days each recurrent head is active, truth vs generated: D12_I5_N 105 0.80 vs 0.99, 114 0.73 vs 0.93, and
spurious 156 0.02 vs 1.00, 64 0.02 vs 0.45, 187 0.02 vs 0.49; D12_I5_S 26 0.71 vs 1.00, 70 0.42 vs 1.00, 14 0.75
vs 1.00 (+5 spurious); D7_I405_N main heads match (0.22-0.48 vs 0.10-0.41) but 8 spurious heads. Two causes:
(1) hidden bottlenecks identified from heads with only 5-8 queue-days - most likely incident locations (5-7 per run
per the docs) - turned into permanent capacities; (2) the demand fit (IoU objective) pushes marginal bottlenecks
(true activation 0.4-0.8) to always active, so demand draws cannot switch them off as between real runs.
Fixes: permanent hidden bottlenecks only where active on >= 15 % of days (rarer ones become random incidents), and
fit demand to each head's day-activation share.

## Correction: the per-panel ups and downs are mostly noise (occ_lab/panel_noise.py on Ada)

Head over-activation and spurious (incident-like) heads exist on EVERY panel, including those that improved
(D7_I405_S: 63 0.74 -> 0.95, spurious 74 0.04 -> 0.82; D7_I210_W: 77 0.64 -> 1.00, spurious 54 0.03 -> 0.99).
The overlap between generated and scored-run activation does not explain the per-panel gain (corr -0.29..-0.41,
n 14; D12_I5_S val +0.064 and pri -0.108 at the same overlap). Noise: the two control seeds differ per panel by up
to +-0.08..0.11; window-level SE of ghb - ctrl per panel 0.014-0.061. Only D7_I405_S pri (+0.134, t 2.9), D12_I5_S
val (+0.063, t 2.2) and D12_I5_S pri (-0.108, t -2.2) exceed |t| 2; the D12_I5_N and D7_I405_N "drops" are
t -0.9..-1.3. All 331 windows: ghb - ctrl +0.0058 (SE 0.0081) - not established. The earlier per-panel explanation
of the drops was read into noise. Generator quality is judged by generator metrics (activation shares, spurious
heads, precursor AUC, run shift); training effects only on all windows with an SE.

## Generator accuracy vs training effect are different questions: the daily adaptation cancels the gain

occ_lab/adapt_overlap.py (Ada), all windows, s20/s21 mean, difference to ctrl (SE):

| | no adaptation (none) | daily adaptation (sup+rec) |
|---|---|---|
| ongoing ghb (CTM + hidden bottlenecks) | **+0.0030 (0.0007)** | **-0.0027 (0.0005)** |
| ongoing gctm (CTM) | +0.0014 (0.0007) | -0.0023 (0.0005) |
| ongoing gdecg (DLSim) | -0.0007 (0.0008) | -0.0030 (0.0005) |
| onset ghb | -0.0028 (0.0074) | +0.0058 (0.0081) |

The generated runs do improve the unadapted model's ongoing, significantly and increasingly with generator
fidelity (DLSim -> CTM -> hidden bottlenecks). The production daily adaptation (pseudo-supervised on the scored
run's past days) gives ctrl +0.041 but the generated-pretrained model only +0.035, so the gain reverses. The
run-invariant knowledge from generated runs and the run-specific knowledge from adaptation overlap and interfere.
Open (user decision): (1) find the interference - adaptation strength / steps / layers for the pretrained model;
(2) meta-learn the adaptation itself across generated runs (Reptile / MAML style) - something only a generator of
many runs allows, since the real data has a single run.

## Onset, analysed on its own (the user's direction: accuracy of the generator and the training effect apart)

Sensitive metrics (work/onset_sensitive.py, onset_hardset.py on Ada; all 331 windows, s20/s21, SE): the generated
runs raise the link-ranking AUC (none: gdecg +0.0091 (0.0023), gctm +0.0059 (0.0021), ghb +0.0027 (0.0019);
sup+rec: +0.0044 / +0.0020 / +0.0030) - the easy part (far links) - but not the ranking among the ctrl model's top-5
candidates (+-0.007-0.009 SE) nor top-1 (swaps fixed / broken 0-8 each). The least accurate generator (DLSim) gives
the largest AUC gain: generator accuracy does not drive it.

Where the onset IoU is lost (work/onset_loss_decomp.py, ctrl sup+rec, IoU 0.871): 30 location misses (9 % of
windows) carry 65 % of the loss (their IoU 0.079); under-sized hits 23 %, over-sized 11 %; 273 windows perfect.
ghb: 27 misses (IoU 0.172), over-sized hits 11 -> 28; mean 0.8792.

The misses (analysis/onset_miss_anatomy.py): 26 / 30 are races between two HABITUAL bottlenecks - the true site
queues on a median 81 % of train days, the model's pick on 84 % - and only 4 are sites rare on train (3 of them
active in generated runs). 11 misses are D7_I405_S private, 7 D7_I10_W. The model misses confidently (p(true) /
p(pick) mostly 0-0.3). The generator's intended contribution (sites unseen in train) covers 4 windows; the bulk is
which habitual bottleneck breaks first today. Next: is that race decidable from the 60-min history?

Race decidability (analysis/onset_race.py): true site vs competitor, load = last-3-slot flow / effective capacity
(hidden-bottleneck capacity where identified), growth = 6-slot slope / capacity. Misses (30): the true site wins on
load in 0.27, on growth 0.33, both 0.17 - the local evidence pointed at the competitor in ~3/4 of them; hits (301):
0.57 / 0.17 / 0.15. The dominant onset loss is not decidable from the candidates' own history, which is why
generated runs cannot fix it through that input. Remaining lever: the demand now upstream that reaches the
candidates within 30 min (a forward CTM from the observed state at T, hidden bottlenecks included) - not yet tested.

## Forward CTM from the observed state at T decides the race (analysis/onset_ctm_forward.py)

Cell densities from the last history slot (k = q / v on the released FD, interpolated along the corridor), demand
for T..T+30 = last-3-slot level x train profile shape (history only), CTM with hidden bottlenecks, 7 slots; score
= simulated density / critical density at T+30. True site ranked above the competitor: misses 0.67 (local load
0.27), hits 0.71. CTM alone top-1 0.302; ctrl p + 0.1 or 0.3 x normalised CTM score: top-1 0.9094 -> 0.9154
(+2 of 331). The upstream arrival over the next 30 minutes, propagated by the right physics, carries race
information the CNN does not use. (The earlier DLSim expert, note 170, started from 00:00 with the wrong link model.)
Next: the CTM score as set-decoder candidate evidence and as a blend correction, with seeds and a placebo.

Set decoder with the forward-CTM race features (4 columns, setdec_film_eval.py ONSET_CTMFEAT): val 0.8365 / pri
0.9003 vs BB 0.8315 / 0.9020 and a random placebo of the same shape 0.8350 / 0.8992 - not separable.
Gating (analysis/onset_ctm_gate.py): switch from the CNN's pick to its best site outside +-1 when the CTM prefers
it and the CNN margin is small: net at most +1 of 331 (e.g. CTM margin > 0.3 and CNN margin < 0.1: 10 switches,
1 fixed, 0 broken); loose gates break many (134 switches, 5 fixed, 37 broken). In most misses the true site is not
even the CNN's second site, and the CTM alone locates only 0.30. The 0.67 pairwise preference cannot be cashed in.

Onset summary: 2/3 of the onset loss sits in ~30 location misses, 26 of them races between habitual bottlenecks;
the candidates' own history points to the wrong one in ~3/4 of them; the forward physics prefers the right one
pairwise but cannot find it. Generated-run training does not move onset because its headroom is concentrated in
these undecidable races - a separate question from the generator's accuracy.
Generator side, activation-share demand fit (dlsim_fitdemand ACT=1, HB_MIN_SHARE 0.15): spurious heads fall
(D12_I5_N 5 -> 1, D12_I5_S 5 -> 1, D7_I405_N 8 -> 0) but total congestion overshoots (D12_I5_S 0.239 vs 0.096,
D7_I405_N 0.270 vs 0.108) - the update raises demand too broadly; unresolved.

## A site-locating feature from the generator analysis: today's upstream entry-demand anomaly

analysis/onset_site_features.py: candidates = the ctrl CNN's top-5 sites (> 1 link apart); share of windows where
a feature ranks the true site first among them. Misses (true site among the 5; 23 windows) / hits (301):
- upstream entry-demand anomaly (mean over the 3 entries merging at or upstream of the candidate of the
  last-3-slot conservation demand / the train weekday profile): **0.48** / 0.26 (chance 0.20; 11 of 23, one-sided
  binomial p ~ 0.002)
- forward-CTM first breakdown time: 0.17 / 0.94 (restates what the CNN knows)
- time to breakdown on the effective capacity: 0.17 / 0.26
- CNN p x anomaly: 0.00 / 0.86 (the CNN is confidently wrong in the misses)
The daily demand draw at the entries - a structure found while building the generator - separates the habitual
bottlenecks exactly where the CNN fails, and is near chance where it succeeds (complementary). Next: robustness
(entries 3 / 5, time span, train-month pseudo windows), then as set-decoder evidence against a placebo.

Robustness (analysis/entry_anomaly_robust.py): lab misses (23) 0.39-0.57 over entries 1/3/5 x span 3/6 slots
(chance 0.20), hits 0.24-0.29; train months with the out-of-fold H (43 misses) 0.19-0.35 (span 3: 0.35 / 0.35 /
0.26; span 6: chance). Reading: the ratio is today's entry demand over the TRAIN profile, so in the scored months it
carries the run's demand redraw relative to train - the factor that decides which bottlenecks a run activates -
while on train it only sees day-to-day noise. Next: estimate the run-level entry ratio from the scored run's past
days (released before T) as a stable feature.

Run-level version (analysis/run_entry_ratio.py): per entry, free-flow demand over the scored run's earlier days
(released before T) / the train weekday profile, 06-11 h. Misses: 0.35 / 0.43 / 0.52 for k = 1 / 3 / 5 entries
(today's 15-min anomaly 0.39 / 0.57 / 0.48); hits 0.20-0.28 (chance 0.20). Two independent estimates agree on the
misses. Potential: misses carry 65 % of the onset loss; fixing half would be ~+0.03 onset IoU - but the feature is at
chance on the 90 % hits, so it must be gated (set decoder) against a placebo.

Set decoder with the anomaly features (today / run, their candidate-relative values; setdec_film_eval.py
ONSET_ANOMFEAT): val 0.8371 / pri 0.9032 vs three window-shuffled placebos 0.8329-0.8385 / 0.8998-0.9032 - inside the
placebo range; private stage 1 is 0.8972 for all (the ranker never uses the columns). The evidence is informative
only on the ~7 % misses (23 windows) and at chance on hits; cross-month fitting has ~10 misses per month to learn
the gating, and the train month cannot teach it (a single run: the ratio has no run-shift content there).
Proposal (user decision): learn the gating on generated runs, which do contain run-level demand shifts with the
bottleneck switching they cause - use the generator for the part real data cannot teach, not as CNN training data.

## Learning the override on generated runs (analysis/gen_gate_train.py) - and why it fails

LambdaRank over the CNN's top-5 candidates, trained on 3,848 generated onset windows (ctrl CNN without adaptation,
occ_lab/gen_onset_infer.py), tested on the 331 lab windows (CNN pick 0.861): CNN-only features 0.773 (fixed 2,
broken 31 per seed); + run-shift evidence 0.773 (fixed 8, broken 37); + shuffled placebo 0.804 (fixed 3, broken 22).
The run-shift evidence fixes more than the placebo, but every variant overrides the CNN far too often.
Reason: the real-data CNN's top-1 on the GENERATED onset windows is 0.269 (lab 0.861): D7_I210_W 0.76, D7_I210_E
0.51, D7_I405_N 0.26, D7_I405_S 0.19, D7_I10_W 0.19, D12_I5_S 0.14, D12_I5_N 0.06, D7_I10_E 0.00. The generator's
first queues of the day happen elsewhere / otherwise than the real ones; maps and run shifts (what was measured)
are not onsets. This is the common cause of every generated-run onset failure so far. New generator metric: the
real-data CNN's onset top-1 on generated windows (should approach 0.86). Next: compare where the first queue of the
day occurs, real vs generated.

## Where and when the first queue of the day occurs, real vs generated (analysis/onset_site_distribution.py)

Onset time (median T): real is late morning to afternoon on most panels, generated is early: D7_I10_W 14.2 vs 10.9 h,
D12_I5_N 12.7 vs 9.5, D12_I5_S 12.2 vs 9.2, D7_I10_E 11.1 vs 7.4, D7_I405_N 10.2 vs 5.6 (D7_I210_E 6.6 vs 6.2).
Site distributions (overlap = sum of min shares): D7_I210_E 0.85, D7_I210_W 0.54, D7_I405_S 0.42, D12_I5_S 0.36,
D7_I10_W 0.23, D7_I405_N 0.20, D12_I5_N 0.05, D7_I10_E 0.00 - the same ordering as the real-data CNN's top-1 on
generated windows (0.51 / 0.76 / 0.19 / 0.14 / 0.19 / 0.26 / 0.06 / 0.00). Real D7_I405_N onsets sit at the
corridor's downstream end (link 351: 0.77; generated 0.00), D7_I10_E at links 7-10 (0.95; generated 0); generated
D12_I5_N onsets are taken by spurious head 156 (0.89, real 0.01). Candidate causes: the downstream boundary (queues
entering from outside the panel are not modelled - free outflow), morning demand inflated by the censored fill /
multipliers, incident-like spurious bottlenecks.

analysis/onset_boundary_check.py:
- Morning (06-11 h) flow, generated / real, median over links: 1.02-1.08 on every panel (D7_I10_E 1.079, D7_I405_N
  1.070, D12_I5_S 1.067). At a knife-edge breakdown a few % of demand move the first queue hours earlier. Sources:
  the lognormal run draws are not mean-corrected (exp(sigma eta), sigma up to 0.13 x 2.5 -> +4.6 %), and the
  censored-demand max() turns noise into upward bias.
- Downstream boundary: only D7_I405_N - 94 % of its real onsets are within 3 links of the downstream end, and the
  downstream-most link is queued with its upstream neighbour free in 100 % of its queued slots: a queue entering from
  a bottleneck outside the panel; the generator lets the last cell flow out freely.
- Spurious incident-like heads (D12_I5_N 156 takes 89 % of the generated onsets).
- D7_I10_E's real onset sites (7-10) are not at the ends: its hidden-bottleneck identification fails (separate).
Fixes: mean-corrected draws, unbiased censored demand, a downstream-boundary capacity from the observed discharge at
the last link, the 15 % permanence rule; judge by the real-data CNN's onset top-1 on generated windows.

## Generator v7: the four fixes (mean-corrected draws, unbiased censored demand, downstream-boundary capacity,
## 15 % permanence rule)

Replication, held-out daily IoU (best so far on all but D7_I10_E / D7_I10_W): D7_I210_W 0.545, D12_I5_S 0.494,
D7_I405_N 0.486, D12_I5_N 0.484, D7_I405_S 0.393, D7_I210_E 0.754. D12_I5_N keeps 2 of 5 hidden bottlenecks
(spurious 156 gone); D7_I405_N gets an outflow capacity of 5,813 vph.
Onset realism - the real-data CNN's top-1 on generated onset windows (ghb -> v7): D7_I405_N 0.25 -> 0.58, D12_I5_N
0.06 -> 0.18, D12_I5_S 0.14 -> 0.20, D7_I10_W 0.18 -> 0.24, D7_I210_E 0.51 -> 0.57, D7_I210_W 0.76 -> 0.32,
D7_I405_S 0.19 -> 0.10, D7_I10_E 0.00 -> 0.01; overall 0.269 -> 0.269. Site overlap D12_I5_N 0.05 -> 0.34,
D12_I5_S 0.36 -> 0.50. New failure: onsets at the UPSTREAM end (D7_I210_W link 0 takes 56 % of generated onsets,
real 0 %; D7_I405_S links 0/2/6). Not fixed: morning flow still +1.5..6 % and onsets still early; D7_I405_N link 351
still 0.01; D7_I10_E. Next: the upstream-end onsets, and whether the demand-multiplier fit inflates the morning.

## v7 diagnosis (analysis/v7_diagnosis.py; train days, no run draws)

- Correction: the "upstream-end" onsets of D7_I210_W / D7_I405_S are at the DOWNSTREAM end - on those panels the z
  index decreases downstream (path_set), so z index 0 is the last link. They do not occur in train-day simulations
  (links 0-2 never queued), so the run draws push the last link, which runs near capacity, over it.
- Morning flow sim / true is already +1.7..4.0 % with multipliers 1 (fitted adds up to +2.7 %, D7_I10_E). Not the
  zero clipping of negative entry estimates (< 0.03 % of mainline flow), not the exit placement (GAP_EXITS: the exit
  positions already coincide with the sensored links; identical results). Profile along the corridor: 0 at the
  upstream end, +2..6 % in the middle, ~0 at the downstream end - consistent with morning metering by bottlenecks in
  the truth that the simulation does not reproduce, rather than a demand bias. To be split by days with and without
  morning queues in the truth.
- D7_I210_E: the first links queue 69 % of the time in the truth (a queue from outside the panel at the upstream end);
  4 % with multipliers 1, 84 % after fitting.
- Split by days: the morning free-flow surplus is the same on days WITHOUT any true morning queue (+1.6..3.1 %) as
  on days with one - not missing metering. An intrinsic bias between the demand estimate and the simulation,
  accumulating along the corridor (~+0.02-0.05 % per gap over 100-350 gaps); zero clipping and exit placement ruled
  out. Candidates: the definition of the simulated 5-min flow vs the estimate's (averaging, exit share in the link
  flow). Proposal: a per-panel demand scale matched to the free-flow morning flows, and a one-gap unit test.

## Root cause of the flow surplus: the censored fill ADDED vehicles (analysis/gap_bias_unit.py and follow-ups)

- Per gap, the surplus appears even in gaps without entries (+1.6..5 veh/h), yet the truth conserves flow in zone-
  free gaps over the day (q_b - q_a: -2.1 veh/h in the rising morning = travel lag, ~0 all day). The error is
  positive all day and does not follow the flow trend (corr -0.32..0.00): not travel time.
- Observed-only demand (no fill): free-flow error +0.24 / +0.14 / -0.05 / -0.24 % (D7_I405_N / D12_I5_N / D12_I5_S /
  D7_I210_E) vs +2.80 / +1.87 / +1.52 / +1.96 % with the censored fill, which adds +6.7 / +2.8 / +3.8 / +2.0 % entry
  demand. A queued entry's demand arrives earlier, not additionally; the max() fill added vehicles.
- Volume-conserving fill (CONSERVE_FILL: within each censored run + 3 recovery slots, the filled demand is rescaled to
  the observed allocation over that run): free-flow error +0.08 / +0.20 / -0.09 / -0.22 %; the per-panel scale
  correction is not needed. But congestion collapses (D7_I405_N share 0.003 vs 0.113, D12_I5_S 0.035 vs 0.099; daily
  IoU 0.43 -> 0.26): the simulation needed 2-7 % extra vehicles to queue as much as the truth, i.e. its bottleneck
  capacities are too HIGH - the fill and the multipliers were compensating a capacity error. Suspect: the hidden-
  bottleneck capacity is measured at the next sensor and includes entries merging in between.
  Next: demand fixed (unbiased), fit only the bottleneck capacities (now identifiable), and check them against the
  discharge minus the merging entries.

## Capacity fit on unbiased demand (analysis/ctm_fitcap.py; CONSERVE_FILL=1, multipliers 1)

Held-out: free-flow flow error -0.07..-1.5 % (D7_I10_E -3.4 %); share / truth and daily IoU (v7 in brackets):
D7_I210_E 0.015/0.013, 0.755 (0.754); D12_I5_S 0.054/0.096, 0.326 (0.494); D7_I210_W 0.092/0.090, 0.320 (0.545);
D7_I10_E 0.072/0.037, 0.271 (0.230); D12_I5_N 0.053/0.092, 0.263 (0.484); D7_I405_S 0.043/0.096, 0.262 (0.393);
D7_I10_W 0.009/0.005, 0.178 (0.152); D7_I405_N 0.121/0.108, 0.167 (0.486). Many capacities cut (D7_I405_N 160 of
349, D12_I5_S 60 of 99) - again possibly absorbing another error. v7's higher map replication was partly bought
with the surplus vehicles. Remaining suspect: the time shape of the censored demand (backlog spread over 3 recovery
slots). Next: generate runs with this flow-consistent version and measure onset realism (real-CNN top-1 on
generated onsets, site overlap, timing).

## v8 (flow-consistent: unbiased demand + fitted capacities) - onset realism

Real-data CNN top-1 on generated onsets (ghb / v7 / v8): D7_I10_E 0.00 / 0.01 / **0.44** (site overlap 0.77, was
0.00), D12_I5_S 0.14 / 0.20 / **0.44**, D7_I210_E 0.51 / 0.57 / 0.60, D7_I10_W 0.18 / 0.24 / 0.26, D7_I405_S
0.19 / 0.10 / 0.24, D7_I210_W 0.76 / 0.32 / 0.29, D7_I405_N 0.25 / 0.58 / 0.21, D12_I5_N 0.06 / 0.18 / 0.14;
overall 0.269 / 0.269 / **0.317** - the first overall gain, from removing the demand bias at its source. Spurious
onset sites drop on most panels (share of generated onsets at sites with < 1 % of real: D7_I10_E 1.00 -> 0.00,
D7_I210_W 0.58 -> 0.03, D7_I405_S 0.67 -> 0.10) but D7_I405_N gets new ones (0.74; 160 of 349 capacities cut).
Onsets still early (D12_I5_S 8.5 vs 12.2 h, D12_I5_N 9.2 vs 12.7, D7_I210_W 5.0 vs 9.0; D7_I405_N matches 9.9 vs
10.2). Hypothesis: capacities fitted to the whole-day congestion are cut to create afternoon queues that the
demand shape under-produces, and the cut caps then break in the morning. Next: congestion share by hour.

## Congestion by hour (analysis/congestion_by_hour.py; unbiased demand)

Truth peaks late (15-21 h) and lasts into the evening; with fitted capacities the simulation over-congests midday
and dissolves too early: D7_I405_N 13-15 h 0.081 vs 0.224, 17-19 h 0.296 vs 0.146, 19-21 h 0.199 vs 0.021; D12_I5_N
17-19 h 0.216 vs 0.080; D12_I5_S 17-19 h 0.237 vs 0.069; D7_I10_E 07-11 h 0.007 vs 0.03-0.05. Released capacities
under-congest everywhere. The capacity cuts buy midday queues (the early onsets) without producing the long
evening queues: the evening persistence is a demand-side (arrival volume / timing) matter the volume-conserving
fill (backlog spread over the run + 3 slots) does not reproduce. Next: do evening queues extend beyond the panel's
upstream end (censored boundary demand)? Otherwise reconstruct arrivals with Newell cumulative counts.

## Evening persistence: arrivals and capacities are right; the diverge must be FIFO

- Arrivals at the true evening queue tails match (sim - true at the free link just upstream: +0.4 / -0.0 / -1.1 %;
  tail-gap entry demand not lost), and the discharge at recurrent heads is constant over the day (+-1 %; D12_I5_S
  evening 0.96 / 0.875, D7_I405_S 0.97) - no time-varying capacity.
- One-day trace (D7_I405_N, evening queue from the downstream end): the true queue spreads upstream (3 links up:
  5,700-6,100 vph at 38 km/h) while the simulation keeps those links free (7,000+ vph at 111 km/h): its exits let the
  exiting vehicles bypass the mainline queue (DLSim-style independent diverge movements).
- FIFO diverge in the CTM (exiting vehicles held back with the mainline; env FIFO=1), released + hidden capacities,
  unbiased demand, no fitting: D12_I5_N by 2-h block 13-15 / 15-17 / 17-19 / 19-21 h = 0.094 / 0.211 / 0.247 / 0.131
  vs truth 0.096 / 0.226 / 0.216 / 0.098 (non-FIFO 0.025 / 0.025 / 0.018 / 0.003); D7_I210_W 15-17 h 0.188 vs 0.202;
  D12_I5_S and D7_I405_S now overshoot the evening (17-19 h 0.559 vs 0.237, 0.340 vs 0.124); D7_I405_N unchanged.
  The generator's diverge is FIFO - the missing evening queues were a diverge-model error, and the earlier capacity
  cuts (midday over-congestion, early onsets) were compensating it. Next: re-identify hidden bottlenecks under FIFO
  (overshooting panels), the D7_I405_N downstream end, then replication and onset realism.

FIFO, released + hidden capacities, unbiased demand, no fitting - all panels (share gen / true, map corr, daily IoU,
discharge sim / true at shared heads): D7_I210_W 0.096/0.090, 0.63, 0.357, 1.025; D12_I5_N 0.095/0.092, 0.44, 0.260,
1.007; D12_I5_S 0.208/0.096, 0.26, 0.291, 1.011; D7_I405_S 0.123/0.096, 0.13, 0.202, 1.029; D7_I10_W 0.008/0.005,
0.98, 0.189, 1.027; D7_I405_N 0.004/0.108, 0.24, 0.033; D7_I210_E 0.001/0.013, 0.58, 0.053; D7_I10_E 0.002/0.037,
0.01, 0.008. Discharges match (1.01-1.03): capacities need no fitting. Three groups: (1) reproduced from physics alone
(D7_I210_W, D12_I5_N, D7_I10_W); (2) queues spread too far / wrong places (D12_I5_S, D7_I405_S); (3) almost no
queues where the truth's queues come from outside the panel or unidentified bottlenecks (D7_I405_N downstream end,
D7_I210_E upstream end, D7_I10_E).

## Cause investigation under FIFO (analysis/fifo_checks.py)

1. D7_I10_E: hidden bottlenecks back on (they had been switched off under the wrong, non-FIFO physics): daily IoU
   0.008 -> 0.284, map corr 0.82 (HB_ALL=1).
2. Unsensored storage: with FIFO the queue extent depends on it; the assumed length x 1/2/3/4 gives D12_I5_S map corr
   0.26/0.48/0.56/0.60 (IoU 0.291/0.331/0.349/0.356), D7_I405_S 0.13/0.34/0.41/0.41, D7_I10_E 0.82/0.84/0.86/0.87
   (IoU up to 0.351); D7_I210_W flat (0.357 -> 0.350); D12_I5_N unchanged (almost no unsensored links). Unsensored
   links are ~3-4 x the median sensored length (1.1-1.5 km). (The earlier "length does not matter" was under non-FIFO.)
3. D7_I210_E: the truth's queue extends beyond the panel's upstream end (upstream-most links queued 69 % of the
   time); the boundary demand is censored to the discharge, so the simulation never has the arrivals to hold it.
4. D7_I405_N: a merge bottleneck at link 348 (discharge 5,815 vs released 5,832; head 347 active on 273 of 273 days)
   plus the downstream-end queue. In the truth the entry merging at 348 brings ~1,900 vph and the mainline backs up
   from 347; the simulation estimates that entry at ~1,050 vph, the merge sits exactly at capacity and never queues.
Common cause of 3 and 4 (and likely of D12_I5_S's over-spread): the demand hidden behind queues - at the boundary
and at merges - is not recovered. Next: re-estimate censored entry demand from the entry's own free-flow level on
that day / other days plus the bottleneck discharge constraint.

## Arrival-curve reconstruction for censored demand (CONSERVE_FILL=2)

Episode = censored run + release period (observed demand above the day's level, up to 12 slots); arrivals follow the
day-level profile shape, total = observed allocation over the episode. FIFO, hidden on, storage x3; daily IoU fill 1
-> fill 2: D7_I210_E 0.052 -> 0.127 (map corr 0.58 -> 0.93; share still 0.002 vs 0.013), D7_I405_N 0.044 -> 0.047
(share 0.004 vs 0.108), D12_I5_S 0.349 -> 0.324, D7_I210_W 0.350 -> 0.300, D7_I10_E 0.341 -> 0.308, D7_I405_S
0.246 -> 0.206. Mixed. D7_I405_N: link 348 runs exactly at its capacity (5,948 of 5,949) while 347 stays free -
inconsistent with a receiving-capacity squeeze at the merge; the CTM internals need a cell-level check first.

## Cell-level trace (D7_I405_N, 17:00 from the observed state) and the merge-position ambiguity

- Starting from the observed 17:00 state, the CTM holds the queue correctly; over a whole day it never creates it.
  Link 347 (6 lanes, 7,858 vph) drops to 3 lanes, 19 unsensored links (~11 km) lead to 348, where an entry
  (~1,500-1,900 vph) merges and 68 % exits. The hidden capacity measured at 348 (5,815) includes that entry, so the
  3-lane stretch got ~5,949 while 347 can feed at most 0.70 x 7,858 = 5,514 into it - it can never queue. The true
  mainline bottleneck is ~4,000 vph.
- Hidden capacity = discharge at h+1 minus the entry merging in the gap (HB_SUB_ENTRY), daily IoU (none / always /
  by geometry = only if the entry merges downstream of the first cell after the head): D7_I210_E 0.052 / **0.753** /
  0.052; D7_I210_W 0.350 / 0.360 / **0.367**; D12_I5_S 0.349 / 0.288 / **0.356**; D7_I405_S 0.246 / 0.258 / **0.283**;
  D7_I10_E **0.341** / 0.137 / 0.137; D7_I10_W **0.217** / 0.150 / 0.150; D7_I405_N 0.044 / 0.071 / 0.067. A
  regression-slope rule failed (profile demand is a weekday average; attenuated slopes).
- Where the bottleneck sits relative to the merge inside an unsensored gap differs per site; no single rule fits.
  Next: per recurrent head, choose between "bottleneck upstream / downstream of the merge" (one bit each) by
  train replication of that head.
Per-head choice (analysis/ctm_hb_choice.py; HB_RULE=choice): held-out daily IoU D7_I210_E 0.753, D7_I210_W 0.372,
D7_I10_E 0.341, D7_I405_S 0.328, D12_I5_S 0.327, D12_I5_N 0.262, D7_I10_W 0.217, D7_I405_N 0.067 - no continuous
capacity fitting at all (released FD, FIFO, measured hidden bottlenecks with one merge-position bit per head,
unbiased demand, storage x3). 7 of 8 panels at or above the same-run next-week reference; D7_I405_N unresolved.

## v9 (physics only: FIFO, measured hidden bottlenecks + per-head merge bit, unbiased demand, storage x3) - onsets

Real-data CNN top-1 on generated onsets, v8 -> v9: D7_I405_N 0.21 -> 0.69, D7_I210_W 0.29 -> 0.34, D7_I405_S
0.24 -> 0.32, D12_I5_N 0.14 -> 0.29, D7_I10_W 0.26 -> 0.25, D7_I210_E 0.60 -> 0.44, D7_I10_E 0.44 -> 0.23, D12_I5_S
0.44 -> 0.14; overall 0.317 -> **0.358** (0.269 at ghb). Onset timing now matches (median real -> generated):
D7_I10_E 11.1 -> 11.2 h, D12_I5_N 12.7 -> 12.2, D7_I405_N 10.2 -> 10.8, D7_I210_E 6.6 -> 6.4, D7_I10_W 14.2 -> 13.4,
D7_I210_W 9.0 -> 8.4; still early on D12_I5_S (12.2 -> 9.2) and D7_I405_S (10.8 -> 9.7) - the early onsets were
the capacity cuts compensating the non-FIFO diverge. Site overlap D12_I5_N 0.48, D7_I405_S 0.51, D7_I210_W 0.52;
spurious onset sites remain (share at sites with < 1 % of real onsets: D12_I5_S 0.69, D12_I5_N 0.49, D7_I210_W 0.45).
Real data: 0.86. Next: where the spurious onset sites come from.

## Spurious onset sites (analysis/spurious_onset_sites.py, spurious_site_timing.py, apre_onset_check.py)

- Structural, not the run draws: train-day simulations give the same spurious share (D12_I5_S 0.61, D12_I5_N 0.55,
  D7_I210_W 0.32). The sites do queue in the truth (7-27 % of the time) but never first; both truth and simulation
  break down locally there (spillback 0-5 %), the simulation 0.4-1.3 h earlier (D12_I5_S site 97: 12.3 vs 13.6 h,
  site 11: 13.2 vs 14.3; D7_I210_W 80: 8.8 vs 9.9), so they win the day's first-queue race.
- The pre-breakdown factor a (causal capacity drop) changes nothing even at 1.3 (queued share 0.1328 -> 0.1328; it
  does reach the kernel): these queues are not knife-edge but bottlenecks set FAR below demand - e.g. D12_I5_S site 97
  at 0.47 x released (discharge minus a 2,818-vph entry, the 'subtract' choice picked by all-day occupancy); the
  truth queues there on 44 of 91 days. The spurious sites are hidden bottlenecks with under-estimated capacity.
- Root: capacity measured at the downstream sensor, entangled with merging entries. Proposal: measure the mainline
  capacity at the queued head itself, (1 - exit share) x q_h during head episodes (q_h is exact on the FD), adding the
  entry's allocation only when it merges upstream of the bottleneck.

## Spurious onsets, resolved for D12_I5_S: the subtracted capacity must cap the MAINLINE movement

- Head-based capacity (1 - exit share) x q_h during head episodes agrees with 'discharge minus entry' (+-200 vph;
  D12_I5_S head 97: 3,472 vs 3,840). Pre-breakdown free-flow throughput p99 / discharge = 0.95-1.05 at almost every
  head: no large capacity drop. The low capacities are real.
- But with no unsensored cell between head 97 and link 98 the 'subtract' capacity was set as the RECEIVING capacity
  of link 98 itself, so mainline + the 2,818-vph entry were squeezed into 3,840 (truth at 98: ~6,858) and site 97
  broke daily. Fix (MAINLINE_CAP): a subtracted head caps the mainline movement into the cell (entries merge on top,
  the cell keeps its released capacity); the per-head choice redone with that mechanism.
- Train-day onset structure (site overlap / spurious / median onset real vs sim): D7_I210_E 0.76 / 0.00 / 6.6 vs 7.1 h;
  D7_I10_E 0.75 / 0.03 / 11.1 vs 11.1; D7_I405_S 0.66 / 0.06 / 10.8 vs 10.2; D12_I5_S **0.62 / 0.12** (was 0.29 /
  0.61) / 12.2 vs 12.2; D7_I210_W 0.52 / 0.32 / 9.0 vs 8.5; D12_I5_N 0.39 / 0.59 / 12.7 vs 12.6; D7_I405_N 0.21 /
  0.04; D7_I10_W 0.19 / 0.02. Remaining: D12_I5_N (sites 239 and 114 break first), D7_I210_W's downstream end,
  D7_I405_N and D7_I10_W site overlap. Replication D7_I210_E 0.753 -> 0.479 (choice flipped to raw).

## v10 (mainline caps for subtracted heads) - final onset metric

Real-data CNN top-1 on generated onsets (ghb / v8 / v9 / v10): D7_I405_N 0.25/0.21/0.69/0.70, D7_I210_E
0.51/0.60/0.44/0.48, D7_I210_W 0.76/0.29/0.34/0.34, D12_I5_N 0.06/0.14/0.29/0.33, D7_I405_S 0.19/0.24/0.32/0.33,
D7_I10_W 0.18/0.26/0.25/0.25, D7_I10_E 0.00/0.44/0.23/0.23, D12_I5_S 0.14/0.44/0.14/0.21; overall
0.269 / 0.317 / 0.358 / **0.379** (real data 0.86). D12_I5_S: spurious onsets 0.12 in train-day simulations but 0.50
in the generated runs - the run draws (draw scale 2.0 for this panel, chosen to match run diversity) break sites the
truth never breaks first. D7_I210_E: onsets late (6.6 -> 8.6 h) and too many (1,086).

## Run-draw scale and spurious onsets (D12_I5_S; 4 runs x 60 days each)

Draw scale 0 / 0.5 / 1 / 2 (current): spurious onset share (sites < 1 % of TRAIN onsets) 0.18 / 0.23 / 0.28 / 0.48,
site overlap with train 0.68 / 0.67 / 0.66 / 0.47, between-run map corr 0.98 / 0.96 / 0.93 / 0.84. But real runs also
move onsets: validation has 25 %, private 20 % of its onsets at train-rare sites (observed layer, evaluation use).
Of the x2 generated onsets at train-rare sites (0.48): 0.12 at sites a scored run uses, 0.36 at sites no real run
uses (z 34: 0.19, z 73: 0.13 - hidden bottlenecks that win the race under large draws). The real onset-site shift
(20-25 %) matches draw scale 0.5-1 (23-28 %); the x2 scale, chosen to match the MAP diversity of real runs
(0.54-0.85 - not reachable by demand draws: 0.93 at x1), overshoots and invents onsets. The measured sigma (x1) is
the right draw; the remaining map diversity must come from something else (incidents / run-level capacity).

## v11: run draw scale 1x on all panels (2026-09-28)
Same as v10 but every panel draws run demand with the measured sigma (1x; v10 used 2.0-2.5 on D12_I5_S/D12_I5_N/D7_I10_E and 1.5 by default; D7_I405_S was already 1.0). OLD_SCALES=1 restores the v10 scales in dlsim_make_runs.py.

Real ctrl CNN top-1 on generated onset windows:

| | I5_N | I5_S | I10_E | I10_W | I210_E | I210_W | I405_N | I405_S | ALL |
|---|---|---|---|---|---|---|---|---|---|
| v10 | 0.33 | 0.21 | 0.23 | 0.25 | 0.48 | 0.34 | 0.70 | 0.33 | 0.379 |
| v11 | 0.36 | 0.42 | 0.23 | 0.25 | 0.48 | 0.36 | 0.73 | 0.33 | **0.418** |

Onset sites vs the real runs (analysis/onset_sites_vs_runs.py; train truth, validation/private observed layer, evaluation only).
- "rare" = sites with < 1 % of train onsets.
- "unused" = sites that no real run uses (train < 1 % and never in validation/private).

| panel | real val / pri at rare | v10 gen rare / unused | v11 gen rare / unused | overlap with train v10 -> v11 |
|---|---|---|---|---|
| D7_I10_E | 0.17 / 0.22 | 0.18 / 0.10 | 0.07 / 0.03 | 0.67 -> 0.78 |
| D7_I10_W | 0.15 / 0.12 | 0.14 / 0.13 | 0.11 / 0.11 | 0.22 -> 0.22 |
| D7_I210_E | 0.14 / 0.03 | 0.01 / 0.01 | 0.01 / 0.01 | 0.79 -> 0.78 |
| D7_I210_W | 0.15 / 0.22 | 0.45 / 0.01 | 0.39 / 0.01 | 0.52 -> 0.53 |
| D7_I405_N | 0.02 / 0.07 | 0.08 / 0.08 | 0.07 / 0.07 | 0.31 -> 0.27 |
| D7_I405_S | 0.22 / 0.30 | 0.18 / 0.04 | 0.18 / 0.04 | 0.53 -> 0.53 |
| D12_I5_N | 0.15 / 0.09 | 0.54 / 0.03 | 0.55 / 0.02 | 0.45 -> 0.44 |
| D12_I5_S | 0.25 / 0.20 | 0.50 / 0.37 | 0.28 / 0.22 | 0.44 -> 0.60 |

Findings:
- 1x improves D12_I5_S most: top-1 0.21 -> 0.42 and overlap 0.44 -> 0.60.
- D7_I10_E drops its invented sites but its top-1 is unchanged.
- D12_I5_N (sites 239/114) and the D7_I210_W downstream end are almost all at sites that real validation/private runs DO use. They are not invented sites but over-weighted ones: 0.55 / 0.39 of generated onsets, against 0.09-0.22 in real runs.
- Invented sites remain on D12_I5_S (0.22), D7_I10_W (0.11) and D7_I405_N (0.07).

## Over-weighted onset sites: root-cause investigation (2026-09-28)
Targets: D12_I5_N 239 / 114 and the D7_I210_W downstream end. Real validation/private runs use these sites, but the generator puts 0.55 / 0.39 of its onsets there, against 0.09-0.22 in real runs.

1. **Not the run draws.** A clean replay of every train day (observed demand, no draws, no incidents) gives the same shares as the generated runs:
   - D12_I5_N 239: truth 0.00, replay 0.41, generated 0.41. 114: 0.00 / 0.11 / 0.13.
   - D7_I210_W end: 0.00 / 0.34 / 0.35.
   (analysis/overweight_sites.py.)
2. **Race with the upstream neighbour** (analysis/head_race.py, head_race2.py).
   - D12_I5_N, 238 vs 239: in truth 238 queues first on 96% of days, and 239 is free (> 0.9 vf) at that moment on 57%. In the replay 239 queues first on 100%.
   - D7_I210_W: head rank 77 (z 3) is active on 91% of truth days, first at 9.7 h. The replay never activates it; its end queue at 9.0 h covers 77.
   - Timing is aligned (median dt 0.0 h). The site is wrong.
3. **Anatomy of D12_I5_N day 0** (analysis/_chk_day.py).
   - About 11.5 h, flow at 236-239 surges from 6,200 to 8,400. The upstream (235) and downstream (240) links stay at about 6,200, so the surge enters in gap 235->236 and leaves in gap 239->240 (ramp-to-ramp paths).
   - Truth: the 5-lane section 236-238 breaks down at about 8,400, while 239 carries about 8,000 at 0.6-0.97 vf.
   - Model: the exit share is fixed at 0.192, so the surge stays on the mainline and hits the mainline cap after 239 (mcap 5,328; raw 6,254 minus the entry). 239 queues at 6,591.
   - The model's entry at 240 is clipped to 0 during the surge, which is the signature of the inconsistency.
   - Free-flow medians are unaffected (replay/truth q = 1.00 in every hour block, analysis/_chk_split.py).
4. **Three fixes tried in the replay** (analysis/replay_sites.py). None removes the over-weighting.
   - EXIT_TV=1 (time-varying exit share: when q_b - (1-F) q_a < rho x the profile, keep the entry at rho x profile and raise F_t).
     - The surge now passes 239 at free speed on some days.
     - 239: 0.41 -> 0.40 (rho 1) / 0.44 (rho 0.5). 238: 0.05 -> 0.19. 105: 0.39 -> 0.24.
     - Remaining: F_t is only set in free slots, so FIFO still throttles 239 at 5,328 / 0.808 during congestion. And 239 dips at 6,600 minutes before the surge.
   - HB_RAW_AT_MERGE=1.
     - A bug in placement: the 'raw' hidden capacity is measured at the next sensor, after the entry merges, but it is put on the mainline-only cell before the merge. At D7_I210_W 77 it never binds (4,969 vs about 3,000 mainline).
     - Moving it to the merge cell: head 77 is still never active, the end still takes 0.33, and D7_I210_W 33 is over-weighted (0.18).
   - OUT_CAP_MODE=ffp95: out_cap 4,726 -> 4,779, no effect.
5. **Per-head pre-breakdown flow vs queue discharge** (truth; max free flow in the 30 min before the first activation / median discharge).

   | panel | head | ratio |
   |---|---|---|
   | D12_I5_N | 105 | 0.971 |
   | D12_I5_N | 114 | 1.003 |
   | D12_I5_N | 239 | 1.020 |
   | D7_I210_W | 74 | 1.000 |
   | D7_I210_W | 77 | 1.004 |
   | D7_I210_W | end | **1.193** |

   Before its outside queue arrives, the end passes 5,654 at free speed; its discharge is 4,733. The CTM breaks every head at its discharge, so heads with a ratio > 1 (239, the end) break too early and 105 (< 1) too late. This matches the direction of every over- and under-weighting above.

   Next step: a per-cell breakdown factor `ap` in the kernel. The receiving side is ap x cap while the upstream cell is uncongested and cap after breakdown; the same for mcap and the boundary (out_ap). It is calibrated so that the replay's ratio matches the truth's (analysis/ctm_apre_fit.py).

### Per-head breakdown factor and follow-ups (clean replay, all train days)
Kernel change: `ap` per cell (receiving side ap x cap while the upstream cell is uncongested; also on mcap), `out_ap` on the downstream boundary, `MERGE_RULE` (0 lane-proportional as before, 1 demand-proportional, 2 entry first). Defaults reproduce v11.

Calibration (analysis/ctm_apre_fit.py, 4 rounds, half the train days):
- **HEAD_FREE=0.9** (h+1 free) gives the ctm_apre_* files: no end head, and D12_I5_N 238 is missed.
- **HEAD_FREE=0.6** (h+1 not queued, end included) gives the ctm_apre3_* files, and the ctm_apre2_* files when combined with merge placement and EXIT_TV.
- Caveat: heads that the replay never activates are lowered blindly by 3% per round. After 4 rounds they sit at 0.885 (clip range 0.85-1.3); the replay ratio does not follow ap at D12_I5_N 239 (ap -> 1.09-1.2, replay ratio 0.99).

| variant | panel | overlap | at truth-rare | same first site | map corr | top replay sites (truth) |
|---|---|---|---|---|---|---|
| base | D12_I5_N | 0.46 | 0.53 | 0.30 | 0.44 | 239 0.41 (0), 105 0.39 (0.52), 114 0.11 (0) |
| apre (h+1 free) | D12_I5_N | 0.58 | 0.41 | 0.38 | 0.44 | 105 0.50, 239 0.31, 114 0.09 |
| apre3 | D12_I5_N | **0.60** | **0.39** | 0.40 | 0.44 | 105 0.53, 239 0.29, 114 0.09 |
| apre2+merge+tv | D12_I5_N | 0.55 | 0.45 | 0.38 | 0.49 | 105 0.37, 239 0.28, 114 0.16 |
| + MERGE_RULE=1 | D12_I5_N | 0.52 | 0.48 | 0.36 | **0.73** | 105 0.35, 239 0.27, 238 0.14 (0.25) |
| base | D7_I210_W | 0.51 | 0.40 | 0.44 | 0.64 | 29 0.49 (0.38), end 0.34 (0), 6 0.12 (0.17) |
| apre3 | D7_I210_W | 0.55 | **0.17** | 0.48 | 0.59 | 29 0.67, 33 0.16 (0.01), 6 0.16 |
| apre2+merge+tv | D7_I210_W | 0.56 | 0.27 | 0.47 | 0.61 | 29 0.51, 33 0.26, 6 0.23 |
| + MERGE_RULE=1/2 | D7_I210_W | 0.56-0.57 | 0.29 | 0.47 | 0.58-0.59 | 29 0.48, 33 0.27, 6 0.23 |

Findings:
- The per-head factor removes most of the end over-weighting on D7_I210_W (end 0.34 -> not in the top 3) and restores D12_I5_N 105 (0.39 -> 0.53, truth 0.52).
- The mass moves to D7_I210_W 29 (0.67 vs truth 0.38) and 33.
- D12_I5_N 239 keeps 0.27-0.31. Its remaining cause is the ramp-to-ramp surge. EXIT_TV only sets F_t in free slots, so during the congestion FIFO throttles 239 with the fixed share (5,328 / 0.808 = 6,591; truth about 8,000).
- D7_I210_W 77: with the capacity at the merge cell, 78 is capped (4,399), but the held-back flow queues on the ramp, not on the mainline. 77 stays free under all three merge rules, while in truth 76-77 fall to 0.45 vf. Unresolved.
- Side finding: the demand-proportional merge (MERGE_RULE=1) raises the D12_I5_N queued-share map corr from 0.49 to 0.73.

### (b) D12_I5_N 239: cause of the remaining onsets (2026-09-28)
**Days where the replay still starts at 239** (apre3 + EXIT_TV rho 1, 74 days; analysis/exit239_cause.py):
- Truth starts at 238 (35), 105 (20) or 237 (9), about 5 min before the replay.
- The implied truth exit share F = 1 - (q240 - P) / q239 is 0.163 before and 0.221 after (q90 0.28); the fixed share is 0.192.
- At the replay breakdown, truth passes 8,277 at 239 (min v/vf 0.68), while the replay is jammed at 7,705.
- EXIT_TV raised F before the onset on 47% of those days, but only in 10% of the slots.
- After the breakdown the replay uses 0.192 (free slots only), so FIFO throttles 239.

**High-pass flow responses** (minus a 1 h mean, free slots, analysis/ramp_to_ramp_scan.py):
- At D12_I5_N 239->240, 0.32 of an entry-236 fluctuation passes the exit, against 0.88 of a mainline fluctuation from upstream (fixed-share model: 0.81 for both). About two thirds of the vehicles entering at 236 leave at 240: ramp-to-ramp paths that a node turning fraction cannot represent.
- Same signature at 4 more pairs (entry pass share 0.04-0.21 vs mainline 0.47-0.92):
  - D7_I210_W exit 77->78 / entry 76
  - D7_I405_S 84->85 / 79
  - D12_I5_S 11->12 / 9 and 60->61 / 58
  - D7_I10_E 97->98 / 94
- Elsewhere, both pass shares are about 1.00, not 1 - F: short-time fluctuations pass the exits entirely. Exit volumes are smooth (OD-profile driven) rather than proportional to the instantaneous mainline flow.
- Only pairs with the entry at least one gap upstream of the exit are valid. With the entry in the exit's own upstream gap, e_j is built from q_h, and q_h's own noise does not pass (a spurious low share).
- Few pairs are testable: entries sit in almost every gap, and the four sensors must be free.

**Implication.** The exit is path-dependent, not a fraction of the mainline.
- Ramp-to-ramp traffic leaves at its exit whatever the mainline does, including during the congestion. Truth: 239 about 8,000, 240 about 6,200, so the exit takes about 2,800.
- The model's FIFO diverge with a fixed F holds that exit traffic back and jams 239.
- Candidate fixes (not implemented):
  - (i) origin-tagged cells (multi-commodity CTM) with entry-to-exit shares estimated from the cross responses;
  - (ii) exit volume as a per-slot demand (smooth profile) instead of a share.

### (b)(i) Origin-tagged ramp-to-ramp classes: implemented, NO effect on the 239 onsets (2026-09-28)
**Implementation.**
- Kernel classes (`cls_ent`, `cls_exit`, `wsh`): a share of entry j's vehicles rides with the mainline and leaves only at the exit after h. Each cell's exit share mixes by composition (FIFO consistent).
- `set_od_pairs`: background exit shares per slot keep the weekday-mean exit volumes.
- `cls_wsh` (OD_BF): the class takes beta0 x ebar + betaf x (e - ebar), ebar = the day's 1 h mean of the entry.
- censored_demand: with classes, the entry of the exit's own gap follows from the class-aware exit volume. Before this, it was clipped to 0 in every surge.
- Default off (OD_PAIRS=1 in replay_sites / _chk_day). Pairs and beta: analysis/od_pairs_fit.py (--calib: secant on the replay's own s_e/s_m).

**Checks.**
- The replay without classes already has s_e/s_m = 0.71 at 236->239 (truth 0.37). beta 0.63 gives 0.04; the calibrated beta 0.252 gives 0.37. The mechanism reproduces the truth statistic.
- Truth, binned by the size of the entry-236 fluctuation:
  - large surges (> +600) pass exit 240 at 0.11-0.27;
  - dips pass at 0.26-0.31;
  - small ones pass at 0.5-0.9.
  So the variable part of entry 236 is the ramp-to-ramp OD. That motivated OD_BF.
- Onset sites in the replay (D12_I5_N 239 share; truth 0.00):
  - base 0.41
  - beta 0.25: 0.42
  - beta0 0.3 / betaf 0.8: 0.44
  - beta0 0.3 / betaf 0.9: 0.45
  - beta0 0.4 / betaf 0.9: 0.42
  - apre3: 0.29; apre3 + beta 0.25: 0.29

**Conclusion.** The ramp-to-ramp paths are real, but they are not what starts the queue at 239.
- Already visible in exit239_cause.py: only 1 of the 74 replay-239 days had a surge (implied F > 0.24) before the onset. The day-0 anatomy that motivated (b) was the exception.
- On day 5 with classes, the replay breaks 239 at about 6,200 (11.25 h), with no surge. Truth passes 6,300-6,800 there at free speed.
- The binding element is the head-239 mainline cap (mcap 5,328 = discharge 6,254 minus the entry profile 926). It sits right at the normal peak mainline flow after the exit (truth q240 about 6,100-6,300 including the entry).

### D12_I5_N 239: capacity type (2026-09-28)
**Truth** (analysis/cap239.py):
- Links 240-243 never exceed 6,518 (p99 of all slots, equal on all four; released caps 8,108 / 6,433 / 8,588 / 12,881).
- While head 239 is active, 236-243 all carry about 6,320 (239: 0.35 vf, 240-243: 1.0 vf).
- There is a ceiling on the TOTAL flow after the merge in gap 239->240. The chosen option 'sub' is a mainline-only cap: 5,328 = raw discharge 6,254 minus the entry profile 926.
- Sequence of the first queued links on a day: (238, 237, 236) on 172 of 273 days.

**Kernel bug found and fixed.** With ap > 1, the receiving side stayed capped by w (kj - k) = cap at k >= kc. The FD apex did not move, so ap > 1 had no effect anywhere; ap < 1 worked.
- Now, for bc > 1: R = min(bc cap, w_b (kj - k)) with w_b = bc cap / (kj - bc cap / vf).
- Consequences for earlier results:
  - The earlier ap calibrations (ctm_apre, apre2, apre3) only had their < 1 parts working.
  - 239's "replay ratio does not follow ap" was this bug. Under sub, ap acts on mcap, which did work.
  - The D7_I210_W out_ap (boundary) was unaffected.

**Head-239 variants in the clean replay** (analysis/cap239_test.py; 239 share, truth 0.00):

| variant | q240 free p99 (truth 6,518) | overlap | 239 | 238 (truth 0.25) | 105 (truth 0.52) |
|---|---|---|---|---|---|
| sub | 6,703 | 0.46 | 0.41 | 0.05 | 0.39 |
| sub x 1.04 | 6,703 | 0.48 | 0.39 | 0.06 | 0.40 |
| raw x 1.00 | 6,254 | 0.50 | 0.38 | 0.12 | 0.35 |
| raw x 1.02 | 6,293 | 0.60 | 0.26 | 0.17 | 0.41 |
| raw x 1.04 | 6,342 | 0.68 | 0.16 | 0.20 | 0.46 |
| raw x 1.06 | 6,372 | **0.73** | **0.10** | 0.24 | 0.47 |

**Across panels** (analysis/hb_type_evidence.py): p99(free q_{h+1}) / raw discharge is 1.03-1.07 at most heads with an entry.
- The ceiling a head holds before breakdown is about 5% above its queue discharge: a capacity drop of the classic size.
- The earlier note line "no significant capacity drop" was wrong.
- Larger ratios (entry adds above the ceiling, i.e. sub-type): D7_I210_W 13 (1.55) and 77 (1.15); D7_I405_S 9 (1.20), 13, 63 (1.29), 84; D12_I5_S 18 (1.16), 60.
- 'sub' was chosen at ratio 1.01-1.06 on I5_N 14 and 239; I210_W 5, 7, 18; I405_N 110 and 347; I405_S 15 and 60; I5_S 11, 70, 89 and 97. These are suspicious: the replication choice was made without the pre-breakdown factor.

**Truth-only type rule** (analysis/ctm_type_rule.py). Heads with raw/sub options: P / raw <= 1.10 -> raw with ap = P / raw, else sub. The end: out_ap = p99 free / out_cap.

| variant | panel | overlap | at truth-rare | same first site | top replay sites (truth) |
|---|---|---|---|---|---|
| rule, merge heads only (AP_OTHER=0) | D12_I5_N | **0.70** | 0.29 | **0.49** | 105 0.46 (0.52), 238 0.22 (0.25), 114 0.15 (0); 239 below 0.15 |
| rule, also ap = P/D at other heads | D12_I5_N | 0.65 | 0.28 | 0.48 | 105 0.37, 238 0.31, 239 0.20 (105 and 114 got ap 1.04: 105 too few) |
| rule + replication fit of ap (ctm_apre_fit, either NEVER_ACT mode) | D12_I5_N | 0.59 | 0.39 | 0.40 | 239 back to 0.30: the breakdown-moment statistic moves 239 the wrong way |
| rule | D7_I210_W | 0.56 | 0.25 | 0.51 | 29 0.57 (0.38), end 0.21 (0), 6 0.18 (0.17) |

For comparison, base: D12_I5_N 0.46 / 0.53 / 0.30; D7_I210_W 0.51 / 0.40 / 0.44.

## v12: truth-only capacity-type rule on all panels (2026-09-28)
v11 settings plus RULE=1 (ctm_rule0_<panel>.npz: AP_OTHER=0, i.e. factors only at the merge heads, and the end out_ap) and the kernel fix for ap > 1.
- Real ctrl CNN top-1 on the generated onset windows:

  | version | ALL | I5_N | I5_S | I10_E | I10_W | I210_E | I210_W | I405_N | I405_S |
  |---|---|---|---|---|---|---|---|---|---|
  | v11 | 0.418 (SE 0.006) | 0.36 | 0.42 | 0.23 | 0.25 | 0.48 | 0.36 | 0.73 | 0.33 |
  | v12 | **0.376** (SE 0.006) | 0.39 | **0.59** | 0.23 | 0.21 | 0.56 | 0.43 | **0.26** | 0.35 |

- Sites vs the real runs (overlap with train v11 -> v12): I210_W 0.53 -> 0.56, I405_S 0.53 -> 0.75, I5_N 0.44 -> 0.59, I405_N 0.27 -> 0.71.
- **D7_I405_N regression.**
  - Onset sites: truth 351 (the end) 307, 349 71; v11 349 0.84; v12 351 0.69.
  - The CNN hits generated 349 onsets at 0.81 (v11), but generated 351 onsets only at 0.34 (it picks 349 on 442 of 706).
  - In real 351 onsets, 349 is ALSO queued at T+30 (median v/vf 0.26, with 350 free and 351 at 0.51). Both heads start together.
  - v12 queues only the end (349 stays at 1.0); v11 queued only 349.
  - A single-site share cannot see this; the CNN metric does.

## Step 3: learning the onset override on generated runs, redone with v11 / v12mix (2026-09-28)
v12mix = v12 runs, with D7_I405_N from v11. Same LambdaRank as before (analysis/gen_gate_train.py; GEN_ONSET_DIR / GEN_RUN_DIR). Lab windows 331, CNN pick top-1 0.861.

| training set | CNN only | + run shift | + shuffled (placebo) |
|---|---|---|---|
| ghb (earlier) | 0.773 | 0.773 (fixed 8, broken 37) | 0.804 |
| v11 | 0.662 (fixed 6, broken 72) | 0.742 (fixed 9.3, broken 48.7) | 0.676 (fixed 5.7, broken 67) |
| v12mix | 0.728 (fixed 2, broken 46) | 0.775 (fixed 8.7, broken 37) | 0.740 (fixed 5, broken 45) |
| v11, CNN reliability matched to 0.86 | 0.840 (0 / 7) | 0.855 (2.7 / 4.7) | 0.848 (0 / 4.3) |
| v12mix, CNN reliability matched to 0.86 | 0.852 (0 / 3) | 0.856 (1.0 / 2.7) | 0.852 (0 / 3) |

(Fixed / broken are per seed.)
- MATCH_CNN resamples the generated windows by group so the CNN's pick is right on 86% of them, as on real data.
- Result: no variant beats the CNN's own pick. The run-shift evidence fixes a little more than the placebo (1-3 windows per seed) but breaks as many.
- Matching the reliability removes most of the damage but leaves no gain.

## Pretraining with v12mix generated runs (Ada, 2026-09-29): the original use, redone
Same procedure as ghb (occ_lab/run_gen5.sh): leap_t2 --gen work/genruns_v12mix, share 0.5 decaying to 0, 6,000 steps, seeds 20/21.
- g12 skips D7_I10_E and D7_I10_W, as ghb did. g12all uses all panels.
- Evaluation: occ_lab/eval_g12.py. 331 lab onset windows, s20/s21 averaged; difference to ctrl with window-level SE.

| mode | metric | ctrl | ghb | g12 | g12all |
|---|---|---|---|---|---|
| none (no daily adaptation) | IoU | 0.7915 | -0.0028 (0.0074) | -0.0030 (0.0070) | **+0.0141** (0.0079, t 1.8) |
| none | top-1 | 0.8595 | -0.0030 | -0.0060 | +0.0045 |
| none | AUC | 0.9731 | +0.0027 (t 1.4) | +0.0038 (t 2.4) | **+0.0079** (t 3.5) |
| sup+rec (production) | IoU | 0.8383 | +0.0058 (0.0081) | -0.0069 (0.0070) | +0.0005 (0.0077) |
| sup+rec | top-1 | 0.9079 | +0.0076 | **-0.0121** (t -2.3) | -0.0060 |
| sup+rec | AUC | 0.9851 | +0.0030 (t 2.0) | +0.0021 (t 1.5) | +0.0039 (t 2.4) |

- Window-weighted IoU (sup+rec):
  - val: ctrl 0.8467, ghb 0.8499, g12 0.8265, g12all 0.8423
  - pri: ctrl 0.8295, ghb 0.8382, g12 0.8366, g12all 0.8351
- Per panel (sup+rec IoU, val + pri, difference to ctrl):
  - g12all: D7_I10_W +0.231 (SE 0.061, n 11); D12_I5_N -0.040 (0.015)
  - g12: D12_I5_N -0.046 (0.020)
  - Everything else is within about 1.5 SE.

**Reading.**
- **Without the daily adaptation**, the generated runs help onset, and more so with the better generator. Onset AUC vs ctrl: ghb +0.0027 -> g12 +0.0038 -> g12all +0.0079 (t 3.5); IoU g12all +0.014 (t 1.8). Adding D7_I10_E and D7_I10_W (g12all) adds to it.
- **With the production daily adaptation**, the gains shrink or reverse (g12 top-1 -0.012, t -2.3).
- This is the same generated-pretraining x adaptation interference seen on ongoing earlier (none +0.0030 / sup+rec -0.0027). The adaptation settings (30 steps, lr 1e-4) were tuned for ctrl.

## Adaptation strength x (ctrl, g12all) (Ada, 2026-09-29; occ_lab/run_adapt.sh, occ_lab/eval_adapt.py)
leap_t2.py lab patch (backup leap_t2.py.pre_adapt): env AD_STEPS, AD_LR, AD_PARAMS (all / blocks / inp, always + rec), AD_ONLY=1 (sup+rec only), AD_SUFFIX. Everything is sup+rec, s20/s21 averaged per window.

Onset IoU, g12all - ctrl at the same setting (SE, window level):

| setting | val ctrl | val g12all | val diff | pri ctrl | pri g12all | pri diff |
|---|---|---|---|---|---|---|
| production (30 steps, 1e-4, all) | 0.8467 | 0.8423 | -0.0044 (0.0114) | 0.8295 | 0.8351 | +0.0056 (0.0102) |
| rerun of production | 0.8461 | 0.8412 | -0.0049 (0.0112) | 0.8352 | 0.8334 | -0.0018 (0.0101) |
| 10 steps | 0.8489 | 0.8358 | -0.0131 (0.0118) | 0.8216 | 0.8425 | **+0.0210 (0.0092)** |
| 60 steps | 0.8387 | 0.8325 | -0.0062 | 0.8252 | 0.8062 | -0.0191 |
| lr 3e-5 | 0.8448 | 0.8389 | -0.0059 | 0.8314 | 0.8315 | +0.0001 |
| lr 3e-4 | 0.8271 | 0.8058 | -0.0213 | 0.8164 | 0.7940 | -0.0225 |
| blocks + rec only | 0.8410 | 0.8306 | -0.0104 | 0.8207 | 0.8283 | +0.0077 |
| inp + rec only | 0.8063 | 0.8144 | +0.0080 | 0.8266 | 0.8255 | -0.0011 |

- Against production ctrl, 10-step g12all is +0.0130 on private (SE 0.0105, t 1.2) and -0.0109 on val.
- **Rerun noise.** The identical production setting rerun moves ctrl private by +0.0056 and g12all - ctrl from +0.0056 to -0.0018. The adaptation is non-deterministic at the size of the effects. The window SE does not contain this.
- Ongoing is always -0.002 to -0.004 (t -3 to -7). Not the target for now.
- **Per panel, onset IoU private** (g12all - ctrl): the signs are stable across the settings.
  - D7_I10_W: +0.24 to +0.39 (n small).
  - D7_I210_W: +0.02 to +0.08 private (val mostly negative).
  - D7_I405_S: +0.03 to +0.07 private (val -0.03 to -0.07).
  - D7_I405_N: -0.03 to -0.10 (every setting).
  - D12_I5_N: val -0.06 to -0.14.
  - D12_I5_S: val positive, private negative.
- Opposite signs between the two months on I405_S, I5_S and I210_W. Consistent losses on I405_N and I5_N.

## Replay in the daily adaptation and constant co-training (Ada, 2026-09-29)
leap_t2.py patch (backup leap_t2.py.pre_cotrain):
- `--gen-dom`: a learned per-channel bias after the input layer on generated batches; inference and real batches go without it.
- Lab env: AD_REPLAY (share of adaptation steps that take a generated batch), AD_REPLAY_DIR, AD_REPLAY_ON (onset share among replayed windows), AD_REPLAY_DOM.
- Runs: occ_lab/run_cotrain.sh. Evaluation: occ_lab/eval_cotrain.py. Every entry = 2 seeds x 2 adaptation runs; the reference is ctrl production, prod + rerun.

Onset IoU, difference to ctrl (SE):

| entry | val | pri | both |
|---|---|---|---|
| ctrl (0.8464 / 0.8324 / 0.8395) | 0 | 0 | 0 |
| g12all (share 0.5 -> 0) | -0.0046 (0.0113) | +0.0019 (0.0099) | -0.0014 |
| co-train 0.10 + dom | +0.0022 (0.0116) | +0.0029 (0.0079) | +0.0025 (0.0071) |
| co-train 0.25 + dom | -0.0097 | +0.0021 | -0.0039 |
| co-train 0.25 (no dom) | -0.0073 | +0.0012 | -0.0032 |
| any replay with AD_REPLAY_ON=0.5 (ctrl, g12all, ct25d; rho 0.3 / 0.5) | about -0.34 to -0.43 | about -0.34 to -0.43 | collapse |

- Top-1: co-train + dom +0.0074 val, -0.0015 pri; replay -0.08 to -0.12.
- **Replay collapse.** Not an input-format bug: gen and lab both build x with cnn.make_x.
  - Half of the replayed windows were generated ONSET windows, whose sites differ from the real ones (real-CNN top-1 on them about 0.4).
  - Over a month of cumulative daily adaptation the model unlearns the real onset sites. D12_I5_N ctrl, mean p on the truly queued onset cells: 0.85 -> 0.33 (val), 0.68 -> 0.38 (pri). The overall p level is unchanged.
  - The pretraining itself sees only about 5% onset windows among the generated ones.
- **Per panel, co-train 0.10 + dom** (val / pri): I10_W +0.21 / +0.32; I405_N +0.00 / -0.03 (g12all -0.06); I5_N -0.02 / 0.00 (g12all -0.06 / -0.02); I5_S 0.00 / -0.03; I405_S +0.01 / 0.00; I210_W -0.03 / +0.01. More balanced than g12all.
- **Fix, running** (occ_lab/run_replay2.sh):
  - replay with generated ONGOING windows only (AD_REPLAY_ON=0), rho 0.1 / 0.3, on ctrl and on co-train 0.10 + dom;
  - co-train 0.05 + dom.
- **Fix results** (occ_lab/run_replay2.sh). Onset IoU difference to ctrl, val / pri:

  | entry | val | pri | top-1 val / pri |
  |---|---|---|---|
  | co-train 0.05 + dom | -0.011 (0.010) | +0.004 (0.008) | +0.004 / -0.005 |
  | ctrl + replay ongoing-only 0.1 | -0.090 | -0.082 | -0.019 / +0.006 |
  | ctrl + replay ongoing-only 0.3 | -0.227 | -0.217 | |
  | co-train 0.10 + dom + replay ongoing-only 0.1 | -0.117 | -0.112 | |
  | co-train 0.10 + dom + replay ongoing-only 0.3 | -0.262 | -0.254 | |

  - Replay in the daily adaptation hurts in every form. IoU falls much more than top-1: the probability scale on the true onset cells shifts (the ratio-rule IoU is sensitive to it).
  - A cumulative month of constant-lr adaptation on generated windows is harmful even with ongoing windows only.
  - Likely factors, not isolated: truth-built generated inputs (no missing cells, ok = 1) against the observed-input pseudo-label windows; no LR decay or early stopping in the adaptation.
- **Best so far: co-train 0.10 + dom**, private +0.0029 (SE 0.0079). Not above the SE.
  - Its private losses are I405_N -0.028 and I5_S -0.027; its gains are I10_W +0.32 and I10_E +0.03.
  - Removing the two losses alone would add about +0.007 to private: the generator-fidelity fixes for those panels (the I405_N 349 + 351 co-onset; I5_S / I5_N over-active heads) are the lever.

## Generator fix for the losing panels (2026-09-29)
**Head-share calibration** (analysis/ctm_share_fit.py). ap per head so that each head's active-day share (h queued, h+1 not queued, >= 2 slots of 05-21 h) matches the truth; train only, 6 rounds on half the days.
- The main heads already match in v12: I5_N 105 0.80 / 0.79, 114 0.72 / 0.76, 239 0.91 / 0.93. "Over-active" is no longer the issue.
- The gaps are heads active in the truth on 0.3-0.8 of days but never in the replay (I5_S 10, 12, 58, 64, 89, 95-99; I5_N 82-101, 199-233). ap down to the 0.85 clip does not create them.
- Site overlap barely moved: I5_N 0.66 -> 0.72, I5_S 0.61, I405_N 0.77.
- **D7_I405_N**: end (z351) onsets with z349 also queued at T+30: truth 0.81, replay 0.00. In the replay the end queue spreads upstream; in the truth z350 stays free between two separate queues.

**Intermediate speed right downstream of a queue head** (h+1 in 0.6-0.9 vf while h is queued), truth vs v12:
- I5_S 0.36 vs 0.11
- I405_S 0.20 vs 0.03
- I5_N 0.23 vs 0.18
- I405_N 0.29 vs 0.28
- I210_W 0.06 vs 0.05

**HB_HEAD_FREE=0.6** (ctm_gen._hidden_bottlenecks and ctm_type_rule: head = h+1 not queued instead of free):
- It adds merge heads on I5_S (12, 23, 58, 64, 71, 75, 76, 78, 84, 90, 95, 97 with ratios 0.87-0.99).
- The never-active heads stay at 0.00. Many truth "heads" with h+1 at 0.6-0.9 look like queue-recovery or partial-queue phases (the downstream part clearing first), not separate bottlenecks.
- Share gap after the fit: I5_S 9.15 -> 7.84; I5_N 6.38 -> 5.99 (238 now 0.63 vs truth 0.82, was 0.32).

v13 = these three panels regenerated with the fitted rule (RULE_OUT=ctm_rule_share_*, HB_HEAD_FREE=0.6). v13mix = v12mix with them replaced. Next: co-train 0.10 + dom on v13mix, compared with ct10d on v12mix.

### Panel-limited v13 variants (co-train 0.10 + dom, 2 seeds x 2 labs; occ_lab/run_v13.sh)
Onset IoU, difference to ctrl (SE):

| variant | val | pri | both | top-1 both |
|---|---|---|---|---|
| ct10d (v12mix) | +0.0022 (0.0116) | +0.0029 (0.0079) | +0.0025 | +0.0030 |
| v13 all three panels | +0.0035 | +0.0037 (0.0079) | +0.0036 | +0.0038 |
| v13 I405_N only | +0.0061 | +0.0021 | +0.0042 | +0.0068 |
| v13 I5_S only | +0.0042 | +0.0024 | +0.0033 | +0.0053 |
| v13 I5_N only | +0.0006 | -0.0026 | -0.0010 | +0.0076 |

- **Panels whose data did not change still move by up to about 0.02.** I5_S private: -0.027 (base), -0.009 (I405_N only), -0.015 (I5_N only). Joint training over all panels propagates any data change; per-panel differences of about 0.02 are training noise.
- **The only consistent per-panel loss is D7_I405_N private**, about -0.035 whatever changed (-0.028 / -0.035 / -0.035 / -0.043; the I405_N fix did not help). The end + z349 co-onset structure is still not reproduced (replay co-queue 0.00 vs truth 0.81).
- All variants are +0.002 to +0.004 on private: a small consistent positive, below the SE.

## Onset composition with the w192 generated-data model: submission CA (2026-09-29)
- **The production onset CNN is not the model the generator experiments used.** BB's second onset CNN (sub_t2pseudo) is t2_cnn_lov_rec_s20: width 64, no SSL. All generated-data comparisons used leap_t2's w192 SSL family (ctrl_s20/s21).
- Same lab windows and adaptation (lab cache pseudo_s20), single member, window-weighted onset IoU val / pri:

  | model | val / pri |
  |---|---|
  | w64 production | 0.7806 / 0.8351 |
  | w192 ctrl | 0.8464 / 0.8324 |
  | w192 + generated (I405_N ctrl) | 0.8485 / 0.8396 |
  | 0.5 w64 + 0.5 w192gen | 0.8453 / 0.8743 |
  | 0.25 w64 + 0.75 w192gen | 0.8807 / 0.8773 |

  The per-panel losses of the w192 family against w64 (I405_S pri 0.777 -> 0.523, I210_W pri 0.978 -> 0.876) come from the w192 SSL family, not from the generated data.
- **Composition grid** (occ_lab/onset_compose.py, simplex step 0.05 over H, ad9b, ps, G, ST; window-weighted):
  - BB members with no set decoder: 0.8385 / 0.8624.
  - Best by the two-month mean: H 0.20, ps 0.35, G 0.40, ST 0.05 -> 0.8877 / 0.9058. The neighbourhood mean is 0.8813 / 0.8988.
- **Set decoder** (onset_setdec_fit.py with ONSET_W / ONSET_G / SETDEC_TAG, cross-month; the BB reference reproduces note 154 exactly). Window-weighted val / pri:
  - BB + set decoder: 0.8558 / 0.9065
  - new (H .15, ps .35, G .40, ST .10), no set decoder: 0.8879 / 0.9034
  - new + refitted set decoder: 0.8859 / 0.9061
  - with G0 (no generated data): 0.8913 / 0.8996
  - In family means, the new composition is 0.8670 / 0.8390 against BB's 0.8315 / 0.9020 (I10_W private, 3 windows).
- Official path: scripts/t2_onset_blend_predict.py --g-dirs / --g-alt-dirs / --w-g. The w192 models were converted by work/leap_to_t2net.py and adapted with t2_pseudo_adapt_official.py (T2CNN_WIDTH=192) into work/sub_t2leap_*. A BB rebuild through the modified script changed 0 cells.
- **CA** = BB with the onset = 0.15 H + 0.35 ps + 0.40 G + 0.10 STID, no set decoder. 45 cells differ from BB (validation 28 cells in 12 windows, private 17 in 8).
  - **Public 0.88112 against BB 0.88697: -0.00585.** Practically every changed validation cell hurt (one onset cell is about 0.00025).
  - The lab onset evaluation again ranked a mixture the wrong way, like AO (+0.033 local -> -0.0046 LB; session memory). The lab windows are official-LIKE windows built from the scored months, not the 40 official onset windows per split.

## Run-adaptive onset weights (Hedge over members from earlier days of the run) - and the official windows' days
Lab test (occ_lab/onset_online_w.py): members' onset IoU on the same panel-month's lab windows of earlier days sets Hedge weights; p = (1 - lam) BB + lam online.
- Cross-month selection picks eta 4, lam 0.75 in both directions: val 0.9041 / pri 0.9125, window-weighted. BB + set decoder is 0.8558 / 0.9065.
- **But the official onset windows lie on days 0-4 of each scored month** (D7_I10_W private on days 2, 4, 6, 8), while the lab onset windows start on day 5.
  - No earlier-day onset outcome exists for the official windows, so the method cannot be applied. The CB build weighted 0 windows (it equalled BB without the set decoder, 19 cells) and was NOT submitted. Table: scripts/onset_online_table.py.
- **This also explains CA.**
  - On lab windows (day >= 5) the per-day adapted CNNs (ps, G, ad9b) have 5+ days of adaptation.
  - On the official windows they have 0-4 days, so the non-adaptive H / STID are relatively stronger.
  - CA moved weight from H / STID to the adapted CNNs, which was right for day-5+ windows and wrong for days 0-4.
  - The lab onset evaluation measures a different phase of the run than the one scored; possibly also AO's failure.
- Consequence: an onset evaluation must reproduce the early-days condition (adaptation limited to 0-4 past days).

## Onset under the official windows' condition (0-4 days of adaptation), local (2026-09-29)
**Members.** occ_lab/pseudo_adapt.py --hist K: reset per lab day and adapt only over the K days before it, with the official recipe (day by day; the window's own day before its first origin).
- Caches: pseudo_h{K}_supprec (BB's ps member) and pseudo_h{K}_rec (9b recipe). The unrestricted rerun reproduces pseudo_s20: 0.7344 / 0.7475 vs 0.7375 / 0.7457.
- ps alone, family mean val / pri: K0 0.667 / 0.643, K1 0.694 / 0.676, K2 0.704 / 0.694, K3 0.708 / 0.705, K4 0.721 / 0.715; unrestricted 0.734 / 0.748. The adapted CNNs are clearly weaker on the official windows' days.

**Weights** (occ_lab/onset_early_eval.py; window IoU averaged over K = 0..4):
- BB weights: 0.8257 / 0.8434 (full adaptation: 0.8385 / 0.8624).
- Cross-month 0.1 grid: +0.003 on pri (SE 0.006), +0.008 on val (SE 0.009); both directions move the ad9b share to ps. Within noise: BB's weights are about optimal for these four members.

**Set decoder** (occ_lab/setdec_early_check.py; bottleneck_cls ONSET_EARLY_K), mean over K:
- Production-like, fitted on full-adaptation month A and applied to early month B: private 0.8434 -> 0.8658 (+0.0224, SE 0.0085), validation 0.8257 -> 0.8405 (+0.0148, SE 0.0085).
- Refitted on early data: +0.0210 / +0.0113. No better.
- BB's set decoder is validated under the official condition: about half its full-condition lab gain, and consistent with its public +0.00062. BB is near optimal within the current members.
- The w192 generated-data member under this condition needs the Ada model files (not yet done).

## w192 ctrl (no generated data) as an onset member under the official condition (2026-09-30)
**Setup.** Ada: occ_lab/pseudo_adapt_hist.py (= local pseudo_adapt.py with --hist) run on leap_ctrl_s20/21 converted to T2Net (T2CNN_WIDTH=192), K = 0..4, sup+rec; caches pseudo_h{K}_ctrl_s{20,21}. 10 runs, under 1 h on 4 GPUs.

**Single members, family mean val / pri.**
- w192 ctrl at K0 (s20): 0.744 / 0.716, against w64 ps 0.667 / 0.643. With little adaptation the w192 SSL family is clearly stronger.
- Window-weighted, averaged over K: H 0.793 / 0.809; ad 0.751 / 0.747; ps 0.753 / 0.777; **G0 0.829 / 0.808**; ST 0.805 / 0.834.

**Weights, cross-month** (onset_early_eval.py, 0.1 grid):
- Chosen on val (G0 0.7) -> pri -0.007 (SE 0.011).
- Chosen on pri (H .5, ps .3, G0 .1) -> val +0.008 (SE 0.008).
- The months disagree again.

**Pre-specified config: BB with the ad9b slot given to G0** (H .35, ps .175, G0 .175, ST .30); setdec_early_check.py with ONSET_G=G0, ONSET_W=0.35,0,0.175,0.175,0.30; mean over K.

| variant | val | pri |
|---|---|---|
| base, no decoder | 0.8291 | 0.8493 |
| decoder fitted on full-adaptation data (production-like) | 0.8329 | 0.8602 |
| **decoder refitted on early-condition data** | **0.8480** | **0.8782** |
| BB production-like, for reference | 0.8405 | 0.8658 |

- Against the BB reference: +0.0075 val, +0.0124 pri.
- Per K, the refitted variant beats BB on private at every K (+0.002 to +0.021) and on validation at 3 of 5 (worse at K0).
- Caveats: the decoder-condition variant was chosen between two after seeing the results; the private difference is about 1 SE.
- **CC submitted (2026-09-30)**: BB with the onset ad9b slot -> w192 ctrl (0.35 H + 0.175 ps + 0.175 w192 + 0.30 STID). Set decoder refitted on the official-condition data (occ_lab/setdec_fit_early.py, both months x K 0..4 pooled; models onset_*_ge; threshold 0.02).
  - 45 cells differ from BB: validation 23 in 10 windows, private 22 in 7.
  - **Public 0.88428 against BB 0.88697: -0.0027.** CA was -0.0058.
  - The early-condition lab (+0.008 val / +0.012 pri, about 1 SE) did not transfer either. BB stays best.

## Onset STID: size and regularization (2026-09-30, local, seed 0 each)
**Size** (--d / --ds / --blocks / --dropout / --size-tag; loaders read d/ds/blocks from the checkpoint). ES bce(T+30) on the train tail:
- S 0.30M params: 0.01539
- M 0.59M: 0.01509
- L 2.9M: 0.01560
- L + dropout 0.1: 0.01580

Every size drives the train loss to 0.0000 by step 4000 (1,840 train windows, 7,251 positive cells). The model is data-limited, not capacity-limited.

**Regularization, one factor per run** (occ_lab/stid_reg_eval.py): STID alone, and inside BB's blend with the day-adapted members at the official condition (K 0..4); window-weighted val / pri, difference to the retrained S seed 0.

| variant | ES bce | alone val / pri | in blend val / pri |
|---|---|---|---|
| S (baseline) | 0.01539 | 0.7936 / 0.8122 | 0.8257 / 0.8394 |
| per-link identity dropout 0.3 | 0.01516 | +0.006 / -0.007 | -0.007 / -0.001 |
| per-link identity dropout 0.5 | 0.01564 | -0.022 / -0.019 | -0.012 / -0.002 |
| identity-consistency 0.5 | 0.02315 | -0.061 / -0.115 | -0.015 / -0.001 |
| identity-consistency 2 | 0.03237 | -0.153 / -0.183 | -0.016 / -0.019 |
| time-of-day smoothness 1.0 | **0.01315** | -0.003 / -0.008 | -0.005 / -0.004 |
| weight decay 1e-2 | 0.01498 | -0.001 / -0.010 | -0.008 / **+0.007** (SE 0.004) |
| production stid30 (4 seeds) | - | +0.012 / +0.022 | 0.000 / +0.004 |

- **Restricting the link identity hurts on the scored months.** The identity carries information that travels across runs; it is not only train-run memory.
- Time-of-day smoothing improves the same-run ES (-15%) but not the scored months. Same-run ES cannot rank these.
- Weight decay 1e-2 is mixed across the months (val -0.008, pri +0.007).
- Seed averaging (4 seeds) moves the STID alone by more than any regularizer.

Less identity dropout (--emb-drop; production is batch-level 0.10), seed 0, same evaluation:

| emb-drop | ES bce | alone val / pri | in blend val / pri |
|---|---|---|---|
| 0.05 | 0.01449 | +0.007 / -0.007 | -0.005 / +0.005 (SE 0.004) |
| 0.0 | 0.01513 | -0.024 / +0.013 | -0.011 / +0.006 (SE 0.006) |

- Less dropout moves the blend up on private and down on validation. That is the same month split as weight decay, within one seed's noise (+-0.01). No consistent gain.

**STID size M in the lab** (seed 0; alone val / pri against S seed 0):
- M: +0.008 / -0.005; in the blend (official condition) +0.003 / -0.001.
- L: -0.019 / +0.013; in the blend -0.008 / +0.005.
- L + dropout: -0.021 / +0.014; in the blend -0.014 / +0.007.

**CD submitted (2026-09-30)**: BB with the onset STID replaced by size M (d192, ds32, 3 blocks; seeds 0-3, ES 0.0151 / 0.0142 / 0.0151 / 0.0148). Weights and set decoder as BB.
- 18 cells differ from BB (validation 14 in 6 windows, private 4 in 3).
- **Public 0.88263 against BB 0.88697: -0.0043.**
- BB remains the best.
