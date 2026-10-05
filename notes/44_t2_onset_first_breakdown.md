# 44 — Task 2 onset as a first-breakdown problem: geometry of the truth sets and where M's onset loss comes from

Date: 2026-09-15 · Script: `analysis/t2_onset_geometry.py` (log `work/t2_onset_geometry.log`)

Prompted by the user's proposal: treat onset as "which of the candidate bottlenecks breaks down first" (competing
risks), not as per-link classification. Step 1 of the proposal was a geometry EDA of the T+30 truth sets. It was run
on first-morning onsets: train truth (1722 events) and the blank-aware official-like windows (val 169, priv 162 —
M's exact evaluation set).

## Geometry of the truth sets

| | cells 1 / 2 / 3 / 4 / 5 / >5 | one run (strict / gaps ≤ 1 merged) | ≥ 3 runs | largest-run share = oracle IoU |
|---|---|---|---|---|
| train | 0.13 / 0.26 / 0.20 / 0.17 / 0.11 / 0.13 | 0.47 / 0.86 | 0.12 | 0.78 |
| validation | 0.34 / 0.25 / 0.18 / 0.14 / 0.05 / 0.03 | 0.59 / 0.79 | 0.12 | 0.80 |
| private | 0.35 / 0.25 / 0.19 / 0.11 / 0.07 / 0.04 | 0.56 / 0.81 | 0.14 | 0.80 |

- The sets are small: 1–2 cells in about 60% of the scored events. They are a single contiguous run only in
  56–59% of them. **Predicting only the largest run would cap onset IoU at 0.80, below M's actual 0.81–0.83.** A
  strict "origin → queue extent" single-run structure is therefore not viable on its own (consistent with note 34:
  forcing contiguity cost −0.05…−0.07). Taking k links from one end of the largest run is worse still (≤ 0.63).
- **The run ends of the scored months fall within ±1 link of train's 5 most frequent ends in 86–100% of events**
  (D7_I10_W, with only 3–8 events, is the exception). The candidate bottlenecks are the same across runs; which one
  fires first is what changes. This is exactly the proposal's competing-risk picture.

## Where M's onset loss comes from

M's onset rule (0.7·H + 0.3·CNN×3, ratio rule), official IoU with ineligible predicted cells dropped:

| | mean IoU | exact (IoU = 1) | IoU = 0 | partial | loss share: IoU = 0 / partial |
|---|---|---|---|---|---|
| validation | 0.810 | 0.75 | 0.12 | 0.14 | **0.62** / 0.38 |
| private | 0.833 | 0.77 | 0.12 | 0.10 | **0.74** / 0.26 |

- The predicted and true counts match on average (2.3–2.4 cells). The partial windows are mostly count errors: the
  prediction is inside the truth (35%) or contains it (52–59%); "both wrong" carries only 2–6% of the loss.
- **Most of the onset loss (62–74%) is the 12% of windows where M picks the wrong bottleneck.** Fixing the location
  choice is the lever: it is worth up to about +0.12 onset IoU (≈ +0.018 total) if solved completely.

## Mapping the proposal onto what has been tested

| Proposal step | Status |
|---|---|
| 1 geometry EDA | this note: single-bottleneck structure too restrictive; location is the loss |
| 2 ranking target inside the window | tested in note 39 (lambdarank): −0.006…+0.005, top-1/top-3 unchanged |
| 3 competing-risk / hazard training over lead times | **new** → `analysis/t2_onset_hazard.py` ("lead" variant) |
| 4 empirical breakdown margin + relative rank features | discharge-capacity margins were negative (notes 21, 22, 27); **within-window ranks and an empirical pre-breakdown flow reference are new** → same script ("rank" variant) |
| 5 origin → extent two-stage | capped at 0.80 by the geometry above (and note 34) |
| 6 calibration before the expected-IoU rule | tested in notes 31 (c scale) and 43 (recalibration): no gain |

## Hazard / rank test, run 1 (`analysis/t2_onset_hazard.py`, log `work/t2_onset_hazard.log`)

The training data is small: about 290 earliest-onset windows per panel, and the whole run takes 95 s. Run 1 used
2 bags and only 25% of the extra-lead windows (leads 4, 5, 7, 8).

| Δ onset vs paired base (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| rank, alone · blend | +0.000 · +0.012 | +0.026 · +0.004 | −0.012 · −0.007 | −0.004 · −0.003 |
| lead, alone · blend | −0.009 · −0.006 | +0.015 · +0.002 | +0.001 · −0.009 | +0.041 · +0.043 (2/3) |
| rank+lead, alone · blend | −0.001 · +0.000 | +0.030 · +0.013 | −0.011 · −0.012 | +0.020 · +0.021 |

- The signs flip between conditions and panels, and the share of windows missed entirely (IoU = 0) does not fall
  (0.11–0.14 in every variant). The rank features take 11% of the gain; the lead feature takes 0.1%.
- Onset IoU moves by up to about 0.08 with the prior-dropout draw alone (note 22), so one run cannot decide. Run 2
  uses all extra-lead windows (leads 2–10), 5 bags, and 3 repeats with different seeds.

## Run 2: 3 seeded repeats, 5 bags, all extra leads (logs `work/t2_onset_hazard_s{0,1,2}.log`, `work/t2_onset_hazard_agg.log`)

About 290 windows per extra lead per panel, so the training data is roughly 9× the base.

| mean Δ vs paired base [min, max] (repeats > 0) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| rank, alone | −0.011 (0/3) | −0.012 (0/3) | +0.004 (2/3) | −0.002 (1/3) |
| rank, blend | +0.004 (2/3) | +0.000 (1/3) | +0.006 (3/3) | −0.007 (0/3) |
| lead, alone | **−0.076 (0/3)** | **−0.040 (0/3)** | **−0.031 (0/3)** | **−0.038 (0/3)** |
| lead, blend | −0.009 (1/3) | +0.001 (1/3) | −0.001 (2/3) | +0.006 [−0.009, +0.031] (1/3) |
| rank+lead, blend | −0.016 (0/3) | −0.000 (2/3) | −0.003 (1/3) | +0.011 [−0.015, +0.031] (2/3) |

IoU = 0 share, alone (val / priv): base 0.124 / 0.134; rank 0.122 / 0.138; lead 0.152 / 0.154. In the blend every
variant stays at 0.11–0.13.

- **The hazard-style training (lead) makes onset clearly worse alone in all 4 conditions and all 3 repeats.** It
  also raises the share of windows missed entirely. The lead feature gets 0.1–0.2% of the gain: the model ignores it,
  so first breakdowns at other leads only blur the T+30-specific relation between the history and the location.
- **The within-window rank / empirical-margin features are noise-level** (signs flip across conditions), and they
  do not reduce the IoU = 0 share.
- **Conclusion:** the proposal's framing is right about *where* the loss sits: the wrong bottleneck in about 12% of
  windows. None of its new ingredients helps choose the bottleneck. This agrees with notes 17, 21 and 39: which
  candidate fires first is decided by demand differences under 1% that the history does not reveal. Onset stays
  as in M.

## Second proposal: pre-breakdown instability features (`analysis/t2_onset_instability.py`, log `work/t2_onset_instability.log`)

Motivation: Tak & Yeo 2026 (TR-C) detect breakdowns early from short-term speed fluctuation combined with an
upstream/downstream imbalance, a "Dynamic Instability Score". Its exact formula was not accessible (403), so the
features follow that idea: 18 per-link features, namely the v/vf std over 15/30/60 min, first and second
differences, the largest drop, neighbour differences and gradients with their change, the neighbours' std,
`std × |gradient|` (DIS-like) and its rank, and the q/cap std and gradient. In the train split
`obs_speed == true_speed`, so train and scoring features are alike. The history-only rule is kept.

**Diagnostic, per-window AUC for "link in the T+30 set":**
- Every single feature scores 0.40–0.59, against 0.95 (candidates only) and 0.99 (all links) for M's probability.
- A few point the same way in train, validation and private:
  - the q/cap gradient: 0.55–0.59;
  - the v/vf std over 30–60 min: 0.52–0.58;
  - the neighbours' std: 0.53–0.58;
  - the q/cap std, with low values pointing to onset: 0.40–0.50.
- This confirms F25 (note 09): there is almost no speed precursor at T−5.

**Paired LightGBM, run 1** (H's configuration; 3 seeds × 5 bags; mean Δ, with the number of seeds > 0 in brackets):

| Δ onset (+18 features) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| alone | +0.001 (1/3) | **+0.012 (3/3)** | **+0.016 (3/3)** | **+0.017 (3/3)** |
| blend 0.7/0.3 | −0.005 (1/3) | −0.002 (1/3) | +0.010 (2/3) | **+0.014 (3/3)** |

- Alone, it improves consistently in 3 of the 4 conditions. In the blend the validation gain disappears, so the
  mean over the 4 conditions is ≈ +0.004 onset (≈ +0.0006 total). The new features take 8% of the gain.
- Run 2 adds a train-AUC-selected subset (q/cap gradient and std, v/vf std over 30/60 min, the neighbours' std, the
  DIS-like product) and a blend weight of 0.85.

**Run 2** (log `work/t2_onset_instability_v2.log`; blends are compared with base in M's 0.7 blend):

| Δ onset | val d1–14 | val all | priv d1–14 | priv all | mean of 4 |
|---|---|---|---|---|---|
| all 18, blend 0.7 | −0.005 | −0.002 | +0.010 | +0.014 | +0.004 |
| all 18, blend 0.85 | −0.005 | −0.001 | +0.000 | +0.009 | +0.001 |
| subset 7, alone | −0.007 | +0.002 | +0.011 | +0.013 | +0.005 |
| subset 7, blend 0.7 | +0.001 | +0.002 | +0.000 | +0.008 | +0.003 |
| subset 7, blend 0.85 | −0.008 | −0.005 | −0.006 | +0.005 | −0.004 |
| base, blend 0.85 vs 0.7 | −0.004 | −0.010 | −0.004 | −0.002 | |

- Raising the LightGBM weight does not carry the stand-alone gain into the blend; 0.85 is worse for every variant.
- The best blend variant gains +0.003…+0.004 onset, about +0.0005 total. It is not consistent across the two runs:
  validation ≤ 0, private > 0. **Not adopted.**
- **Conclusion for the second proposal:** the instability family (fluctuation × spatial imbalance) is the one
  idea that moved the stand-alone LightGBM consistently. The synthetic data carries almost no speed precursor at
  T−5, though, and the CNN in the blend already supplies what little there is. Its multi-horizon hazard heads
  equal the lead test above (negative), and its multi-task part equals CNN v2 A (note 43, noise).
