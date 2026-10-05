# 16 — Leaderboard E = 0.82542; validation T2 keeps landing about 0.15 below our local estimates

Date: 2026-09-14 · Rebuild script: `scripts/run_submission_E.ps1`

## Submission
| ref | Content | Public LB | Δ |
|---|---|---|---|
| 56229889 | C': T1 M3 v1, T2 rules v2, T4 public_reference | 0.80099 | — |
| **56231027** | **E**: T1 mean of 3 CNNs, T2 LightGBM models (note 15), T4 public_reference | **0.82542** | **+0.0244** |

## Decomposition
- T1 ensemble: about +0.002 (note 07). That leaves T2 about +0.022, so ΔS_queue ≈ +0.075. Validation
  S_queue went from ≈ 0.56 to **≈ 0.635**.

## The systematic validation gap
| T2 method | Local Jan–Feb holdout | Validation (implied by the LB) | Gap |
|---|---|---|---|
| Persistence | ≈ 0.30 (train official windows) | 0.30 (official) | ≈ 0 |
| Rules v2 | 0.688 | ≈ 0.56 | −0.13 |
| LightGBM models | 0.837 | ≈ 0.635 | −0.20 |

Validation reached only about half of the model's local gain. The gap has the same sign for every
method, and it grows as the method becomes more precise. That pattern looks systematic (labels or
selection), not like one hard month.

## H15 tested and rejected: our queue cut is the organizer's
Script `analysis/t2_label_cut_test.py`, log `work/t2_label_cut_test.log`.

| Label cut | Structure holds (onset / ongoing) | Persistence S_queue | Rules S_queue |
|---|---|---|---|
| **0.6 × links free speed (ours)** | **0.975 / 1.000** | **0.307** (official 0.3017) | 0.707 |
| flat 63 km/h (0.6 × 105) | 0.850 / 0.900 | 0.372 | 0.666 |
| 0.55 × vf | 0.825 / 0.900 | 0.385 | 0.663 |
| 0.65 × vf | 0.875 / 1.000 | 0.302 | 0.712 |

Only our cut reproduces the official window structure almost exactly, and it also matches the official
persistence number. The label definition is not the gap.

## New leading hypothesis
- **H17**: *Validation and private are a different simulation run.* The rebuild regenerated them "from new
  random seeds", while train is unchanged. Recurrence priors learned on train transfer only partly to
  another run. Our models lean hardest on those priors (onset: `pq30_rel`, `pon30`; ongoing: `pq_h`,
  `pq_diff`), and that fits a gap that widens with precision. Supporting evidence: at validation onset
  windows the prior is less concentrated (top-3 share 0.355 against 0.45 on train; note 13).
  Test: compare where and when queues occur in the validation/private months (visible masked-layer
  cells, horizons excluded) with the train priors.

## Hypotheses to test next
- **H15**: *The organizer's queue cut differs from ours.* We use 0.6 × links.csv free speed (≈ 66–72
  km/h). A flat 105 km/h (the `lwr_mainline_topology.csv` value) would give ≈ 63 km/h. Test without a
  submission: official windows were selected with the organizer's labels, so the onset structure ("no
  queue in history, first queued at T+30") should look cleanest under the true cut.
- **H11**: Noise-free versus noisy labels near the cut. Precise models would suffer most, which fits
  the widening gap.
- **H16**: Scoring eligibility differs. `future_coverage` is 1.0 on every official window, so the
  organizer may score horizon cells that are ineligible in the release.
- **Diagnostic for tomorrow's slots**: E with every onset window left empty scores ongoing/2, which
  splits validation S_queue into its onset and ongoing parts.
