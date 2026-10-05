# 52 — T2 ONGOING: the first error decomposition → the loss is missed queue GROWTH, and a per-window threshold is worth +0.08

Date: 2026-09-16 · Script: `analysis/t2_ongoing_error_anatomy.py` · Log: `work/t2_ongoing_error_anatomy.log` (10 s,
cached probabilities only)

Ongoing is half of S_queue and had never been decomposed: notes 44 and 51 analysed onset only, and with M's onset
near its information ceiling (note 51: even a perfect bottleneck choice gives 0.93, worth ≈ 0.897 total), **ongoing
is the only remaining route to a materially higher score.** M's ongoing rule: 0.5·LightGBM(I) + 0.5·CNN×3, τ 0.60,
six horizon steps T+5…T+30, scored on the blank-aware official-like windows (3351 val, 3374 priv).

| | validation | private |
|---|---|---|
| M ongoing IoU (macro / micro) | 0.8207 / 0.8792 | 0.8154 / 0.8661 |
| persistence (freeze the queue seen at T) | 0.6660 | 0.6644 |
| errors: false positive / false negative | 0.305 / **0.695** | 0.297 / **0.703** |
| per-window IoU p10 / p50 / p90 | 0.566 / 0.878 / 0.988 | 0.556 / 0.875 / 0.987 |
| worst 10% of windows hold | 0.34 of the loss | 0.34 of the loss |

**M under-predicts: 70% of the errors are missed queue cells.**

## 1. Where the errors sit, relative to the queue visible at T

| class | meaning | cells | share of all errors | error rate in the class |
|---|---|---|---|---|
| **grow** | not queued at T, queued in the horizon | 43k | **0.42 / 0.47** | **0.33 / 0.40** |
| stay | queued at T and still queued | 224k | 0.27 / 0.24 | 0.041 / 0.039 |
| dissolve | queued at T, cleared in the horizon | 37–40k | 0.17 / 0.18 | 0.154 / 0.170 |
| far | never queued | 1.63M | 0.13 / 0.12 | 0.003 |

- **The single largest bucket is queue growth**: of the cells the queue newly reaches, we miss a third to a
  half. Holding what is already there is nearly solved (4% error), and spurious queues far away are rare (0.3%).
- Dissolution is the second problem: 15–17% of the cells that clear are still predicted as queued.
- So ongoing is not a "where is the queue" problem but a **queue-boundary dynamics** problem: how far the tail
  extends upstream and how fast the head clears.

## 2. By horizon step and by distance to the queue edge

| | +5 | +10 | +15 | +20 | +25 | +30 |
|---|---|---|---|---|---|---|
| IoU (val) | 0.902 | 0.870 | 0.844 | 0.813 | 0.794 | 0.785 |
| IoU (priv) | 0.907 | 0.869 | 0.849 | 0.806 | 0.779 | 0.768 |
| share of errors | 0.10 | 0.13 | 0.15 | 0.18 | 0.21 | 0.23 |

The last two steps carry 43–45% of the loss. Distance to the nearest edge of the T queue: d = 0 → 0.22–0.23 of the
errors, d ≤ 2 → 0.43–0.46, d ≤ 4 → 0.58–0.60, **d ≥ 5 → 0.40**. Most of the loss is at and near the moving
boundary, but a substantial minority is further away (long extensions, or a second queue forming).

## 3. Ceilings (paired delta vs M, 95% bootstrap CI over windows)

| rule | validation | private |
|---|---|---|
| global τ sweep, best | **τ 0.50: +0.0118 [+0.009, +0.015]** | **τ 0.50: +0.0093 [+0.007, +0.012]** |
| τ per horizon step (chosen globally per step; all land at 0.48–0.54) | +0.0178 [+0.015, +0.021] | +0.0107 [+0.008, +0.013] |
| **ORACLE per-window τ** | **+0.0803 [+0.076, +0.084]** → IoU 0.901 | **+0.0808 [+0.077, +0.085]** → IoU 0.896 |
| ORACLE per-window blend weight α | +0.0525 | +0.0506 |
| global α sweep (best 0.4–0.5) | +0.000 | +0.000 |
| persistence | −0.1547 | −0.1511 |

- **τ 0.60 is not optimal; 0.50 is, in both splits, worth ≈ +0.010 ongoing (≈ +0.0015 total).** This matches the
  earlier threshold check (note 43, "≈ +0.001 total"), now with a CI that excludes zero on both splits.
- M's 0.5/0.5 blend weight is confirmed optimal globally (the α curve is flat at 0.4–0.5).
- **An oracle that knew the right threshold per window would gain +0.080 ongoing = +0.040 S_queue ≈ +0.012 total.**
  For comparison, note 51's oracle gate for onset was worth only +0.017…+0.028 onset (≈ +0.003 total), and onset
  has 169 windows per split against 3351 here. This is the largest headroom measured anywhere in T2 so far, and it
  needs no new base model.

## 4. What this changes

The previous conclusion "onset is information-limited, so T2 is stuck" was drawn from onset alone. Ongoing has:
- a clear physical target (boundary growth and dissolution, the last two horizon steps), and
- a large decision-side headroom (per-window threshold), on 20× more windows than onset.

Next: `analysis/t2_ongoing_tau_gate.py` measures how much of the +0.08 is learnable, by fitting τ-choosing rules on
one split and applying them to the other (a derivation/evaluation device; a production gate would be fitted on the
train run only). Rules: global, per panel, per queue-size and per probability-mass bucket, and a small LightGBM on
window features.
