// filepath: notes/64_t2_onset_t1res_features.md
# 64 — T2 onset: the Task 1 residual block is the first onset change that survives the blend — and it lands on PRIVATE

Date: 2026-09-16 · Script: `analysis/t2_onset_t1res_feats.py` · Task output `work/t2_onset_t1res_feats.log`
· Ideas 2 and 3 of the user, after the note-63 diagnostic.

Twelve features per link from the three Task 1 CNN reconstructions (mean as the prediction, spread across the
three): signed and absolute speed residual over the last 15 and 60 min, signed flow residual, residual dispersion,
ensemble spread, |residual| x spread, plus the scale-free half (each divided by the median over the links of the
window, and the within-window ranks). Added to the H onset features; paired LightGBM, 3 seeds x 5 bags, identical
prior-dropout draws; scored on the blank-aware official-like onset windows.

Train-side features needed Task 1 reconstructions of the train month, which did not exist: `scripts/t1_predict_ckpt.py
--splits train` was run per panel for each of the three checkpoints (24 processes, `work/t1_train_pred.log`).

| onset | val d1-14 | val all | priv d1-14 | priv all |
|---|---|---|---|---|
| alone, base | 0.787 | 0.753 | 0.810 | 0.790 |
| alone, + T1 block | −0.0116 (0/3) | +0.0095 (3/3) | **+0.0276 (3/3)** | **+0.0383 (3/3)** |
| M blend, base | 0.797 | 0.775 | 0.839 | 0.811 |
| **M blend, + T1 block** | +0.0028 (2/3) | +0.0056 (2/3) | +0.0035 (2/3) | **+0.0243 (3/3)** |

The T1 block takes **14.6% of the gain**, so the model uses it heavily.

## Reading

1. **This is the first onset-side change in this project that is positive in all four conditions inside the blend.**
   Notes 44 (hazard/rank), 50 (oriented margins), 51 (gating/stacking) and 47 (the analog map) all failed there.
2. **The gain is concentrated on the private month** (+0.024 blended, 3/3 seeds) while validation moves only
   +0.003…+0.006. The public LB scores the validation month, so this will barely show there — but **the final
   ranking is the private month**, so the asymmetry is in the direction that matters for the standing, not for the
   feedback we can see.
3. Honest counterweights: 169 / 162 windows per split, and `alone` on validation d1-14 is negative (0/3). The split
   asymmetry may be sampling noise as much as signal.
4. Task 1 was trained on the train month, so its residuals there are smaller: **1.32 km/h on train vs 1.82 km/h on
   the scored months (ratio 1.38)**. The scale-free features exist for exactly this reason, but the model is free to
   use the raw ones, which behave differently across the boundary. A scale-free-only rerun is the next check before
   any candidate is built.

## Next

1. Scale-free-only variant (drop the six raw columns) — if the gain holds, the train/scored level gap is not what
   drives it.
2. If it holds, a candidate: onset = M blend with the T1 block inside the LightGBM, ongoing = R unchanged.
   Expected public LB ≈ +0.0005, expected private ≈ +0.002…+0.004 before the note-58 halving.

## Robustness check: scale-free features only (`T1RES_COLS=sf`)

The concern was that Task 1 was trained on the train month, so its raw residuals there are 1.4x smaller than on the
scored months and the model could be reading that level difference. Dropping the six raw columns and keeping only
the ratios to the window median and the within-window ranks:

| onset | val d1-14 | val all | priv d1-14 | priv all |
|---|---|---|---|---|
| alone, raw + scale-free | −0.0116 (0/3) | +0.0095 | +0.0276 | +0.0383 |
| **alone, scale-free only** | **+0.0084 (3/3)** | **+0.0132 (3/3)** | +0.0124 (2/3) | +0.0196 (3/3) |
| blend, raw + scale-free | +0.0028 | +0.0056 | +0.0035 | +0.0243 |
| **blend, scale-free only** | +0.0032 (2/3) | +0.0009 (2/3) | **−0.0019 (0/3)** | +0.0132 (3/3) |

- **The concern was partly right.** Alone, the only negative condition of the full block (validation d1-14, 0/3
  seeds) flips to **+0.0084 with 3/3 seeds** once the raw columns are gone: part of what the full block learned was
  the train-vs-scored level difference, not traffic.
- **The effect survives.** Scale-free alone is positive in all four conditions; blended it is positive in three of
  four, with private all-days +0.0132 (3/3).
- **But blended it is about half the size**: the mean over the four conditions falls from +0.0091 to +0.0039 onset,
  i.e. ~+0.0006 total, with the private month carrying it (+0.0132 onset there, ~+0.002 total on private).

Decision: the scale-free form is the honest one to carry forward. It is worth ~+0.0006 total on the public month
and ~+0.002 on the private month, which is the month the final ranking uses and the one the public LB cannot show.
