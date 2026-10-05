# 58 — LB: R = 0.87687, a new best (+0.0018), but only 44% of the predicted gain

Date: 2026-09-16 · Submission 56268281 · Candidate: [57](57_candidate_R.md)

| submission | content | public LB |
|---|---|---|
| L | T1 TTT + expected-IoU onset + stress ongoing | 0.86893 |
| M | L + T2 CNN blend (onset 0.7/0.3, ongoing 0.5/0.5 at τ 0.60) | 0.87505 |
| P, Q | M/N + the state-conditioned analog onset map | 0.87429 / 0.87345 |
| **R** | **M with only the ongoing half replaced by the note-56 ensemble** | **0.87687** |

R changed 715 rows of 6,980,503, all `task == queue`, all in `queue_ongoing` windows (note 57). The gain is
therefore attributable entirely to the ongoing ensemble.

## Calibration: local official-like gains are optimistic by about 2×

| change | predicted from the official-like windows | realised on the LB | ratio |
|---|---|---|---|
| L → M (T2 CNN blend) | +0.008 | +0.0061 | 0.76 |
| M → Q (analog onset map) | +0.008…+0.013 onset ≈ +0.002 total | **−0.0016** | negative |
| **M → R (ongoing ensemble)** | **+0.0041** | **+0.0018** | **0.44** |

Why the shortfall: the public LB scores **5 onset + 5 ongoing windows per panel** (40 ongoing windows on
validation), while our evaluation uses 3351 official-like ongoing windows selected by coverage ≥ 0.7, hours 7–21
and persistence IoU ≤ 0.9. The proxy set is 84× larger and filtered differently, so its mean moves much more
smoothly than the 40 windows that actually count.

**Pattern across the three data points: ongoing-side changes transfer (0.44–0.76 of the prediction), onset-side
changes do not (the map went negative).** That is consistent with the cell counts — an ongoing window carries
hundreds of cells and an onset window 1–5, so onset estimates are dominated by window sampling noise.

## Operating rule from here

1. Multiply any local official-like gain by **≈ 0.5** before deciding whether a change is worth a submission slot.
2. Require the paired bootstrap CI to exclude zero **in both months**, not just on the mean.
3. For onset-side changes, treat local gains as unproven regardless of size (two failures: the analog map, and the
   onset hill climb which was negative out-of-sample).

Under this rule the remaining items are worth: onset template prior +0.0015 local → **unproven** (onset side);
four more transformer seeds +0.001…+0.002 local → **≈ +0.0005…+0.001** on the LB.

## Where the score now stands against the target

The user's target was +0.01 total (0.885). The full stack of everything that survived measurement delivered
**+0.0018**. The measured ceilings (notes 51, 55) say the rest is not reachable with the information we have:
perfect onset bottleneck choice caps the total at ≈ 0.897, and perfect ongoing prediction adds +0.025 at most,
with every decision-side lever already exhausted.
