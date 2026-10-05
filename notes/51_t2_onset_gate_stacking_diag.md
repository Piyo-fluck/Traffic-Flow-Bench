# 51 — T2 onset: diagnostics for the gating / stacking proposal → both premises fail; only the queue template survives

Date: 2026-09-16 · Script: `analysis/t2_onset_gate_diag.py` · Log: `work/t2_onset_gate_diag.log` (7 s, cached
probabilities only, no training)

The user's third proposal: stop building base models; route or re-rank the outputs of M's LightGBM (H) and CNN.
Its steps were (1) top1/top2 margin analysis of the missed windows, (2) an adaptive per-window blend weight (gate),
(3) top-10-bottleneck stacking, (4) a pairwise judge for ambiguous windows, (5) a bottleneck-conditioned queue
template, (6) paired bootstrap instead of Public-LB reading. Every premise is testable on cached probabilities.

**Setup.** Candidates = the 10 largest peaks of the train first-onset cell histogram, at least 3 links apart, and
each link is assigned to the nearest candidate (a Voronoi partition; radius masks overlap and make "which
bottleneck" ambiguous). Candidate score = the sum of M's probability over its cell. Windows: the blank-aware
official-like onset set (val 169, priv 162), M = 0.7·H + 0.3·CNN×3, ratio rule. Deltas are per-window paired with a
2000-sample bootstrap CI.

| | validation | private |
|---|---|---|
| M onset IoU | 0.810 | 0.833 |
| true cells inside the assigned bottleneck's cell (Voronoi / ±4 / ±6) | 0.84 / 0.94 / 0.98 | 0.85 / 0.96 / 0.97 |
| bottleneck top1 correct · true in top3 | 0.72 · 0.99 | 0.80 · 0.98 |

## 1. The margin premise is reversed: M is confidently wrong

| top1 − top2 candidate score | exact (IoU = 1) | partial | **miss (IoU = 0)** |
|---|---|---|---|
| validation, median (share of windows) | 1.25 (0.75) | 0.64 (0.14) | **2.00 (0.12)** |
| private, median (share) | 1.63 (0.77) | 0.89 (0.10) | **2.25 (0.12)** |
| top1 correct | 0.87 / 0.95 | 0.43 / 0.59 | **0.10 / 0.00** |

- **The missed windows carry the LARGEST margin, not the smallest.** The low-margin windows are the *partial* ones,
  i.e. count errors, which hold only 26–38% of the loss (note 44).
- Routing "low-margin windows to a pairwise judge" (proposal step 4) would therefore target the wrong 14% and leave
  the misses untouched. The proposal's own fallback branch applies: look at the confusion instead.

## 2. What the confusion actually is: the habitual bottleneck vs a rare one

| missed windows | validation | private |
|---|---|---|
| \|chosen − true\| in links, median (share ≤ 4) | 26 (0.10) | 26 (0.05) |
| train frequency of the TRUE bottleneck | 0.28 | 0.25 |
| train frequency of the CHOSEN bottleneck | 0.45 | 0.42 |
| (exact windows: true / chosen frequency) | 0.65 / 0.70 | 0.74 / 0.76 |

- The misses are not near-misses. On those days an **unusual** bottleneck fires and M predicts the **habitual** one,
  half a corridor away. Most of them sit in two panels (D7_I405_S 0→2 and 0→1, D12_I5_S 7→1, D7_I210_W 2→0).
- This is the same wall as notes 17, 21, 39 and 44: which candidate fires first is decided by demand differences the
  history does not show.

## 3. The gate premise fails: LightGBM and the CNN agree exactly where M is wrong

| | validation | private |
|---|---|---|
| LGBM top1 = CNN top1 | 0.84 | 0.90 |
| IoU when they agree / disagree | 0.804 / 0.845 | 0.835 / 0.814 |
| share of the misses held by disagreement | 1/20 | 1/20 |

Disagreement is not an error signal, so a gate keyed on it has nothing to route. The ceiling is small anyway:

| blend weight α (α·LGBM + (1−α)·CNN) | validation | private |
|---|---|---|
| fixed sweep, best | 0.7 → 0.810 | 0.7 → 0.833 |
| **ORACLE per-window α** | 0.838, **+0.0280 [+0.013, +0.045]** | 0.850, **+0.0174 [+0.006, +0.032]** |
| α = 0.7 already optimal | 0.91 of windows | 0.94 of windows |

**An oracle gate is worth +0.017…+0.028 onset (≈ +0.003 total), and 0.7 is already optimal in 91–94% of windows.**
A learned gate reaches a fraction of an oracle, so this line is closed. M's fixed 0.7 is confirmed as the optimum.

## 4. Oracles: the shape is solved once the bottleneck is known

| paired delta vs M | validation | private |
|---|---|---|
| ORACLE true bottleneck, M's shape inside its Voronoi cell | −0.0593 [−0.103, −0.016] | −0.0580 [−0.098, −0.018] |
| ORACLE true bottleneck, M's shape within ±6 | +0.0498 [+0.020, +0.080] | +0.0233 [−0.000, +0.051] |
| ORACLE best of M's top-3 (±4) | +0.0624 [+0.029, +0.096] | +0.0358 [+0.012, +0.063] |
| **ORACLE true bottleneck, train template** | **IoU 0.931, +0.1204 [+0.067, +0.177]** | **IoU 0.937, +0.1042 [+0.054, +0.155]** |

- Restricting M to the true bottleneck's cell *hurts*: the true sets spill past one cell (only 84–85% of the true
  cells are inside it), so hard restriction truncates correct cells. The queue extent is not a within-cell object.
- **Given the true bottleneck, the static train template P(link in the T+30 set | bottleneck) reaches 0.93–0.94.**
  This is the proposal's own decisive test, and it answers it: the residual loss is bottleneck classification, not
  queue extent.
- A perfect re-ranker among M's top-3 is worth +0.03…+0.06 onset (≈ +0.005…+0.009 total) — an oracle bound for
  proposal step 3, on windows where the true bottleneck is the rare one and the probabilities already point away.

## 5. The one legitimate gain: the train template as a shape prior on M

Not an oracle: take M's OWN top1 bottleneck and reshape M's probability with that bottleneck's train template.

| | validation | private |
|---|---|---|
| M's top1 + template alone (M's shape discarded) | +0.0022 [−0.026, +0.031] | −0.0037 [−0.022, +0.015] |
| M × (1 + 2·template[top1]) | **+0.0119 [+0.004, +0.023]** | +0.0082 [−0.004, +0.023] |
| M × (1 + 5·template[top1]) | +0.0118 [−0.005, +0.030] | **+0.0113 [+0.001, +0.024]** |
| 0.9·M + 0.1·template[top1] | +0.0055 [−0.001, +0.015] | +0.0051 [−0.006, +0.019] |

- Multiplying by the template is worth about **+0.010 onset (≈ +0.0015 total)**. It needs no training and no new
  model: the template comes from train, the bottleneck from M itself.
- Caveat: β was chosen on the same 331 windows, and for every β the CI crosses 0 in one of the two splits. The
  honest reading is "+0.01 ± 0.01", the same size as the effects that failed on the Public LB in note 48.

## 6. Verdict on the proposal

| step | outcome |
|---|---|
| 1 top1/top2 analysis of the misses | **done here**: the premise is reversed, misses are the highest-margin windows |
| 2 adaptive gate (per-window α) | **closed**: oracle ceiling +0.017…+0.028, disagreement uninformative, 0.7 optimal in 91–94% |
| 3 top-10 bottleneck stacking | oracle bound +0.03…+0.06; the misses need the rare bottleneck the probabilities point away from |
| 4 pairwise judge on ambiguous windows | **closed**: the misses are not the ambiguous windows |
| 5 bottleneck-conditioned queue template | **the one survivor**: oracle 0.93–0.94 given the bottleneck; legitimate version ≈ +0.01 onset |
| 6 paired bootstrap instead of LB reading | **adopted**: `ci()` in this script; used for every delta above |

The proposal's core insight is confirmed in one direction — the queue shape is solved once the bottleneck is
known — and refuted in the other: the routing signals it proposed (margin, model disagreement) do not mark the
windows that fail.
