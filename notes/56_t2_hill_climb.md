# 56 — T2 blend weights by hill climbing: ongoing +0.029…+0.033 (≈ +0.0045 total), onset must stay as M

Date: 2026-09-16 · Script: `analysis/t2_hill_climb.py` · Log: `work/t2_hill_climb.log` (69 s) · The user
authorised ensemble work for the +0.01 stacking route (note 55).

M's weights were set by hand: onset 0.7·LightGBM + 0.3·CNN×3, ongoing 0.5·LightGBM + 0.5·CNN×3 at τ 0.60. Greedy
ensemble selection (Caruana, with replacement so members get integer weights) over every cached probability source,
with τ chosen jointly for ongoing. **Fitted on one split, applied to the other, both directions.**

Pool (11 sources): `lgbm` (production LightGBM I), `gs0`/`gs1` (global-state LightGBM), `cnn3` (CNN×3),
`tfm3` (transformer×3), `tfm_more` (transformer seeds s3–s6), `tfm_w128`, `tfm_pri`, `cnn_glob`, `cnn_aux`, `cnn_aux8`.

## Ongoing: the largest single gain measured so far

| | fit val → priv | fit priv → val |
|---|---|---|
| M production (0.5 lgbm + 0.5 cnn3, τ 0.60) | 0.8154 | 0.8207 |
| M at τ 0.50 (notes 52/53) | +0.0093 | +0.0118 |
| candidate O (0.5 gs1 + 0.5 tfm3, τ 0.55; note 45) | +0.0219 | +0.0192 |
| **hill climb** | **+0.0329 [+0.029, +0.037]** | **+0.0294 [+0.025, +0.034]** |
| | → total **+0.0049** | → total **+0.0044** |

Selected recipes:
- fit on validation, τ 0.50: 0.29·tfm_more + 0.14·(cnn3, gs1, tfm_w128, cnn_aux, gs0)
- fit on private, τ 0.45: 0.50·tfm_more + 0.25·cnn3 + 0.25·gs1

Both directions agree on the core — **`tfm_more` (the transformer seeds) takes the largest weight, with `cnn3` and
`gs1` beside it, at τ 0.45–0.50** — and both give ≈ +0.03 ongoing out-of-sample.

**The production LightGBM `lgbm` is never selected in either direction.** Its information is already covered by the
global-state LightGBM (`gs0`/`gs1`) plus the neural families. This also means the shockwave work (note 54, +0.004…
+0.006 inside M's blend) has to be re-measured *inside the new ensemble* before its gain can be added to the total:
its probabilities are not yet cached as a pool source.

## Onset: hill climbing overfits and must not be used

| | fit val → priv | fit priv → val |
|---|---|---|
| M production (0.7 lgbm + 0.3 cnn3) | 0.8327 | 0.8101 |
| hill climb: fitted IoU / applied IoU | 0.8170 / 0.8122 | 0.8486 / 0.8101 |
| applied delta | **−0.0205 [−0.051, +0.009]** | **−0.0000 [−0.021, +0.020]** |

The fitted score rises to 0.849 while the applied score does not move at all: with only 169 (val) and 162 (priv)
official-like onset windows, weight selection fits window noise. **Keep M's 0.7/0.3.** This agrees with note 51,
where α = 0.7 was already optimal in 91–94% of windows and the oracle per-window α was worth only +0.017…+0.028.

## Which recipe to ship (`analysis/t2_ongoing_consensus.py`, log `work/t2_ongoing_consensus.log`)

A 6-member recipe with 1/7-style weights is the kind of thing that fits window noise, so the round-number recipes
over the agreed members were evaluated on **both** months. Family mean is the quantity the official score uses.

| recipe | val IoU / family | priv IoU / family |
|---|---|---|
| M production (τ 0.60) | 0.8207 / 0.8057 | 0.8154 / 0.7928 |
| M at τ 0.50 | 0.8325 / 0.8172 | 0.8247 / 0.7995 |
| candidate O (τ 0.55) | 0.8399 / 0.8254 | 0.8374 / 0.8098 |
| **2·tfm_more + 1·cnn3 + 1·gs1, τ 0.45** | **0.8501 / 0.8327** | **0.8490 / 0.8216** |
| 1:1:1, τ 0.50 | 0.8490 / 0.8312 | 0.8455 / 0.8179 |
| + tfm3 as well (2:1:1:1) | 0.8496 / 0.8331 | 0.8476 / 0.8209 |
| + lgbm kept | 0.8489 / 0.8315 | 0.8460 / 0.8178 |
| + gs0 | 0.8486 / 0.8318 | 0.8457 / 0.8176 |

- **The 2:1:1 recipe at τ 0.45 is the best in both months**, and it is exactly the recipe the private→validation
  hill climb selected, so note 56's cross-split +0.0294 on validation is its out-of-sample confirmation.
- Keeping the production `lgbm` in the mix costs 0.001–0.003; adding `gs0` or `tfm3` changes nothing. The simple
  three-member recipe is the one to ship.
- The gain is not carried by one panel: the worst panel rises too (val D7_I210_E 0.734 → 0.741; priv D7_I10_W
  0.711 → no longer the worst).
- **Family-mean gain +0.0270 (val) and +0.0288 (priv) → ≈ +0.0042 total.**

Caveat: the member set was chosen on these two months, so the both-month table is not fully out-of-sample; the
honest headline remains the cross-split numbers above.

**The shockwave LightGBM of note 54 was tested as a fourth member and rejected**: family mean unchanged on
validation (0.8327) and −0.0032 on private (0.8216 → 0.8184). The transformer seeds already carry the queue-boundary
dynamics. The shipping recipe stays the three-member 2:1:1 at τ 0.45.

## Consequence for the +0.01 route (note 55)

| lever | Δtotal | status |
|---|---|---|
| **ongoing ensemble (hill-climbed weights, τ 0.45–0.50)** | **+0.0044…+0.0049** | measured out-of-sample, both directions |
| onset ensemble | 0 (negative) | closed |
| ongoing shockwave v1 | +0.0005…+0.0008 inside M's blend | must be re-measured inside the new ensemble |
| onset template prior (note 51) | +0.0015 | not built in |
| T1 seeds / longer TTT (T3 follows) | +0.002…+0.003 | not started |

The ongoing ensemble alone is worth about half of the +0.01 target, and it supersedes both the τ change and
candidate O (their gains are contained in it). Next: cache the shockwave LightGBM probabilities so it can enter the
pool, then re-run this selection with it included.

## Doubling the transformer seeds (2026-09-16, after four more were trained on Ada)

Seeds s7-s10 were trained on the Ada machine with the same recipe (T2CNN_ARCH=tfm, LOVASZ 0.5, 30k steps, ~17 min
each on one GPU), giving eight transformer seeds instead of four. Ongoing family mean on the official-like windows:

| recipe | validation | private |
|---|---|---|
| R as shipped: 2*tfm(s3-s6) + cnn3 + gs1, tau 0.45 | 0.8327 | 0.8216 |
| 2*tfm8(s3-s10) + cnn3 + gs1, tau 0.50 | 0.8350 | 0.8222 |
| **3*tfm8 + cnn3 + gs1, tau 0.50** | **0.8357** | **0.8233** |

**Doubling the seeds is worth +0.0030 / +0.0017 family mean, i.e. about +0.0004 / +0.0003 total** - the low end of
the estimate in note 58, and roughly +0.0002 on the LB after halving. The heavier transformer weight (3:1:1) is
slightly better than 2:1:1 in both months, which is consistent with the greedy selection that first picked the
transformer group.
