# 47 — Task 2 onset: a state-conditioned (analog) prior beats LightGBM on private by itself, but not on validation

Date: 2026-09-15 · Script: `analysis/t2_analog_prior.py` (log `work/t2_analog_prior.log`, 344 s)

The user's direction: develop the prior rather than the model. The static train prior carries the location base
rate: 98–99% of onsets sit at train-frequent bottlenecks, and dropping pos/priors hurts. But it breaks under the run
shift (notes 17, 19, 20), while capacities are stable and sub-1% demand differences decide which candidate breaks
down (note 21). So condition the prior on today's demand state.

**Method (history only).**
- Demand state: r = log(flow / train mean flow of the same weekday type, slot and link) over T−60…T−5.
- Two keys:
  - `cand`: per candidate bottleneck (train's 10 most frequent first-onset run ends), the mean r within ±2 links
    over the last 30 min and its change, plus the time of day;
  - `pca`: 8 principal components of the last-30-min r over all links, plus the time of day.
- Library: the train earliest-onset windows of the same weekday type (~280 per panel), each with its T+30 onset set.
  For train rows, same-day windows are excluded.
- Analog map: the K = 20 nearest library windows with Gaussian weights, shrunk to the library frequency (α = 2).
- LightGBM features: the map, its ±1-link smooth, its rank in the window, and its lift over the frequency, with 50%
  dropout on training rows.
- Paired LightGBM in H's configuration, 3 seeds × 5 bags. Scored on masked and on T1-filled (official-like)
  histories.

Train-internal sanity check (leave-day-out, one panel): the true set is in the map's top 3 in 93–95% of windows,
vs 72% for the static frequency.

## Results (onset IoU, family mean; LightGBM = mean of 3 seeds; Δ vs paired base, seeds > 0)

| | masked: val d1–14 / val all / priv d1–14 / priv all | filled (official-like) |
|---|---|---|
| LightGBM H base | 0.787 / 0.753 / 0.810 / 0.790 | 0.795 / 0.766 / 0.823 / 0.798 |
| **analog map ALONE, pca key** | 0.767 / 0.756 / **0.856 / 0.841** | 0.749 / 0.741 / **0.858 / 0.832** |
| analog map alone, cand key | 0.753 / 0.739 / 0.819 / 0.809 | 0.768 / 0.743 / 0.822 / 0.801 |
| LightGBM + analog (pca) | −0.005 (1/3) / +0.013 (3/3) / **+0.028 (3/3)** / +0.012 (2/3) | −0.001 / +0.003 / **+0.018 (3/3)** / +0.008 |
| LightGBM + analog (cand) | −0.008 / +0.007 / +0.025 / +0.022 (3/3) | −0.022 (0/3) / −0.004 / +0.009 / +0.011 (3/3) |

- **The map by itself, with no model, beats LightGBM on private by about +0.04** (0.841 vs 0.790 masked; 0.832 vs
  0.798 filled). It is the largest onset gap seen on private. On validation it is 0.02–0.05 below LightGBM.
- As LightGBM features (25–35% of the gain), it helps private consistently (+0.008…+0.028) and is flat or
  negative on validation.
- The route-wide pca key is better than the per-candidate key on private.
## Using the map as a prior blended AFTER the model (`analysis/t2_analog_blend.py`, log `work/t2_analog_blend.log`)

The re-run (`work/t2_analog_prior_v2.log`, identical numbers) caches the per-window probabilities in
`work/cache/t2analog_{masked,filled}_<split>_<panel>.npz`. Here p = w·LGBM_base + (1 − w)·map (pca key), against
LGBM_base (the re-trained H, mean of 3 seeds):

| Δ onset | masked: val d1–14 / val all / priv d1–14 / priv all | filled (official-like) |
|---|---|---|
| w 0.8 | +0.006 / +0.020 / +0.021 / +0.022 | +0.006 / +0.014 / +0.012 / +0.009 |
| **w 0.7** | **+0.003 / +0.021 / +0.021 / +0.042** | **+0.007 / +0.015 / +0.022 / +0.018** |
| w 0.6 | −0.006 / +0.020 / +0.027 / +0.044 | +0.005 / +0.013 / +0.035 / +0.025 |
| map as LightGBM features (for comparison) | −0.009 / +0.017 / +0.025 / +0.028 | −0.002 / +0.001 / +0.020 / +0.004 |

- **With w 0.7–0.8 every one of the 8 cells is positive** (masked and filled × 4 conditions): about +0.016 onset,
  or ≈ +0.002–0.003 total. Blending the map in as a prior after the model is more stable than feeding it as a
  feature. The cand key and logit blends are weaker.
- Mean neighbour distance (pca key, standardised): train leave-day-out 1.90, validation 2.8–3.0, private 3.4–3.6.
  Private is the FARTHER run yet benefits more, so the "validation is out of the library" hypothesis is wrong.
- Next: the same prior on top of M's production onset rule (0.7 H + 0.3 CNN), from caches:
  `analysis/t2_analog_m_blend.py`.

## On top of M's production onset rule (`analysis/t2_analog_m_blend.py`, log `work/t2_analog_m_blend.log`)

Masked histories only; no filled production CNN probabilities are cached. Δ vs M = 0.7 H + 0.3 CNN (panels
better/worse):

| | val d1–14 | val all | priv d1–14 | priv all | mean / min |
|---|---|---|---|---|---|
| **0.7·M + 0.3·map (pca)** | +0.006 (1/1) | +0.003 (2/3) | +0.006 (1/0) | +0.009 (2/1) | +0.006 / +0.003 |
| 0.5 H + 0.2 CNN + 0.3 map (pca) | +0.006 | +0.003 | +0.006 | +0.008 | +0.006 / +0.003 |
| 0.8·M + 0.2·map (pca) | +0.008 | +0.006 | −0.002 | +0.005 | +0.004 / −0.002 |
| H + map (no CNN), best w 0.6 | −0.013 | −0.001 | +0.008 | +0.013 | +0.002 / −0.013 |

- On top of M, the map still helps in all 4 conditions, but only by +0.003…+0.009, about +0.001 total. Most of
  its gain over LightGBM alone overlaps with what the CNN already adds.
- Production script (written, not run): `scripts/t2_analog_probs.py` builds the pca map on the official windows
  from their window histories. To use it, `t2_cnn_blend_predict.py` needs an `--analog-dir` / weight option.

## Improving the map itself, sweep 1 (`analysis/t2_analog_tune.py`, log `work/t2_analog_tune.log`)

One factor at a time, scored from caches (no model re-training). 'ref' reproduces the numbers above.

- **K 40 is the only variant that helps almost everywhere:**
  - map alone on filled histories: +0.007…+0.014 in all 4 conditions;
  - on top of M: +0.003 / +0.003 / 0 / 0;
  - base + map on filled histories: +0.002 / +0.003 / +0.009 / +0.001.
- **Harmful:**
  - smoothing the map over links: −0.03…−0.10;
  - a key over the whole hour (flow12): −0.02…−0.04;
  - a morning-only library: priv all −0.04;
  - a ±1 h time-of-day window: private −0.02…−0.03.
- **Mixed:** pca16 (private up, validation down), shrinking to the time-of-day prior (validation up, private down),
  alpha 0.5 or 5, pca4, and flow+speed. All are within ±0.01.
- "tod weight 3" was identical to ref. That is a bug: the key is standardised per dimension, which cancels the
  weight. It is fixed in sweep 2.
- Single-factor tuning moves things by about ±0.005 at most, so the map needs a structural change, not tuning.

## Sweep 2: structural changes (`t2_analog_tune.py sweep2`, log `work/t2_analog_tune2.log`; base = K 40)

- **Supervised key + speed (`xcov8 flow+speed`) improves the map ALONE in all 8 cells.**
  - The key is the 8 leading directions of the cross-covariance between the state (log flow ratio and v/vf over the
    last 30 min) and the library onset maps: a PLS-style key for "which demand pattern explains WHERE".
  - Masked histories: 0.750 / 0.767 / 0.870 / 0.860 (+0.003 / +0.013 / +0.006 / +0.011).
  - Filled histories: 0.766 / 0.778 / 0.869 / 0.855 (+0.005 / +0.031 / +0.004 / +0.009).
  - On average the map alone now beats LightGBM (4-condition mean 0.812–0.817 vs 0.785–0.795). It is still below
    on val d1–14.
- Other changes:
  - K60 and bw ×0.5 are mixed.
  - With the tod-weight bug fixed, weight 3 hurts private (−0.03) and weight 0.5 hurts validation.
  - xcov16 raises private (+0.02…+0.03) and lowers val d1–14 (−0.05). xcov4 is worse.
- **At a fixed map weight of 0.3, every map gives about the same blend** (on top of M about +0.007 vs no map).
  A better map may need more weight; checked next by `analysis/t2_analog_blendw.py` (map weight 0.2…0.5).

## Map weight (`analysis/t2_analog_blendw.py`, log `work/t2_analog_blendw.log`)

Δ vs the model alone; columns val d1–14 / val all / priv d1–14 / priv all:

| | on top of M (masked) | on top of the re-trained H (filled, no CNN) |
|---|---|---|
| **K40, map weight 0.4** | +0.005 / +0.008 / +0.006 / +0.013 (mean +0.008, min +0.005) | +0.009 / +0.014 / +0.041 / +0.028 (mean +0.023, min +0.009) |
| K40, 0.3 | +0.008 / +0.007 / +0.006 / +0.008 (mean +0.007) | +0.008 / +0.018 / +0.031 / +0.019 (mean +0.019) |
| K60, 0.3 | +0.001 / +0.004 / +0.006 / +0.008 | +0.010 / +0.023 / +0.031 / +0.020 (mean +0.021, min +0.010) |
| xcov8 flow+speed, 0.3 | +0.003 / +0.009 / +0.006 / +0.011 | +0.002 / +0.016 / +0.035 / +0.023 |
| weight 0.5 (any map) | validation −0.006…−0.027 | mixed |

- **Against LightGBM without the CNN the map adds about +0.02 onset. Against M it adds only about +0.008**
  (≈ +0.001 total). The map and the CNN carry largely the same onset-location information.
- The supervised key makes the best map ALONE but not the best blend partner for M; K40 at weight 0.3–0.4 is.
- Open: M itself was never scored on filled (official-like) histories, because filled CNN probabilities are not
  cached. Under official-like inputs the map's increment over M may differ (over H it grows from masked to filled).

## M on filled (official-like) histories, and the map's increment there (`analysis/t2_analog_mfilled.py`, log `work/t2_analog_mfilled.log`)

Production H (5 bags) and CNN×3 are recomputed on masked and on T1-filled histories; M = 0.7 H + 0.3 CNN. Δ vs M;
columns val d1–14 / val all / priv d1–14 / priv all:

| | masked | **filled (official-like)** |
|---|---|---|
| M | 0.802 / 0.780 / 0.847 / 0.818 | 0.783 / 0.783 / 0.845 / 0.811 |
| **0.6 M + 0.4 map K40** | +0.005 / +0.008 / +0.006 / +0.013 (mean +0.008, min +0.005) | **+0.023 / −0.001 / +0.012 / +0.020 (mean +0.013)** |
| 0.6 M + 0.4 map K60 | +0.004 / +0.003 / +0.006 / +0.009 | +0.026 / +0.004 / +0.004 / +0.014 (mean +0.012, all 4 > 0) |
| 0.6 M + 0.4 map xcov8 f+s | −0.018 / −0.006 / +0.009 / +0.010 | +0.018 / +0.001 / +0.011 / +0.021 (mean +0.013, all 4 > 0) |
| 0.5 M + 0.5 map K40 | −0.009 / −0.006 / +0.018 / +0.016 | +0.020 / −0.004 / +0.021 / +0.028 (mean +0.016) |
| 0.6 H + 0.4 map K40 (no CNN) | −0.010 / +0.003 / +0.011 / +0.020 | +0.022 / +0.001 / +0.023 / +0.008 |

- **On official-like histories the map's increment over M roughly doubles: +0.012…+0.016 onset** (masked:
  +0.008). That is about +0.002 total.
- The most robust choice across both history types is **0.6 M + 0.4 map K40**: masked positive in all 4
  conditions, filled 3 of 4 positive with val all −0.001 ≈ 0.
- Production path: `scripts/t2_analog_probs.py` (the pca map on the official windows; set K = 40) plus an
  `--analog-dir` / weight option in `t2_cnn_blend_predict.py`, then swap into L (or into N/O for their ongoing).

## Candidate P = N + the analog onset map (built and verified; log `work/build_P.log`)

- `scripts/t2_analog_probs.py` now reuses the evaluated functions (`t2_analog_tune`: state / build / amap, config
  K40). On the official windows the histories are 76% observed and the per-window map maxima are 0.4–0.8.
- `t2_cnn_blend_predict.py --analog-dir work/sub_t2analog --w-analog 0.4` sets onset p = 0.6·(0.7 H + 0.3 CNN×3) +
  0.4·map. Ongoing is N's (0.5·g1 + 0.5·CNN×3, τ 0.55). Output: `work/sub_t2P`.
- File: `work/sub_P_ttt_t2gsmap_t4ref/submission.csv` (6,980,503 rows).
- Against N only `queue_pred` differs: 19 cells, all in onset windows (validation 13 cells in 5 windows, private 6
  cells in 3 windows). No NaN, and every other column is identical.
- Only 5 onset windows per panel and split exist, so the map flips few official decisions; the realised LB change is
  small and noisy.
- Estimate ≈ M + 0.004 (N's ongoing ≈ +0.002, map ≈ +0.002) ≈ 0.879.

- **Open question (answered above: not a distance effect):** why validation does not benefit. Hypothesis: validation's demand states lie farther from the
  train library, so the neighbours are poor analogs. Next: log neighbour distances per split, cache per-window
  probabilities, and test simple LightGBM ⊕ map blends offline.
