# 54 — T2 ONGOING: shockwave boundary features work → +0.009…+0.011 alone, +0.004…+0.006 in M's blend

Date: 2026-09-16 · Script: `analysis/t2_ongoing_shockwave.py` · Log: `work/t2_ongoing_shockwave.log` (26 min, local)

Notes 52/53 located the ongoing loss (missed queue GROWTH, 42–47% of all errors) and closed the decision side
(no threshold rule helps beyond a global τ). The remaining lever was the probabilities of boundary cells. The queue
boundary is a shockwave, so its motion is predictable from the history in two independent ways.

**Features** (history only, T−60…T−5; the sign is always taken *towards the link being scored*, so nothing depends
on which index side is upstream — notes 49/50 showed orientation rules do not transfer):
- `sw_r` signed distance to the boundary (+ to the nearest queued link if free, − to the nearest free link if queued),
  computed on a time-forward-filled queue mask so that unobserved slots do not break a run;
- `sw_appr3/6` the boundary's **measured** approach speed over the last 15 / 30 min (links per slot), `sw_eta` the
  implied arrival time;
- `sw_rh` the **Rankine–Hugoniot** speed |(q₁−q₂)/(k₁−k₂)| at the nearest boundary, converted to links per 5-min slot
  with `length_km`, and `sw_toward` its direction (+1 when the arriving flow exceeds the discharge);
- `sw_qratio`, `sw_dqcap`, `sw_kratio`, `sw_len` the flow/density imbalance across the boundary;
- per horizon step: `sw_margin_h` = `sw_r` − `sw_appr3`·h, `sw_rh_marg_h` the same with the physical speed,
  `sw_arr_h`, `sw_frac_h`.

**Protocol.** Production I configuration (demand-stress features, no priors, LightGBM 2000 rounds / 127 leaves),
150k sampled rows per panel, 2 seeds, base vs base+shockwave **on identical rows** (a single keep mask — calling
`sample_og` twice draws different random cells and silently breaks the pairing; that bug cost the first run).
Scored on the blank-aware official-like ongoing windows, paired bootstrap CI over windows.

| ongoing IoU (mean of 2 seeds) | validation | private |
|---|---|---|
| LightGBM alone, τ 0.60 | 0.7842 → **0.7932**, **+0.0090 [+0.007, +0.011]** | 0.7779 → **0.7886**, **+0.0108 [+0.009, +0.013]** |
| LightGBM alone, τ 0.50 | 0.7824 → 0.7912, +0.0088 | 0.7762 → 0.7871, +0.0109 |
| **M's blend (0.5 LGBM + 0.5 CNN×3), τ 0.60** | 0.8183 → 0.8221, **+0.0038 [+0.002, +0.005]** | 0.8124 → 0.8180, **+0.0055 [+0.004, +0.007]** |
| **M's blend, τ 0.50** | 0.8296 → 0.8315, +0.0019 [+0.000, +0.003] | 0.8220 → 0.8250, +0.0031 [+0.002, +0.004] |

Every cell is positive, in both seeds, with CIs excluding zero. **This is the first ongoing model change that
survives the blend with the CNN** — the onset attempts (notes 44, 50, 51) all lost their stand-alone gain there.

**The mechanism is the intended one** (error rate per note-52 class, τ 0.60, LightGBM alone):

| class | validation | private |
|---|---|---|
| **grow** (the target) | 0.2984 → **0.2757** (−0.023) | 0.3722 → **0.3449** (−0.027) |
| stay | 0.0423 → 0.0324 (−0.010) | 0.0412 → 0.0307 (−0.011) |
| dissolve | 0.2382 → 0.2496 (+0.011) | 0.2660 → 0.2597 (−0.006) |
| far | 0.0109 → 0.0106 | 0.0095 → 0.0097 |

Growth errors fall in both months, and persistence errors fall with them. Dissolution is unchanged on average —
the features describe the tail's advance better than the head's discharge.

The shockwave block takes **10–17% of the LightGBM gain**, and `sw_margin_h` (the projected boundary crossing) and
`sw_r` enter the top five features overall, behind only `vr_last` and `kc_last`.

## What is bankable

Against the production setting (τ 0.60, no shockwave features), the package **τ 0.50 + shockwave** gives
ongoing 0.8183 → 0.8315 (validation) and 0.8124 → 0.8250 (private), i.e. **+0.013 ongoing ≈ +0.0065 S_queue
≈ +0.002 total**. Of that, about +0.010 is the threshold (notes 52/53) and about +0.003 the new features.

**Caveats.** The base here is retrained, not the exact production I model (150k rows per panel, 2 seeds), so the
absolute levels sit ≈ 0.002–0.003 below note 52's production numbers; the deltas are paired and unaffected.

## v2: both sides tracked separately (`analysis/t2_ongoing_shockwave2.py`, log `work/t2_ongoing_shockwave2.log`, 39 min)

v1 keyed everything on the *nearest* boundary, so for a link inside a queue it could be looking at the tail or the
head, which have opposite dynamics — hence dissolution did not move. v2 tracks the **low-index and high-index side
of every link separately** (`s2_u1/u2` signed distances, `s2_v1/v2` their measured speeds over 15 and 30 min, the
run length and its rate of change, the flow/density/speed at the queued and free link bounding each side, and per
horizon step the projected margin of each side). No orientation assumption is needed.

| ongoing IoU (2 seeds) | validation: base → v1 → v2 | private: base → v1 → v2 |
|---|---|---|
| LightGBM alone, τ 0.60 | 0.7842 → 0.7932 → **0.8019 (+0.0176)** | 0.7779 → 0.7886 → **0.7976 (+0.0197)** |
| **M's blend, τ 0.60** | 0.8183 → 0.8221 → **0.8255 (+0.0073)** | 0.8124 → 0.8180 → **0.8208 (+0.0084)** |
| M's blend, τ 0.50 | 0.8296 → 0.8315 → **0.8347 (+0.0051)** | 0.8220 → 0.8250 → **0.8273 (+0.0053)** |

**v2 roughly doubles v1 in every condition**, with CIs excluding zero. Error rates (τ 0.60, alone, base → v2):
grow 0.298 → 0.265 (val) and 0.372 → 0.330 (priv); stay 0.042 → 0.029 and 0.041 → 0.026; dissolve 0.238 → 0.244
(val, still slightly worse) and 0.266 → 0.254 (priv, now improved); far unchanged. The two-sided block takes 10.7%
of the gain and `s2_m2_h` / `s2_m1_h` (the projected margins of each side) are the top features after `vr_last` and
`kc_last`.

So the growth errors are down by 11–12% relative, and dissolution is finally moving on one of the two months.

## Does it survive inside the note-56 ensemble? **No — and that settles it**

The v2 model's probabilities were cached as a pool source (`analysis/t2_cache_sw2_probs.py`, tag `t2_sw2`) and
added to the shipping recipe (log `work/t2_ongoing_consensus_sw2.log`):

| ongoing, family mean | validation | private |
|---|---|---|
| 2·tfm_more + 1·cnn3 + 1·gs1, τ 0.45 (note 56) | 0.8327 | **0.8216** |
| + t2_sw2 (2:1:1:1) | 0.8327 (±0) | **0.8184 (−0.0032)** |

**The shockwave block adds nothing on validation and hurts on private.** Its gain was real inside M's old blend
(production LightGBM + CNN), but the note-56 ensemble is led by four transformer seeds, and a panel-wide attention
model evidently already represents the queue-boundary dynamics that these hand-built features encode. The
features are redundant there, so **v2 is not shipped**.

What stands: the anatomy (notes 52/55) and the finding that boundary motion is the ongoing loss; what falls: the
belief that hand-built shockwave features are the way to capture it. The transformer was the better vehicle, and it
already existed — measured in note 45 and then left unsubmitted for two days while the effort went into onset.

## Next

1. Dissolution is untouched: add head-side features (discharge flow at the queue head, its trend, the downstream
   free-flow recovery), which the current set only describes indirectly.
2. The blend halves the gain, so the CNN duplicates part of it. Either give the CNN the same boundary information as
   input channels, or re-fit the blend weight now that the LightGBM is stronger.
3. Only about a tenth of the growth errors were removed (0.30 → 0.28). Multi-slot robust boundary tracking and a
   demand-based arrival estimate at the tail are the obvious extensions.
4. Build the submission candidate (production ongoing model with these features, τ 0.50) so the gain is available
   when the user next decides to submit.
