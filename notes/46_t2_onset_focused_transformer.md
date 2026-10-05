# 46 — Task 2: onset-focused training of the transformer (fine-tuning on onset windows with a listwise loss)

Date: 2026-09-15 · Code: `analysis/t2_cnn.py`, env flags `T2CNN_ON_SHARE`, `T2CNN_LISTWISE`, `T2CNN_INIT`,
`T2CNN_ONSET_ONLY`, `T2CNN_LR` (transformer only).

The user's direction: improve the models themselves; ensembles come last. The judge is the SINGLE model's onset IoU
on the blank-aware official-like windows.

Why the transformer is weak on onset:
- Only about 290 of about 3,300 training windows per panel are onset windows.
- In those windows most of the loss falls on T+5…T+25 cells, which are empty by definition.

The model is therefore mostly an ongoing model.

The changes, all tied to onset:
- oversample onset windows;
- a listwise loss at T+30 on onset windows: softmax over the eligible links against the normalised true set, i.e.
  "which link breaks down first", the competing-risk view of note 44;
- fine-tuning a trained transformer on onset windows only (lr 2e-4, 5k steps, early stopping on train ES windows).

## Fine-tuning robustness (local, recipe: onset-only, listwise 0.5, lr 2e-4, 5k steps)

Logs `work/t2_tfmon_C1_local_s{1,2}.log`. Single-model onset IoU, before → after fine-tuning:

| | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| tfm seed 1 | 0.765 → **0.775** | 0.690 → **0.712** | 0.784 → **0.820** | 0.691 → **0.704** |
| tfm seed 2 | 0.750 → **0.802** | 0.691 → **0.736** | 0.808 → **0.837** | 0.704 → **0.708** |
| (smoke: seed 0, 300 steps) | 0.746 → 0.790 | 0.672 → 0.703 | 0.781 → 0.852 | 0.695 → 0.715 |
| LightGBM H, for reference | 0.793 | 0.765 | 0.824 | 0.798 |

- **Fine-tuning improves the single model's onset in 8/8 seed × condition cells** (+0.004…+0.052).
- On days 1–14, the closest proxy for the official onset days 1–5, it reaches LightGBM's level (0.78–0.84 against
  0.79–0.82). On all days it stays 0.03–0.09 below LightGBM.
- Ada, seed 0, 5k steps (logs on the remote: `work/t2_tfmon_C1.log`, `work/t2_tfmon_C2.log`). Base 0.746 / 0.672 /
  0.781 / 0.695:

  | single onset | val d1–14 | val all | priv d1–14 | priv all |
  |---|---|---|---|---|
  | C1 fine-tune + listwise 0.5 | 0.767 | 0.715 | 0.796 | 0.689 |
  | **C2 fine-tune, no listwise** | **0.790** | **0.726** | **0.810** | **0.691** |

  **C2 ≥ C1 in all 4 conditions.** What helps is fine-tuning on onset windows; the listwise "first breakdown" loss
  does not.
- Every variant stays far below LightGBM on all days (priv all 0.69 vs 0.80), while it is level on days 1–14.
  Hypothesis: later days of the scored months have more T1-masked history cells, and the networks, trained on
  near-complete train truth, handle missing inputs worse than LightGBM. Checked by `analysis/t2_hist_missingness.py`.
- **Missingness check** (`analysis/t2_hist_missingness.py`, log `work/t2_hist_missingness.log`):
  - The days hypothesis is wrong. Days 1–14 and 15+ have the same share of observed history cells (0.51–0.53 vs
    0.51–0.54).
  - **But there is a large train/score mismatch that every model shares.** Training windows come from train
    truth, where 76.4% of history cells are observed (only ineligible cells are missing). The scored windows come
    from the T1-masked release, where only 51–53% are observed.
  - LightGBM and the networks are all trained on histories about a third more complete than those they are
    scored on.
  - Measured mask structure: about 30% of eligible cells, dropped cell by cell (median run 1 slot, no whole-link
    gaps, almost no spatial correlation).
  - **Correction after reading `docs/MASK_SPEC.md`:** "cells inside a Task 2 window history are excluded from
    [the T1 mask] because that history republishes them". The OFFICIAL window histories (`window_history.parquet`)
    are therefore as complete as the training windows. It is our official-like PSEUDO windows, cut from the masked
    layer, whose histories are about a third more masked than the real ones.
  - So masking augmentation would move training away from production and is dropped. Instead, the pseudo-eval
    scores every model on harsher inputs than the LB does, and models that need complete inputs (the networks)
    may be underrated there relative to LightGBM.
- Listwise weight 1.0 (local, seed 0, 5k steps; `work/t2_tfmon_lw1_local.log`): 0.762 / 0.685 / 0.850 / 0.713.
  Compared with C2 (no listwise, 0.790 / 0.726 / 0.810 / 0.691), it wins only priv d1–14, so again no consistent
  benefit from the listwise loss. Weight 2.0 (`work/t2_tfmon_lw2_local.log`): 0.770 / 0.701 / 0.819 / 0.690, again
  not better than C2. **Conclusion: the listwise loss is dropped; plain onset fine-tuning (C2) is the recipe.**
- **Ada from-scratch variants (seed 0, 30k steps; `work/t2_tfmon_eval.log`).** Single-model onset (val d1–14 / val all
  / priv d1–14 / priv all); base tfm s0 is 0.746 / 0.672 / 0.781 / 0.695:
  - A, onset share 0.5: **0.767 / 0.710 / 0.831 / 0.723**; ongoing alone about unchanged.
  - B, onset share 0.5 + listwise 0.5: **0.772 / 0.713 / 0.810 / 0.747**; ongoing alone −0.03…−0.06.
  - Both lift the single model's onset in all 4 conditions, but on all days they stay below LightGBM (0.765 / 0.798).
  - In blends with H, B at w 0.85 gives +0.012…+0.017 in all 4 conditions, similar to M's CNN blend, not better.
  - The fine-tuned models (C1, C2) hurt on all days when blended.
- All of these numbers come from the harsh (masked-history) pseudo-eval; see the T1-filled re-evaluation below.

## Re-evaluation with T1-filled histories (`analysis/t2_imputed_eval.py`, log `work/t2_imputed_eval.log`)

The masked history cells of the pseudo windows are filled with our Task 1 predictions (submission L's T1). The
observed share of onset history cells rises from 0.524 to 0.764, which is the official windows' level. The window
selection is unchanged. Family-mean IoU, masked → filled:

| onset | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| LightGBM H | 0.793 → 0.791 | 0.765 → 0.769 | 0.824 → 0.825 | 0.798 → 0.797 |
| TFM ×3 | 0.760 → 0.764 | 0.694 → 0.707 | 0.821 → 0.812 | 0.706 → 0.705 |
| **TFM onset-fine-tuned (seeds 1+2)** | 0.795 → **0.805** | 0.732 → 0.758 | 0.860 → **0.849** | 0.732 → 0.721 |
| M's blend (0.7 H + 0.3 CNN) | 0.802 → 0.783 | 0.780 → 0.783 | 0.847 → 0.845 | 0.818 → 0.811 |
| 0.5 H + 0.5 TFM-ft | 0.823 → 0.827 | 0.775 → 0.787 | 0.849 → 0.854 | 0.801 → 0.803 |

| ongoing | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| LightGBM I | 0.774 → 0.794 | 0.772 → 0.794 | 0.788 → 0.799 | 0.757 → 0.777 |
| TFM ×3 | 0.804 → 0.822 | 0.801 → 0.822 | 0.807 → 0.831 | 0.777 → 0.799 |
| M's blend | 0.809 → 0.831 | 0.806 → 0.831 | 0.823 → 0.848 | 0.793 → 0.815 |

- **Onset hardly moves with complete histories**, so the networks' weakness on all days is not a masking artefact.
- **On days 1–14, the closest proxy for the official onset days (1–5), the onset-fine-tuned transformer beats
  LightGBM as a single model** (0.805 vs 0.791, 0.849 vs 0.825). On all days it is still below (priv 0.721 vs 0.797).
- Ongoing gains about +0.02 for every model with complete histories, and the ranking is unchanged. The pseudo-eval
  understates the absolute ongoing level.
- **The Ada variants on T1-filled histories** (`work/t2_imputed_eval_v2.log`). Single-model onset, filled:

  | | val d1–14 | val all | priv d1–14 | priv all |
  |---|---|---|---|---|
  | LightGBM H | 0.791 | 0.769 | 0.825 | **0.797** |
  | A (onset share 0.5) | 0.795 | 0.735 | 0.835 | 0.727 |
  | **B (onset share 0.5 + listwise, from scratch)** | **0.807** | **0.789** | **0.848** | 0.750 |
  | C2 (onset-only fine-tune) | 0.810 | 0.743 | 0.817 | 0.712 |

  **B beats LightGBM as a single model in 3 of 4 conditions** (+0.016 / +0.020 / +0.023). It is the most
  masking-sensitive model: on masked histories its val all was only 0.713. With the official-like complete
  histories it becomes the best onset model. Its remaining weakness is priv all (0.750 vs 0.797).
- **Day-gap diagnostic** (`analysis/t2_onset_daygap.py`, log `work/t2_onset_daygap.log`):
  - 98–99% of onset events, on days 1–14 and on later days alike, sit at train-frequent bottlenecks (±1 link of
    train's 10 most frequent first-onset run ends).
  - So the gap is not "recurrent vs new places"; it is the choice AMONG the same candidates.
  - Among recurrent events on days 15+: H 0.798, TFM-ft 0.767, TFM 0.733. On days 1–14: 0.836 / 0.842 / 0.814.
  - The transformer has neither absolute position nor the train recurrence priors, which LightGBM has (pos, panel,
    table priors with prior dropout).
  - → Next: the transformer with the position + prior channels and prior dropout (`T2CNN_PRIORS=1`, now also in
    the whole-panel batch path). Ada: onset share 0.5 × seeds 0/1/2 for robustness, plus no onset share (seed 0).
    The poller writes `work/t2_tfmpri_eval.log`.
- **Transformer + position/prior channels (Ada, 30k steps; `work/t2_tfmpri_eval.log`), masked-history pseudo-eval.**
  Single-model onset (val d1–14 / val all / priv d1–14 / priv all):

  | | onset alone | ongoing alone |
  |---|---|---|
  | A (onset share 0.5, no priors), for reference | 0.767 / 0.710 / 0.831 / 0.723 | 0.787 / 0.772 / 0.778 / 0.742 |
  | **priors + onset share 0.5, seed 0** | 0.774 / **0.742** / 0.825 / **0.743** | 0.770 / 0.763 / 0.785 / 0.750 |
  | **priors + onset share 0.5, seed 1** | 0.774 / **0.734** / 0.843 / **0.757** | 0.762 / 0.757 / 0.760 / 0.743 |
  | **priors + onset share 0.5, seed 2** | 0.761 / **0.719** / 0.844 / **0.758** | 0.772 / 0.766 / 0.781 / 0.749 |
  | priors only (seed 0) | 0.765 / 0.713 / 0.780 / 0.728 | **0.788 / 0.786 / 0.809 / 0.775** |
  | LightGBM H / I | 0.793 / 0.765 / 0.824 / 0.798 | 0.774 / 0.772 / 0.788 / 0.757 |

  - **The prior channels lift the weak all-days onset in all 3 seeds** (val all +0.009…+0.032, priv all
    +0.020…+0.035 vs A). The gap to LightGBM on priv all shrinks from −0.075 to about −0.04.
  - Priors alone (no onset share) make the best single ongoing transformer so far.
  - The T1-filled re-evaluation, with the B recipe + priors (seeds 0/1, running), follows as `work/t2_imputed_eval_v3.log`.
  - B recipe + priors (onset share 0.5 + listwise 0.5 + prior channels), masked pseudo-eval, onset alone:
    - seed 0: **0.791 / 0.738 / 0.859 / 0.766**, the best masked-eval onset so far, with priv all up to 0.766;
    - seed 1: 0.770 / 0.726 / 0.796 / 0.707.
    - The seed spread is large. Ongoing alone falls (0.71–0.76) because the listwise term takes capacity.
    - seed 2: 0.775 / 0.726 / 0.852 / 0.728. Mean of 3 seeds: 0.779 / 0.730 / 0.836 / 0.734, vs B without priors
      0.772 / 0.713 / 0.810 / 0.747, better in 3 of 4 conditions. Only seed 0 approaches LightGBM's level on all days.
- **Prior-channel transformers on T1-filled histories** (`work/t2_imputed_eval_v3.log`). Single-model onset, filled:

  | | val d1–14 | val all | priv d1–14 | priv all |
  |---|---|---|---|---|
  | LightGBM H | 0.791 | 0.769 | 0.825 | **0.797** |
  | **B, no priors (seed 0)** | **0.807** | **0.789** | 0.848 | 0.750 |
  | priors + onset share 0.5, seeds 0 / 1 / 2 | 0.773 / 0.771 / 0.764 | 0.740 / 0.739 / 0.747 | 0.849 / 0.821 / 0.838 | 0.775 / 0.768 / 0.757 |
  | priors + B, seeds 0 / 1 | 0.791 / 0.744 | 0.749 / 0.721 | 0.855 / 0.845 | 0.767 / 0.727 |
  | priors only (seed 0) | 0.775 | 0.715 | 0.794 | 0.735 |

  - The priors nudge priv all up (0.750 → 0.757–0.775) but cost validation. **Net, they do not beat B without
    priors.** No model reaches LightGBM on priv all.
  - The seed spread with the same recipe is large (priors + B: s0 vs s1 differ by up to 0.05). B without priors
    is a single seed, so its lead is not yet shown to be robust.
  - `work/t2_imputed_eval_v4.log` was meant to add priors + B seed 2, but that model was never registered in
    `NETS`, so v4 repeats v3. The seed-2 filled result is missing (masked: 0.775 / 0.726 / 0.852 / 0.728). Not
    re-run: the user judged this line thin and moved on to state-conditioned priors.
  - **Status of the transformer line:** not a submission candidate for onset (single-seed lead, priv all below
    LightGBM, blend untested on filled histories). For ongoing, the edge over the CNN nearly vanishes on filled
    histories (M-style blend +0.003 / +0.004 / −0.004 / +0.001), so candidate O's gain over M is uncertain.
- Replacing the CNN by the fine-tuned transformer in M's two-model onset blend: 0.827 / 0.787 / 0.854 / 0.803 vs
  M's 0.783 / 0.783 / 0.845 / 0.811 → +0.044 / +0.004 / +0.009 / −0.008.
- Pending (done now): the Ada runs:
  - A: oversampling only, from scratch;
  - B: oversampling + listwise, from scratch;
  - C1: the recipe above on seed 0;
  - C2: C1 without the listwise loss.
  - The local listwise-weight sweep (1.0, 2.0).
