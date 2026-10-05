# 124 - Browsed 2022-2026 methods, first round: Chronos-2, a diffusion forecaster, and IoU decision theory

The user closed the "find remaining headroom" phase and asked for heterogeneous architectures from the recent
literature. Licences checked against the Rules page and Kaggle Foundational Rule 6c (no limit on commercial use):
Chronos-2 (Apache 2.0), MOMENT-1-large (MIT), TEMPO-80M (Apache 2.0), TabPFN **v2** weights (Apache 2.0 +
attribution) are usable; TabPFN-2.5 / 3 (non-commercial) are not. All installed in `.venv` without changing torch
or numpy (`momentfm` with `--no-deps`); Chronos also on Ada (weights in the project's `.hf_cache`).

All numbers: ongoing / onset family IoU on the official-like pool of both scored months, validation / private.

## Chronos-2 as a Task 2 forecaster (`fm_lab/chronos_t2.py`, `fm_lab/chronos_ft.py`)

Whole corridor as one multivariate series of v/vf up to T-5; P(v/vf <= 0.6) from the quantile forecast.

| variant | ongoing | onset |
|---|---|---|
| zero-shot, same day only | 0.591 / 0.576 | 0.029 / 0.095 |
| zero-shot, + 3 earlier days | 0.655 / 0.643 | 0.228 / 0.252 |
| fine-tuned (full, 3k steps, 64-link crops of train days), same day | 0.677 / 0.661 | 0.037 / 0.088 |
| fine-tuned (LoRA), + 3 earlier days | 0.613 / 0.594 | 0.155 / 0.203 |
| production old(12) / H | 0.836 / 0.823 | 0.765 / 0.798 |

Longer context and fine-tuning each help, but the model stays far below the production members and every blend
loses on at least one month (ongoing -0.004..-0.02; onset +0.011 private / -0.006 validation at 0.15).

## Candidate A: conditional diffusion of the future speed field (`gen_lab/t2_diffusion.py`)

History encoder initialised from the SSL w192 backbone (note 123), 1-D dilated denoiser over links, 30k steps,
32 DDIM-50 samples per window. Alone: ongoing 0.773-0.792 / 0.752-0.768, onset 0.56-0.61 / 0.57-0.59. As a member
it helps the 12 frozen models (+0.003..+0.004) but adds **0.000** to AT's stack (best +0.0002 after the corrector).

## Choosing each window's threshold (the oracle tau ceiling is 0.9183 against AT's flat 0.8606)

| decision on AT's p | ongoing mean |
|---|---|
| flat tau 0.50 (AT) | **0.8606** |
| tau by diffusion expected IoU (`decide_eval.py`) | 0.8360 |
| same, gated by the expected-IoU margin 0.02..0.2 | 0.8388..0.8566 |
| copula: AT's marginals, diffusion dependence (`copula_eval.py`) | 0.8574 (controls: independent 0.8551, shuffled 0.8557) |
| tau shifted only by the dependence effect (copula pick minus independent pick) | 0.8593 (shuffled control 0.8602) |
| GFM-style expected IoU (Dembczynski 2011, Nowozin 2014) with a learned size distribution (`gfm_eval.py`) | 0.8588 (independence 0.8468, fixed prior 0.8540) |
| the same on isotonic-calibrated p | 0.8601 (val +0.0005, priv -0.0014) |

The diffusion's dependence structure and the learned size distribution each carry real information against their
own controls (+0.002 and +0.012), but no decision rule beats the flat threshold. The per-window budget is not
recoverable from what is observable at T - the same conclusion as note 121, now from Bayes-decision methods. The
remaining ongoing gain has to come from a better p, not from the decision layer.

## Round 2 (same day)

- **TabPFN v2 for onset H** (`fm_lab/tabpfn_onset.py`, fitted on 2030 days): train tail / validation / private
  0.771 / 0.650 / 0.664 against LightGBM on all 266k rows 0.895 / 0.783 / 0.801 - the 10k-row context cannot hold
  the data. Side finding: **LightGBM without any recurrence prior drops -0.101 train tail -> scored months, the same
  as with priors (-0.103)**, so the onset run-shift loss does not come from the prior features.
- **Chronos-2 fine-tuned (full, quantile loss) with 3 earlier days**: ongoing 0.732 / 0.712 (best Chronos variant;
  production old(12) 0.836 / 0.823), blend -0.001..-0.004.
- **Chronos-2 as a backbone trained directly on queue labels** (`fm_lab/chronos_queue.py`, BCE + 0.5 Lovasz,
  context up to 7 earlier days, 4-GPU DDP, block-diagonal group attention + activation checkpointing so 16 windows
  fit per GPU; the block attention matches the library to 2e-7). 6.9 s per 64-window step; ES bce 0.0573 at step
  500 (the SSL CNN logs 0.047 at its step 500, about 0.022 at the end; the ES window sets differ).
- **T1 TTT with the official gap geometry** (block masking from Gan et al., Physica A 2026; `TTT_MASK=target`,
  `EVAL_MASK=target` in `analysis/t1_ttt.py`; SSL w192 all1, 1200 steps, validation). Real T1 targets come in runs
  of 1 (52%), 2-3 (38%), 4-11 (9%), none longer. On a target-geometry evaluation, point TTT +0.0010 total vs
  target-geometry TTT +0.0009: **no gain**. The i.i.d. 20% evaluation reads S_LWR 0.772 where the target-geometry
  one reads 0.719 (closer to the LB-implied level) and flips point-TTT's S_LWR sign (-0.0059 vs +0.0020), so the
  target-geometry evaluation is the better T1 surface.
- **T1 multi-scale U-Net** (`M3_ARCH=unet` in `analysis/t1_m3_cnn.py`, SSL wide 25 + ft 25, Ada logs
  `work/logs_unet/`). Train-tail test block, 0.35 S_state + 0.15 S_phys~: production dilated w192 all1 **0.4485**,
  U-Net 8.3M 0.4478 / 0.4478 (seeds 0/1), U-Net 14.8M 0.4369. Validation, target-geometry eval before TTT:
  0.9467 / 0.7188 (production) vs 0.9461 / 0.7150 and 0.9463 / 0.7140. **Multi-scale does not help.** The same
  production TTT read S_LWR +0.0020 locally and -0.0025 on Ada: TTT's S_LWR effect is inside run-to-run noise.
- **Unused legal data (fact, `scripts/t2_pseudo_adapt_official.py`)**: the pseudo-supervised adaptation cuts
  same-day data at the day's FIRST official origin. 62 of the 80 official ongoing windows are not the first
  window of their day, with on average 9.1 h (>= 4 h for all 62) of same-day released data before their own T
  that no member uses.
