# 36 — Task 2 CNN (idea #4), first run: far below LightGBM on onset; blend helps ongoing only slightly

Date: 2026-09-15 · Script: `analysis/t2_cnn.py` · Run on the 4× RTX 6000 Ada machine
(`/data1/nagao/Kaggle/TrafficFlowBench`, GPU 0). Log: `work/t2_cnn.log` there.

Setup:
- Spatio-temporal CNN, 0.64M parameters. Input: 12 history slots × links, 12 channels (v/vf, q/cap, observed,
  queued, static link features, time of day, weekend). Output: 6 horizon steps × links.
- Training: every earliest-onset window plus up to 3,000 ongoing candidates per panel. 6,000 steps, batch 32,
  64-link crops. No recurrence priors, no position channel.
- The whole run (training plus scoring) took 106 s.
- Scored on the blank-aware official-like windows, against the cached LightGBM probabilities (K's rules:
  onset expected-IoU ratio, ongoing τ 0.65).

| IoU (panels better/worse vs LightGBM) | LightGBM | CNN | mean |
|---|---|---|---|
| onset val d1–14 | 0.793 | 0.306 | 0.728 (−0.064, 0/5) |
| onset val all | 0.765 | 0.361 | 0.725 (−0.040, 1/6) |
| onset priv d1–14 | 0.824 | 0.426 | 0.817 (−0.007, 2/3) |
| onset priv all | 0.798 | 0.487 | 0.814 (+0.016, 5/3) |
| ongoing val d1–14 | 0.774 | 0.740 | 0.773 (−0.001, 4/4) |
| ongoing val all | 0.772 | 0.744 | 0.773 (+0.002, 5/3) |
| ongoing priv d1–14 | 0.788 | 0.741 | 0.795 (+0.007, 6/2) |
| ongoing priv all | 0.757 | 0.725 | 0.760 (+0.003, 4/4) |

- **The CNN onset is very weak (0.31–0.49).** LightGBM's onset leans on the train recurrence priors (with
  dropout) and on link identity. The CNN has neither.
- The ES BCE was still falling at 6,000 steps (0.0583 → 0.0256), so the model is undertrained.
- **30,000 steps** (log `work/t2_cnn_30k.log`; best ES BCE 0.0217 at step 22,500; 417 s in total):

  | IoU (panels better/worse vs LightGBM) | LightGBM | CNN 30k | mean |
  |---|---|---|---|
  | onset val d1–14 / all | 0.793 / 0.765 | 0.525 / 0.504 | 0.787 (−0.006) / 0.751 (−0.015) |
  | onset priv d1–14 / all | 0.824 / 0.798 | 0.619 / 0.509 | 0.809 (−0.016) / 0.788 (−0.010) |
  | ongoing val d1–14 / all | 0.774 / 0.772 | 0.768 / 0.769 | **0.781 (+0.007, 5/3) / 0.784 (+0.013, 5/3)** |
  | ongoing priv d1–14 / all | 0.788 / 0.757 | 0.772 / 0.751 | **0.798 (+0.010, 5/3) / 0.770 (+0.013, 6/2)** |

  The CNN alone nearly matches LightGBM on ongoing. **The ongoing blend is better in all four conditions**
  (about +0.011 ongoing IoU, i.e. +0.0016 total). Onset stays LightGBM-only.
- **Prior and position channels** (`T2CNN_PRIORS=1`, 30k steps, prior dropout 0.5; log `work/t2_cnn_pri.log`;
  best ES BCE 0.0211 at step 21,500):

  | Δ IoU of mean(LGBM, CNN) vs LGBM | plain CNN 30k | prior CNN 30k |
  |---|---|---|
  | onset val d1–14 / all | −0.006 / −0.015 | +0.010 / +0.000 |
  | onset priv d1–14 / all | −0.016 / −0.010 | +0.006 / −0.022 |
  | ongoing val d1–14 / all | **+0.007 / +0.013** | +0.003 / +0.008 |
  | ongoing priv d1–14 / all | **+0.010 / +0.013** | +0.005 / +0.006 |

  - The priors lift the CNN's onset alone (0.57–0.69 vs 0.50–0.62), but the onset blend stays mixed (±0.02). Onset stays
    LightGBM-only.
  - For ongoing the plain CNN blends better than the prior CNN.
- **BCE + 0.5 × Lovász hinge** (a per-window IoU surrogate, Berman et al. 2018; `T2CNN_LOVASZ=0.5`, 30k steps;
  log `work/t2_cnn_lov.log`, model `t2_cnn_lov.pt`; best ES BCE 0.0234 at step 29,500; 746 s):

  | Δ IoU vs LGBM (panels better/worse) | CNN alone | mean(LGBM, Lovász CNN) |
  |---|---|---|
  | ongoing val d1–14 / all | −0.003 / +0.003 | **+0.016 (5/3) / +0.020 (6/2)** |
  | ongoing priv d1–14 / all | −0.007 / +0.003 | **+0.009 (6/2) / +0.022 (7/1)** |
  | onset val d1–14 / all | −0.128 / −0.116 | −0.054 / −0.066 |
  | onset priv d1–14 / all | −0.095 / −0.181 | −0.041 / −0.052 |

  - With Lovász the CNN alone matches LightGBM on ongoing, and the ongoing blend is the best so far: about +0.017
    ongoing IoU, i.e. **+0.0025 total**, against +0.0016 for the plain CNN.
  - The onset CNN improves (0.62–0.73), but the onset blend gets worse. The hinge training de-calibrates the
    probabilities, and the expected-IoU ratio rule assumes calibrated probabilities. Onset stays LightGBM.
- **Blend weights and threshold** (`analysis/t2_blend_eval.py`, from cached probs, Lovász seed 0 only):

  | Δ IoU vs LGBM (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
  |---|---|---|---|---|
  | ongoing 0.6·LGBM + 0.4·CNN, τ 0.60 | **+0.028 (7/1)** | **+0.031 (8/0)** | **+0.024 (7/1)** | **+0.031 (8/0)** |
  | ongoing 0.5/0.5, τ 0.65 (previous) | +0.016 | +0.020 | +0.009 | +0.022 |
  | onset 0.85·LGBM + 0.15·CNN (expected-IoU ratio) | +0.008 (3/1) | +0.012 (5/1) | +0.012 (2/1) | +0.013 (3/1) |
  | onset 0.5/0.5 (previous) | −0.054 | −0.066 | −0.041 | −0.052 |

  - The setting that is best on validation alone is also best on private (a separate run), so the selection bias is
    small.
  - Averaging the plain and Lovász CNNs before blending gives +0.020…+0.027 on ongoing at τ 0.65, so CNN
    ensembling helps too.
  - Estimate: ongoing about +0.029 and onset about +0.011, i.e. **about +0.006 total** over L.
  - Seeds 1 and 2 (Lovász) are training for a 3-seed average.
- **Decision (updated)**: ongoing = mean(LightGBM I, Lovász CNN), onset = LightGBM with the expected-IoU rule.
- Superseded decision: ongoing = mean(LightGBM I, plain CNN 30k), onset = LightGBM with the expected-IoU rule. This is
  about +0.011 ongoing IoU, i.e. +0.0016 total over L. It needs a production path: CNN inference on the official
  window histories.
