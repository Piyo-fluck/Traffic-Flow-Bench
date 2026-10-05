# 137 - General Kaggle techniques on the SSL w192 T2 CNN: all within the noise of their control

Survey (NVIDIA Grandmasters Playbook 2025, stacking / hill-climbing posts, 2021-23 winning-ingredient summaries,
Kaggle Chronicles 2025) -> the techniques not yet used on Task 2, each as one whole-panel fine-tuning of p_PS s20
(`occ_lab/leap_t2.py`, 6k steps) against the identical control ctrl_s20, then note 106's sup+rec adaptation.
Adversarial weights: `occ_lab/adv_weights.py` (per-panel LightGBM train-vs-scored on window summaries of the released
history; a first version with the observed share reached AUC 1.000 on the Task 1 masks alone and gave uniform
weights - that run, adv_s20, is an extra copy of the control and measures the noise; the fixed version still has
AUC 0.93-0.98, weights concentrated on 10-20% of the train windows).

| ongoing IoU (val / priv) | single, all | single, days 1-5 | in AU's stack (replacing p_PS s20), all / days 1-5 vs AU |
|---|---|---|---|
| control | 0.856 / 0.829 | 0.878 / 0.879 | +0.0008 / +0.0007 |
| control again (uniform-weight run) | 0.855 / 0.829 | 0.878 / 0.879 | +0.0006 / +0.0003 |
| adversarial weighting | 0.853 / 0.827 | 0.876 / 0.877 | +0.0013 / +0.0010 |
| full data (ES tail trained on, last weights) | 0.858 / 0.830 | 0.880 / 0.883 | +0.0009 / +0.0013 |
| distillation from the 4 SSL seeds | 0.855 / 0.829 | 0.879 / 0.881 | +0.0008 / +0.0007 |
| EMA 0.999 | 0.855 / 0.830 | 0.878 / 0.879 | +0.0009 / +0.0006 |
| AWP 1e-3 | 0.856 / 0.827 | 0.875 / 0.879 | +0.0009 / +0.0013 |
| label smoothing 0.05 | 0.857 / 0.830 | 0.877 / 0.884 | +0.0006 / +0.0006 |
| mixup Beta(0.4) | 0.854 / 0.829 | 0.878 / 0.876 | +0.0003 / +0.0016 |
| masked TTA x8 (inference) | 0.856 / 0.828 | 0.877 / 0.878 | +0.0007 / +0.0015 |

The two identical controls differ by up to 0.0004 in the stack; every technique lands within +-0.001 of the control.
Only full-data retraining is weakly positive in all four single-member cells (+0.0012 all, +0.0029 days 1-5), i.e.
about +0.0001 total in the stack. Negative as a lever.
