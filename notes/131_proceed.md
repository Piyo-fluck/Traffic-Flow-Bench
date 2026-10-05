# 131 - Proceed (KDD 2025) on the SSL w192 CNN: negative; a by-product of its control

`occ_lab/proceed.py`: concept encoders E(X_r, Y_r) / E'(X_t) (per-link MLP, panel average), drift
delta = E'(X_t) - E(X_r, Y_r), bottleneck generator (r 32, zero-init W2) producing per-layer channel coefficients
alpha (input) x beta (output) for the 16 block convs, the collapse and the head (paper eq. 1-4). Trained jointly with
the forecaster on the train run, whole panels, bs 16, 6k steps; references half the target day's latest labelled
window (origin T-35 min), half random windows (synthetic drifts). Lab: note 106's causal per-day sup+rec fine-tuning
(generator off), then each window predicted with coefficients from its own drift against the same day's T-35 window
of the released layer. Control: identical training with the generator off.

| ongoing family IoU, sup+rec (val / priv / mean) | s20 | s21 |
|---|---|---|
| p_PS as shipped | 0.8364 | 0.8422 |
| control (whole-panel fine-tuning, no generator) | **0.8427** | **0.8469** |
| Proceed | 0.8416 | 0.8459 |

Proceed vs its control -0.0011 / -0.0010: **the same-day drift carries nothing usable** (as note 124's same-day
adaptation, +0.0007). In AU's stack, Proceed replacing p_PS s20/s21 +0.0010 and the control +0.0008.

By-product: 6k more steps of whole-panel fine-tuning of the SSL w192 checkpoint (no 64-link crops) beats the
shipped seeds by +0.0063 / +0.0047 single, +0.0008 in the stack (early days +0.0013). About +0.0001 total - below
anything worth a submission on its own.
