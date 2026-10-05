# 143 - Shift-robust training (E): V-REx over months, TADA time warps, engression, run-context hypernetwork, TESTAM-style MoE, OneNet

`occ_lab/leap_t2.py` (p_PS s20, whole-panel 6k-step fine-tuning, note 106 adaptation, vs ctrl_s20):
- vrex: batches of 4 train months x 4 windows, loss = mean + 10 var of the month risks (environments known, not inferred)
- tada: worst of 3 monotone time warps of the history (ends fixed) added to each batch
- engr: pre-additive Gaussian noise (sd 0.05) on observed v/vf, q/cap, two samples, scored on their mean probability
- hyper: channel-wise FiLM of the collapsed features from 4 run-context scalars (queued share and mean v/vf over the
  3 h before the history and the same hours of the previous day), identity at start; context passed in adaptation too
- moe: a per-link temporal expert (own 12-slot history only) mixed with the CNN by a per-link gate (starts on the CNN)
`occ_lab/onenet_eval.py`: run-level Hedge weights over the seven members on the windows released so far.

| ongoing | single all (val / priv) | single days 1-5 | stack vs AU, all / days 1-5 |
|---|---|---|---|
| control | 0.856 / 0.829 | 0.878 / 0.879 | +0.0008 / +0.0007 |
| V-REx | 0.855 / 0.827 | 0.877 / 0.878 | +0.0009 / +0.0017 |
| TADA | 0.859 / 0.830 | 0.880 / 0.875 | +0.0016 / +0.0016 |
| engression | 0.858 / 0.828 | 0.877 / 0.879 | +0.0015 / +0.0010 |
| hypernetwork | 0.852 / 0.825 | 0.876 / 0.876 | +0.0002 / +0.0003 |
| MoE | 0.851 / 0.824 | 0.876 / 0.873 | +0.0005 / +0.0006 |
| OneNet Hedge (decision layer, eta 100, gamma 0.7) | | | +0.0009 / +0.0011 all, +0.0003 / +0.0003 days 1-5 |

Against the control (two identical controls differ by up to 0.0004): TADA +0.0008 / +0.0009 and engression
+0.0007 / +0.0003 in the stack, V-REx +0.0001 / +0.0010; the hypernetwork and the MoE lose. Run-level online weights
(OneNet) reach half of what note 141's per-window selection does: locality is what the selection gains from.

## Are TADA and engression independent? No.

- Their changes against the control correlate at 0.458 per eligible cell; of the 50,655 cells the control gets
  wrong, TADA fixes 5,374 and engression 5,232, 2,779 of them the same (53% of the smaller); each also breaks about
  as many (4,924 / 6,138, 2,412 shared).
- The mean of the two in the stack: +0.0007 / +0.0010 all, +0.0018 / -0.0003 days 1-5 vs the control - no more than
  either alone.
- Both in one training run (`--tada --engr 0.05`): single 0.857 / 0.830 all, 0.874 / 0.876 days 1-5; stack +0.0003 /
  -0.0002 vs the control. Combined, they lose what each had - the individual gains are fine-tuning noise, not two
  separate mechanisms.

TADA as an eighth candidate inside AX's per-window selection (so it is used only where it did well on similar
released windows): vs AU, AX +0.0024 / +0.0031 all, +0.0027 / +0.0013 days 1-5; + TADA member at prior 0.02
+0.0026 / +0.0031, +0.0027 / +0.0013; at prior 0.10 +0.0031 / +0.0029, +0.0026 / +0.0012. No gain on the days that
matter; not added.
