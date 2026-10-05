# 170 The fitted DLSim generator as a physics expert at prediction time: negative

Date 2026-09-28. Code `occ_lab/dlsim_expert.py`, `occ_lab/dlsim_stack_eval.py`. Generator = note 169 (train-fitted
capacities). Per scored-month lab window: the day simulated from 00:00 with demand from that day's released flows
before T, then T..T+30 with the last level x the train weekday profile shape (causal; slot T unseen).

Ongoing family IoU, tau 0.5 (AU 0.8743 / 0.8579):

| use | val | pri |
|---|---|---|
| sim alone (absolute) | 0.3606 | 0.3255 |
| persistence | 0.6520 | 0.6321 |
| observed state + simulated change (T-5 -> T+h) | 0.5955 | 0.5895 |
| AU + 0.1 / 0.3 x simulated change | -0.0001 / -0.0016 | -0.0000 / -0.0019 |
| (1 - w) AU + w sim, w 0.1 | -0.0016 | -0.0032 |
| change member in p0, w 0.1 | -0.0054 | -0.0066 |

Onset (AP ratio rule 0.8160 / 0.8623): + w sim, w 0.1: 0.8117 / 0.8387; + w x simulated growth, w 0.05:
0.8134 / 0.8595. Every use loses.

The simulated 30-minute change is worse than no change at all (0.59 vs persistence 0.65): with capacities and
demand identified only to the level of note 169, the generator reproduces maps and run shifts but not the
30-minute evolution of a given window. Together with note 169's training-mix result, both uses of the generator
are negative.
