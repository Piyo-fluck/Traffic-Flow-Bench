# 157 C: mean-field switching model (explicit free / queued Markov chain per link, learned hazards)

Script: `occ_lab/msw_t2.py` (Ada, occ_lab/logs/msw; seeds m0 / m1, 12k steps, ES bce 0.047-0.048); caches
fam_msw_m0/m1/avg. Per-link temporal conv encoder + two link convs; pi_0 from the last queue flag; per step learned
onset / clearance hazards from h_j and the neighbours' pi (+-2).

| | val | priv |
|---|---|---|
| alone, all (m0 / m1) | 0.754 / 0.762 | 0.728 / 0.735 |
| alone, days 1-5 | 0.809 / 0.818 | 0.789 / 0.740 |
| avg in AU's p0, w 0.05 | -0.0017 | -0.0008 |
| w 0.10 | -0.0032 | -0.0020 |
| w 0.20 | -0.0060 | -0.0065 |

~0.1 below the single production members and negative in the stack at every weight (spoils > rescues).
