# 21 — Task 2: CTM from released parameters fails; bottleneck discharge capacity transfers across runs

Date: 2026-09-15 · Scripts: `analysis/t2_ctm_sim.py` (PoC), `analysis/t2_capacity_transfer.py` · Log: `work/t2_capacity_transfer.log`

## CTM forward simulation (PoC, D7_I10_E, validation days < 14)

A cell-transmission model was run from the T−5 state through T+30. It used the released FD parameters, the train
flow and on-ramp profiles scaled by today's level, off-ramp split fractions, and a Daganzo merge. It uses no
same-day ramp data and nothing from the horizon.

| Variant (600 onset+ongoing windows) | sim queued share (truth 0.051) | IoU |
|---|---|---|
| init = observed k, cap = fd | 0.085 | 0.099 |
| init = profile k, cap = fd | 0.021 | 0.003 |
| init = observed k, cap = max(fd, train q99.9) | 0.068 | 0.086 |
| persistence of the last visible queue flags (ongoing) | — | 0.794 |

- The ramp/off-ramp bookkeeping is right: it reproduces the train profile's link-to-link flow balance with a
  median error of 1.1% (max 5.3%).
- The false queues form upstream of real bottlenecks, e.g. s20 (3 lanes, cap 5,756, a busy train bottleneck).
  Peak demand there is about 5,720, so an error of under 1% in the demand decides whether a queue forms. In the
  validation run s20 is not active.

## Discharge capacity at queue heads (link s free, s−1 queued; median flow at s)

| Panel | links with a discharge in both train and the split | median \|ratio − 1\| | corr | top-8 head-link overlap |
|---|---|---|---|---|
| D7_I10_E | val 4 / priv 5 | 0.009 / 0.005 | 0.961 / 0.988 | 4 / 7 |
| D12_I5_N | 3 / 3 | 0.001 / 0.003 | 0.999 / 0.996 | 5 / 5 |
| D7_I405_N | 13 / 4 | 0.003 / 0.004 | 0.998 / 1.000 | 4 / 5 |

- **Discharge rates are a network property: they match between runs to within about 0.5%.** Which bottlenecks
  are active changes, because demand changes.
- The discharge rate often equals the released capacity of the head link (s20 5,758 vs 5,756; s350 5,813 vs
  5,832; many at about 7,660 vs 7,675). Several bottlenecks sit well below it (s82 5,757 vs 13,162; s11 5,585 vs
  7,675; s115 6,311 vs 10,524; s240 6,247 vs 7,949).

## What it changes

- Direct simulation needs demand accurate to under 1%, which the noisy, sparse history cannot give. Drop the CTM
  as a direct rule.
- Build features around the transferable quantity instead: effective capacity (train discharge where known, else
  the released capacity), and the margin between today's measured flow and the tightest downstream capacity,
  with its trend and a profile-projected value. The experiment is `analysis/t2_struct_noid.py`.
