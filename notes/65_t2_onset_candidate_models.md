// filepath: notes/65_t2_onset_candidate_models.md
# 65 — T2 onset, ideas 6/7/8: dropping the location prior moves the missed windows from 0.05 to 0.30

Date: 2026-09-16 · Script: `analysis/t2_onset_candidate_models.py` · Data: `work/t2_onset_candidate_models.csv`

Three of the ideas share one frame, so they were built once: rows are (window, candidate) pairs, the label is the
candidate whose Voronoi cell covers most of the T+30 truth.

- **idea 8** — the features are written in a **candidate-centered coordinate system**: v/vf, q/cap, k/kcrit, lanes
  and on-ramp attachment at offsets −8…+8 links around the candidate, the cell means, the Task 1 residual block of
  note 63, hour and weekday. **No panel id, no absolute position, no candidate identity.**
- **idea 7** — class-balanced weights, so the train frequency of a bottleneck cannot be read off the labels.
- **idea 6** — a likelihood ratio: pairs (true candidate, other candidate) of the same window, sampled equally per
  class, trained on the feature **difference**.

19,242 training rows (12.2% positive) from the train split, 7,056 pairs.

| ranking model | all: top-1 / top-3 | **missed (40): top-1 / top-3** |
|---|---|---|
| M (production onset probability) | **0.76 / 0.98** | 0.05 / **0.88** |
| candidate model, unbalanced | 0.75 / 0.95 | 0.10 / 0.80 |
| **idea 7, class-balanced** | 0.75 / 0.96 | 0.12 / 0.88 |
| **idea 6, likelihood ratio** | 0.56 / 0.86 | **0.30** / 0.60 |
| (notes 59/60/63 for comparison) | 0.21–0.48 | past-day 0.17, path load 0.42, ORACLE demand 0.45, T1 residual 0.35 |

## Reading

1. **Idea 8 works as a representation.** With no panel id, no position and no candidate identity, the
   candidate-centered model reaches 0.75 top-1 against M's 0.76. A bottleneck really can be described in a way that
   transfers across corridors — which is what the earlier "structure features" (notes 22/27/50) failed to do,
   because they were extra columns in the old coordinate system rather than a change of coordinates.
2. **Idea 6 does exactly what it was designed to do.** Removing the base rate costs a lot overall (0.76 → 0.56) and
   buys the failure mode: on the windows M misses, top-1 goes **0.05 → 0.30**. Idea 7 is the milder version of the
   same trade (0.12).
3. **The decisive number is M's own top-3 on the missed windows: 0.88.** M already holds the true bottleneck in its
   first three candidates on almost every window it gets wrong. So the useful object is not a replacement ranker —
   none of these beats 0.76 — but a **re-ranking of M's top-3**, which note 51 measured as an oracle worth
   +0.03…+0.06 onset.

Top features of the balanced model: `hour`, then the Task 1 residual block (`t1_mix`, `t1_sig`, `t1_a`), then the
local demand (`qc_cell`, `qc+0`, `kc+4`) — the note-63 anomaly signal is the strongest non-temporal input here too.

## Next

A rescue rule: among M's top-3 candidates, re-rank with the likelihood-ratio score and move M's probability mass
towards the winner only when the margin is clear. Note 51 showed that swapping top-1 and top-2 blindly is negative
(−0.016…−0.024); the difference now is that there is a score with 0.30 top-1 on exactly those windows.

## The rescue rule fails: the signal does not convert into score (`analysis/t2_onset_rescue.py`)

Rule: among M top-3 candidates, re-rank with the balanced or the likelihood-ratio score and, when the winner is not
M top-1 and its margin exceeds `m`, multiply M probability inside the winner cell by (1 + beta). Grid over
m in {0, 0.05, 0.10, 0.20} and beta in {0.5, 1, 2}. M = 0.8212 on the 331 windows.

| rule | fires | of which on missed windows | best delta |
|---|---|---|---|
| balanced, margin 0 | 28/331 (0.08) | 12 | **+0.0029 [−0.0007, +0.0073]** |
| balanced, margin 0.10 | 13/331 | 5 | +0.0004 |
| likelihood ratio, margin 0 | 61/331 (0.18) | 20 | −0.0013 |
| likelihood ratio, margin 0.10 | 29/331 | 12 | −0.0031 [−0.0066, −0.0005] |

**Every setting has a CI that crosses zero, and the likelihood-ratio version is net negative even though it fires
on 20 of the 40 missed windows.** What it gains on those it loses on the windows M already had right: firing on a
correct window breaks it. This is note 51 result again — the windows to rescue cannot be identified — now with a
score that genuinely ranks better on the missed subset.

**Verdict for ideas 6, 7 and 8: measured, informative, not convertible.** Best available gain +0.0029 onset
(≈ +0.0004 total) with a CI crossing zero, which under the note-58 rule is not submittable.
