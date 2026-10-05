# 28 — Bias in the official-like onset eval: 27–30% of pseudo events had an empty history (official T2 blanks)

Date: 2026-09-15 · Scripts: `analysis/t2_first_onset_eval.py` (`official_blank`), `analysis/t2_official_like_eval.py`

- Every official Task 2 window blanks the Task 1 layer for its horizon and the hour after it (MASK_SPEC). The
  blank covers the whole corridor, 18 slots per window.
- The official onset windows are exactly each day's first morning queue. In every panel checked they sit on
  days 1–5.
- The first-morning-onset pseudo events (note 25) were detected on the visible layer. On official-window days the
  blank hides the real first queue, and the "no queue in the previous 90 min" test passes trivially on blank
  cells. So the detector picked artificial events right after a blank:

  | days < 14 | pseudo onset events | overlapping an official blank | history coverage: overlapping vs clean |
  |---|---|---|---|
  | validation | 81 | 22 (27%) | 0.00 vs 0.52 |
  | private | 80 | 24 (30%) | 0.01 vs 0.52 |

- **Notes 25–27 (first-onset and official-like onset numbers, E's 0.631 calibration, H/I/J estimates and the onset
  variants) include these empty-history events.** They are re-measured without them.
- Fix:
  - `first_onsets(..., blank=official_blank(panel, split, z))` skips days that hold an official onset window, and
    any candidate whose history, slot T or horizon touches a blank.
  - `ongoing_official(..., blank)` drops windows touching a blank. Ongoing was barely affected, because an ongoing
    window needs a visible history queue and a visible horizon queue.
- With the official onset days (1–5) removed, days 1–7 hold only about 2 events per panel. The main yardsticks
  become days 1–14 and all days.
- Re-run: `work/t2_official_like_eval_v2.log`, which also scores onset + past-day first-onset locations
  (β = 0.5, days < d only).

## Blank-aware results (family mean; days 1–7 now hold about 2 events per panel and are too noisy to use)

| | val d1–14 onset / ongoing / S_queue | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| E | 0.620 / 0.728 / 0.674 | 0.589 / 0.724 / 0.656 | 0.662 / 0.745 / 0.703 | 0.570 / 0.715 / 0.643 |
| G | 0.732 / 0.758 / 0.745 | 0.680 / 0.760 / 0.720 | 0.703 / 0.779 / 0.741 | 0.713 / 0.748 / 0.730 |
| H | 0.737 / 0.758 / 0.748 | 0.719 / 0.760 / 0.740 | 0.734 / 0.779 / 0.757 | 0.726 / 0.748 / 0.737 |
| I | 0.730 / 0.774 / 0.752 | 0.719 / 0.772 / 0.746 | 0.690 / 0.788 / 0.739 | 0.709 / 0.757 / 0.733 |

- **J = H's onset + I's ongoing**: S_queue 0.756 / 0.746 / 0.761 / 0.742, i.e. **+0.0245 / +0.0268 / +0.0174 /
  +0.0296 total vs E**, mean ≈ +0.025.
  - With the T1 month base (+0.003), J ≈ 0.853 LB (range about 0.846–0.858).
  - The earlier biased estimate was 0.846–0.853. The ordering E < G ≤ H and the stress gain on ongoing are unchanged.
- E's level rises to 0.656–0.674, against about 0.635 LB-implied. The blank-aware eval reads a little optimistic
  in level, but the gaps between methods are what the estimate uses.
- The past-day location boost is +0.0003…+0.002 and **cannot help the official windows**. They sit on days 1–5,
  and the earlier days' first morning queues are exactly the blanked official onsets, so no past-day events are
  visible for them. It is not adopted.
