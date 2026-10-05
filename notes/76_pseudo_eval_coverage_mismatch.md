# 76 - The official-like onset pseudo-evaluation is shifted: its histories are 52% observed, the official ones 76%

Date: 2026-09-19. The user asked whether the private pseudo-evaluation might be off.

## Facts

- The official queue label threshold is `v_cut` from fd_parameters when present, else 0.6 x free speed
  (`trafficflowbench-public/src/task2/queue_utils.py`). No panel has a `v_cut` column, so the threshold is
  identical to ours.
- Links vs stations: 1:1 on every panel.
- **History coverage** is the share of the 12 x S history cells that are eligible and observed (speed and flow).

| | mean coverage |
|---|---|
| official onset windows, from window_history.parquet | **0.761** |
| the same official windows, read from the masked corridor layer | 0.761 (identical) |
| official ongoing windows | 0.765 |
| **our pseudo onset windows** (`first_onsets`, blank-aware) | **0.524**, per-panel 0.48-0.61; max 0.64-0.67 |

- The config has `minimum_history_coverage: 0.7`. **Not one pseudo onset window reaches 0.7.**
- Cause: the Task 1 target masks. Share of eligible cells that are visible, by day regime: R1 about 0.78-0.80,
  R2 about 0.69, R3 about 0.50.
  - Official windows sit on R3 days too, yet their histories are 0.76 observed. So the organizer keeps Task 1
    target cells out of the official Task 2 histories, or picks origins where they are absent.
  - Pseudo windows inherit the full Task 1 masking.
- Training is not affected: it uses the true layer with eligibility, whose coverage is about the official level.
  **So training matches the official inputs; the local pseudo-evaluation is the odd one out.**

## Consequences

- Every local onset comparison (U, X, AB, the ratio c, the delta correction, w_on) was measured on windows with
  about a third fewer observed history cells than the official ones. The ranking of decision variants can differ
  there. This is a concrete candidate reason why three onset decision changes looked positive locally and lost on
  the LB.
- A coverage filter cannot fix it inside the scored months: no pseudo window qualifies.
- Options for an LB-aligned onset check:
  - Train-month time-split OOF (coverage is right, but there is no run shift; this is what sim_lab report 22 used).
  - Re-mask the train month with official-like availability.
  - Treat the scored-month pseudo-eval only as a run-shift check.
