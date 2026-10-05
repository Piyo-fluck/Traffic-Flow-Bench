# 41 — Data re-examination: Task 4 counts do not explain the run's queue shift; contract and history files hold nothing new

Date: 2026-09-15 · Measurement only (inline checks; numbers below)

## Task 4 link counts as a signal of the run's demand

- `task4/<panel>/<split>/synthetic_link_counts.csv` gives one count per link for the **PM demand period**
  (`SYN_PM_TRAIN` / `SYN_PM_VALIDATION` / `SYN_PM_PRIVATE`). The counts differ per split, so they reflect each run's
  demand draw. Example, link L5N-186: 3,458 in train, 3,989 in validation, 3,728 in private.
- The ratio split/train per link has a relative sd of only 4–7% within a panel. Spearman ρ, over active links, of that ratio
  against the run's queue-share change (averaged over 8 panels × 2 splits):
  - **+0.071** over all hours
  - **+0.034** in the morning
  - **+0.039** against the first-morning-onset frequency

  Individual panel-splits range from −0.30 to +0.34.
- **Conclusion:** the counts carry almost no information about where the run's queues move. They are not useful
  for onset, so the rule question does not arise.

## Other files

- `synthetic_network_manifest.json`: structural inputs only (path set, incidence, connectivity). Nothing
  run-specific.
- `config/competition_contract.json`: consistent with everything we use. The ongoing/onset rule is "≥ 2 queued
  observations with ≥ 2 on one link", the score is space-time IoU, and every aggregation is an equal mean.
- `window_history.parquet` holds mainline links only; no ramp rows are dropped by our loader.

## The one unused legitimate source found

**History-period ramp flows (T−60 … T−5)** from the split's `ramp_states`. Only horizon ramp flows are excluded.
These are pre-origin data, but same-day data of the scored month, so they are measured first:
`analysis/t2_ramp_hist.py` (paired H/I ± ramp features on the official-like windows).

**Result** (log `work/t2_ramp_hist.log`). The ramp features are attached on-ramp inflow over the last 3 slots (mean,
max, trend), the maximum over the next 3 downstream links, off-ramp outflow and attachment flags:

| IoU (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| onset base → +ramps | 0.793 → 0.749 (1/2) | 0.765 → 0.700 (4/4) | 0.824 → 0.795 (1/4) | 0.798 → 0.750 (2/5) |
| ongoing base → +ramps | 0.775 → 0.778 (5/3) | 0.774 → 0.774 (5/3) | 0.788 → 0.795 (6/2) | 0.757 → 0.765 (6/2) |

- Onset gets clearly worse everywhere (−0.030…−0.066), although the model gives the ramp features 7.7% of the
  gain. Ramp inflows are run-specific, so the model learns train-run associations that do not transfer. This is
  the same failure as the train priors (note 17).
- Ongoing changes by only +0.000…+0.008. **Not adopted.**
- The data re-examination is complete. No unused legitimate information in the release improves Task 2.
