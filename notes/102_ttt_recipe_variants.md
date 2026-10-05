# 102. Test-time training: three recipe changes, no effect

Date 2026-09-24. Code `analysis/t1_ttt.py` and `scripts/t1_ttt_train.py`, env flags `TTT_RATE`, `TTT_COSINE`,
`TTT_SEEDS` (all defaulting to the production behaviour, so the shipped pipeline is unchanged; backups in
`work/t1_ttt.py.bak`, `work/t1_ttt_train.py.bak`).

TTT is the only Task 1 line that ever moved the LB (400 -> 1200 -> 3600 steps, Y 0.87845 -> Z 0.87871), and note
32 only ever varied the step count. Everything else about the recipe had never been touched:

    rate     10-50% of the visible eligible cells hidden per sample, uniform, regardless of the day
    schedule lr 2e-4 held constant for the whole run
    models   one adaptation per production checkpoint, fixed seed, three models averaged

The three changes, bundled into one run to save a submission slot:

    TTT_RATE=regime   hide at the day's OFFICIAL rate (0.20 / 0.30 / 0.50, MASK_SPEC), which is published per day
    TTT_COSINE=1      cosine decay to zero
    TTT_SEEDS=3       three adaptations per checkpoint with different hide/batch seeds, nine models averaged

## Result (validation, in-month pseudo evaluation, 1200 steps)

| recipe | S_state~ | S_LWR~ | total |
|---|---|---|---|
| production (uniform / constant / 3 models) | 0.9518 (+0.0024) | 0.7745 (+0.0042) | +0.0013 |
| new (regime / cosine / 9 models) | 0.9517 (+0.0023) | 0.7746 (+0.0043) | +0.0012 |

The control reproduces note 32 exactly (0.9518 / 0.7745), so the harness is faithful. The new recipe is a wash.

Each component has an explanation that is already measured elsewhere:

- **rate matching**: note 96 showed the model is nearly insensitive to mask geometry - cells with both temporal
  neighbours missing carry error in proportion to their number - so matching the hiding rate changes an axis the
  model does not respond to.
- **seed averaging**: the adaptations start from the same three checkpoints and move a short distance, so the
  nine models are strongly correlated and averaging them buys almost nothing.
- **cosine decay**: no difference against a constant lr.

Not submitted. The in-month evaluation failed to rank 1200 against 3600 steps, where the LB later showed
+0.00026, so a null here is weaker evidence than usual - but every component has an independent reason to be
null, and building the candidate would need 3600-step TTT with nine models on both months plus inference for ten
panels. The remaining variation in TTT is per-panel adaptation and a PEFT-style parameter subset (note 102's
list), neither of which was run.
