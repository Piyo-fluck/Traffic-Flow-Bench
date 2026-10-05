# 72 - Results of the test-time adaptation experiments A-G (run by the user on Ada, 2026-09-17)

Sources: Ada `tta_lab/reports/00-09*.md`, raw tables in `tta_lab/runs/verify_*/report.txt`. The plan is note 71; the
code and docs are in `tta_lab/`.

**Outcome: nothing beats candidate T on both months.** Idea 9b itself is confirmed: without adaptation the onset
T-blend drops by -0.0049 on validation and -0.0207 on private.

| exp | best setting | onset T-blend, family delta (val / priv) | verdict |
|---|---|---|---|
| 9b replayed | default | +0.0000 / -0.0012 (the -0.0012 is the adaptation RNG alone) | reference |
| A | Task 1 residual, causal, scale-free, hidden | adapted minus global +0.0040 / +0.0004; no arm beats base | no |
| B | micro-adaptation, K=3, lr 1e-5 | +0.0000 / +0.0003 | no |
| C | norm / incal / last2 / film / lora / adapter / inp | every mode worse on at least one month | no |
| D | loss = mask + 1.0 x next, k=1 | +0.0026 / +0.0021, CIs include 0, ongoing T-blend unchanged | weak |
| E | FOMAML 3000 steps, inner SGD 5 x 1e-3 | -0.0096 / -0.0103; with and without adaptation are identical | no, see below |
| F | 4 trajectories, uniform mean or gated | private onset alone +0.0055, but T-blend -0.0024 | no |
| G | - | not implemented | - |

Readings:
- **A (important).** Once the leak is removed (Task 1 sees only the 12 history slots), the Task 1 residual no longer
  reproduces the note 63/64 gains. The positive result of ideas 2 and 3 was most likely carried by the ~107-slot
  temporal receptive field, which sees the pseudo-window labels. Never use `work/sub_m3*` day reconstructions as
  Task 2 inputs.
- **D.** Its gain is the same size as the RNG spread of 9b itself (0.0012 family). Deciding it needs several
  adaptation seeds for both 9b and D. Expected LB gain is about 0.15 x 0.0023 x 0.5 = +0.0002.
- **E was not a real test.** Plain SGD at 1e-3 for 5 steps on a reconstruction loss of about 0.004 barely moves the
  weights, and the evaluation shows exactly that: adapted equals unadapted. The inner loop was therefore close to
  inactive during meta-training too. What was measured is "further Task 2 fine-tuning without early stopping",
  and that made the model worse. A real test needs an inner step that is actually effective: a larger alpha
  (1e-2 to 1e-1), a learned per-layer alpha, or Adam-like inner steps. It should be checked first that
  adaptation changes the predictions.
- **F.** An ensemble helps the CNN on its own but hurts inside the 0.7/0.3 blend, which suggests the flatter
  probabilities interact badly with the ratio rule.

## E re-check (2026-09-17, stopped early by decision)

The inner loop is the bottleneck, so it was probed before any more meta-training. The probe used rec_s20
without meta-training, restarting from the checkpoint every day (`--reset-daily`) with 5 steps.
Panels were D7_I10_E and D7_I210_W, both months. The table gives the mean |p - p_noadapt|.

| inner step | adapt loss (start is about 0.005) | onset | ongoing |
|---|---|---|---|
| SGD 1e-3 (the first E run) | - | ~0 (evaluation identical to no adaptation) | - |
| SGD 3e-3 | 0.3-2.2, **diverging** | 0.0011 | 0.0026 |
| SGD 1e-2 / 3e-2 / 1e-1 | 1e4 to inf / nan (meta-training also went nan from step 100) | 0.15 (broken) | 0.18 |
| AdamW 1e-4 x 5 steps | 0.006-0.009 | 0.0008 | 0.0012 |
| 9b, cumulative (30 AdamW steps per day, carried forward) | - | **0.0089** | 0.0117 |

- **Plain SGD has no usable step size.** At 1e-3 it does nothing, from 3e-3 it diverges.
- The stable normalised inner loop (AdamW, 5 steps from the checkpoint) moves the predictions about **10x less
  than 9b**. 9b's effect builds up only through about 30 days x 30 steps of cumulative adaptation.
- A MAML-style outer loop over a 5-step inner loop therefore works on a lever about a tenth the size of the
  one that gave 9b's gain. Unrolling an inner loop as long as 9b's is not feasible.
- Expected upside is small, so **E is stopped** (user rule: stop immediately if the upside is small). D was also
  dropped by the user, gain too small.
- The half-finished E2 runs were stopped: `tta_lab/runs/E2_*`, driver `tta_lab/run_E2.sh`.
