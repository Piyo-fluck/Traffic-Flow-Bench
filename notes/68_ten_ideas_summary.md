# 68 - The ten ideas, all measured: one works in production (9b), two are usable, seven are not

Date: 2026-09-16 · Every idea the user listed was implemented and measured against a control that differs only in
the thing being tested (same seed, same steps, same architecture, same windows), and every production claim is on
the blank-aware official-like windows with a paired bootstrap CI.

| # | idea | standalone effect | in the production recipe | verdict |
|---|---|---|---|---|
| 1 | Task 1 trunk as a pretrained encoder | onset -0.066...-0.095 | - | **rejected** (note 66) |
| 2 | Task 1 reconstruction residual as features | onset +0.013...+0.038 | **onset +0.0039, priv all +0.0132** | **usable** (notes 63/64) |
| 3 | Task 1 ensemble spread | folded into 2; `t1_mix` is the top non-temporal feature | as 2 | **usable** (notes 63/64) |
| 4 | privileged teacher, soft-target distillation | onset CNN +0.006 / +0.014 | +0.0015 / +0.0005, CIs cross 0 | measured, **not bankable** (note 67) |
| 5 | history/future contrastive pretraining | onset -0.048...-0.051 | - | **rejected** (note 67) |
| 6 | onset as a likelihood ratio | missed-window top-1 0.05 -> **0.30** | rescue rule net negative | **not convertible** (note 65) |
| 7 | rare-bottleneck specialist (class-balanced) | missed-window top-1 0.05 -> 0.12 | +0.0029, CI crosses 0 | **not convertible** (note 65) |
| 8 | candidate-centered cross-panel motifs | top-1 0.75 with no panel id (M 0.76) | below M alone | **representation works, no gain** (note 65) |
| 9a | reconstruction head during training | CNN better everywhere | **no ensemble gain** | measured (note 67) |
| **9b** | **causal adaptation on past days** | **onset +0.0485 [+0.026, +0.072]** | **onset family +0.0065 / +0.0046, priv CI excludes 0** | **WORKS** (note 67) |
| 10 | joint Task 1/2/4 latent | onset -0.04 vs 9a | - | **rejected** (note 67) |

## Attribution for 9b

The adapted model carries two changes at once, so both were measured in the same blend:

| onset, 0.7 LightGBM + 0.3 CNN | validation | private |
|---|---|---|
| M production (CNN x3) | family 0.7797 | family 0.8181 |
| reconstruction head, **not** adapted | 0.7813 (+0.0019) | **0.8020 (-0.0043)** |
| reconstruction head, **adapted on past days** | **0.7862 (+0.0079)** | **0.8227 (+0.0189)** |

**The gain is the adaptation, not the head and not model-to-model variation.**

## What the ten ideas taught, as a rule

Three groups, and the group predicts the outcome:

1. **Change the initialisation** (1, 5) - always worse. The T2 window is 12 slots x links and 30k steps on a few
   thousand windows already saturate the model, so a better starting point buys nothing and a mismatched one hurts.
2. **Re-decide after the model** (6, 7, and the note-51 gating) - never converts. The scores genuinely rank the
   missed windows better, but the windows that need rescuing cannot be identified, so the rule breaks as many
   correct windows as it fixes.
3. **Add a signal that keeps acting** (2, 3 as features; 9b as weights that track the run) - this is where every
   gain came from. Idea 10 is the exception that proves it: an auxiliary objective pointing at a different target
   (latent demand) pulls the trunk away and costs 0.04 onset.

## What is now on the table

| change | total, local | status |
|---|---|---|
| ongoing ensemble with 8 transformer seeds (3:1:1, tau 0.50) | +0.0004 over R | measured, not built |
| onset: Task 1 residual features (scale-free) | +0.0006 val / +0.002 priv | measured, not built |
| **onset: past-day causal adaptation of the CNN** | **+0.0010 val / +0.0007 priv** | measured, needs a pipeline step |

Under the note-58 calibration (halve the local estimate; onset-side gains have never shown on the public LB) these
are worth roughly +0.001 on the public board and perhaps +0.002 on the private one, which is the board that decides
the standing.
