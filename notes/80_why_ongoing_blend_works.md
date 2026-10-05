# 80 - Why the ongoing blend beats every single model (measured)

Date: 2026-09-20. Cached probabilities, official-like ongoing windows (validation 3351 / private 3374).
Each model at its own best threshold; the blend is the production 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 at tau 0.50.

## 1. The three models have different error profiles

| validation | IoU | missed queue cells / true | false alarms / true |
|---|---|---|---|
| gs1 (global-state LightGBM) | 0.8105 | 0.057 | **0.100** |
| tfm8 (transformers) | 0.8420 | 0.055 | 0.062 |
| cnn3 | 0.8012 | **0.094** | **0.040** |
| **blend** | **0.8514** | **0.051** | 0.056 |

- gs1 over-predicts, cnn3 under-predicts, tfm8 sits between them.
- **The blend is better than every single model on BOTH axes at once** (fewest missed cells and fewer false
  alarms than gs1 or tfm8). Private is the same picture.

## 2. Their errors are only half-shared

Cell-level error correlation: gs1-tfm8 0.50, cnn3-tfm8 0.53, cnn3-gs1 0.41. "Both wrong" is 0.009-0.010 of cells
against 0.016-0.023 wrong for each model alone, so roughly half of each model's errors are its own.

## 3. The whole gain sits on the 2.9% of cells where they disagree

| | validation | private |
|---|---|---|
| eligible cells | 1,933,834 | 1,944,006 |
| cells where the three predictions differ | 2.9% | 2.9% |
| of those, truly queued | 44% | 47% |
| accuracy there: gs1 | 0.466 | 0.467 |
| accuracy there: cnn3 | 0.575 | 0.550 |
| accuracy there: tfm8 | 0.659 | 0.676 |
| **accuracy there: blend** | **0.703** | **0.715** |
| accuracy on the 97.1% agreed cells | 0.9936 | 0.9925 |

- Agreed cells are already 99.3% right, so nothing can be won there.
- On the disputed cells the blend beats the best single model by 4 points. These are near-coin-flip cells (44-47%
  queued), i.e. the queue boundary, which note 52 identified as the main error mass.

## 4. What this implies

- The blend works by **mixing opposite biases and decorrelated errors**, not by calibration (both tfm8 and the
  blend are already well calibrated: predicted 0.55 -> actual 0.51-0.53).
- A new model only helps if it is wrong in a *different* place. That is why the extra-data CNNs (note 77), the
  same-day and cross-panel adaptations (note 78) and the adapted transformers (note 75) changed nothing: they are
  variants of models already in the blend, so they mostly move cells that were already agreed.
- The per-window oracle over the three singles is 0.886 / 0.881 against the blend's 0.851 / 0.850, so per-window
  selection still holds +0.035, but every selector tried so far has failed to capture it (notes 51, 53, 73).
