# 75 - Plan B: idea 9b on the ongoing transformers. Adaptation hurts badly (2026-09-19)

Code: `tfm_lab/` (Ada), with `tfm_rec.py`, `eval_tfm_adapt.py` and `run_planB.sh`. Log: `tfm_lab/logs/report.txt`.

## Setup

- Model: the production transformer recipe (T2Tfm, whole-panel batches, BCE + 0.5 Lovasz, 30k steps, seeds 3-10)
  plus a reconstruction head `Linear(c, 2H)` on each link's final token, weight 0.3, 20% of cells hidden. All 8
  seeds trained normally (ES BCE 0.017-0.020, the same level as production).
- Adaptation: exactly 9b. Predict-then-adapt, 16 windows from strictly earlier days, 30 AdamW steps at 1e-4 on
  the reconstruction loss only. Every parameter except the queue head adapts.
- Metric: ongoing family delta vs candidate T's production tfm8, official-like windows, tau 0.50. "T-blend" puts
  the model in the 0.60 transformer slot.

| variant | val alone | val T-blend | priv alone | priv T-blend |
|---|---|---|---|---|
| tfm-rec, no adaptation | +0.0018 | -0.0021 | -0.0056 | -0.0020 |
| **tfm-rec, adapted (9b)** | **-0.0250** | **-0.0187** | **-0.0326** | **-0.0259** |
| production tfm8 + adapted, 1:1 | -0.0012 | -0.0026 | -0.0076 | -0.0077 |

All window CIs of the adapted rows are clearly negative.

## Reading

- The rec head by itself adds nothing (no adaptation: about 0 to -0.002 in the blend).
- **Adapting it is harmful.** The likely cause is architectural:
  - In the CNN, the rec head reads the block output **before** the history-collapsing conv and the queue head.
    Adapting the trunk for reconstruction moves a representation that the queue head sees only through the
    collapse layer.
  - In the transformer, the rec head and the queue head read **the same final token**. Reconstruction updates
    therefore move the queue head's direct input, and in a direction unrelated to queue prediction.
- Untested ideas that could change the picture:
  - Put the rec head on the GRU output, before the attention blocks, and adapt only the GRU (input calibration).
  - Use a much smaller learning rate or number of steps.
