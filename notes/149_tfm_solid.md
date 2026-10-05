# 149 SOLID v3 for the transformer family (tfm8)

Ada run `occ_lab/logs/tfmsolid` (pseudo_adapt_tfm.py --solid3, seeds 3-10; norm + head fitted per window on its 32
closest released pseudo windows, post-block link tokens); eval `occ_lab/tfm_solid_eval.py`.

| ongoing IoU | all val | all priv | days 1-5 val | days 1-5 priv |
|---|---|---|---|---|
| single tfm8 shipped | 0.8575 | 0.8350 | 0.8682 | 0.8824 |
| single tfm8 SOLID v3 | 0.8582 | 0.8366 | 0.8696 | 0.8822 |
| stack AU | 0.8743 | 0.8579 | 0.8907 | 0.8950 |
| stack tfm8 -> SOLID v3 | +0.0001 | +0.0001 | -0.0002 | +0.0001 |
| stack AX | +0.0024 | +0.0031 | +0.0027 | +0.0013 |
| stack AX + tfm8 SOLID v3 | +0.0026 | +0.0028 | +0.0032 | +0.0012 |

The single member gains +0.0007/+0.0016 (all) but nothing survives the stack (tfm8 has weight 0.24 x 0.4 and the
corrector); AX + tfm SOLID = AX within +-0.0005. Not built officially.
