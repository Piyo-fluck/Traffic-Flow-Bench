# 136 - LEAP 1st-place tricks (multiple input representations, confidence head) on the SSL w192 T2 CNN: negative

`occ_lab/leap_t2.py`: whole-panel fine-tuning of the p_PS checkpoints (s20, s21), 6k steps, with
- multirep: km/h, flow per lane, density proxy, log flow, per-link standardisation over the hour (RevIN-style),
  time differences, concatenated before a zero-initialised input conv;
- conf: a head predicting each cell's own BCE (MAE, training only);
- both; and the control (same fine-tuning, neither). Lab: note 106's sup+rec adaptation.

| ongoing IoU, sup+rec (val / priv) | s20 all | s20 days 1-5 | s21 all | s21 days 1-5 |
|---|---|---|---|---|
| control | 0.856 / 0.829 | 0.878 / 0.879 | 0.857 / 0.837 | 0.883 / 0.888 |
| conf | 0.855 / 0.830 | 0.880 / 0.883 | 0.858 / 0.838 | 0.885 / 0.886 |
| multirep | 0.856 / 0.829 | 0.879 / 0.880 | 0.857 / 0.839 | 0.888 / 0.888 |
| both | 0.856 / 0.831 | 0.879 / 0.883 | 0.856 / 0.840 | 0.884 / 0.884 |

Differences to the control are within +-0.004 with no consistent sign; ES BCE 0.0170 vs 0.0172 (s20), 0.0166 vs
0.0168 (s21). In AU's stack (replacing p_PS s20/s21): control +0.0015 all / +0.0012 days 1-5; conf +0.0013 /
+0.0014, multirep +0.0012 / +0.0003, both +0.0008 / +0.0008. The tricks add nothing over their control.

The control reproduces note 131's by-product (whole-panel fine-tuning of the shipped seeds: +0.0008 there, +0.0015
here in the stack), i.e. about +0.0002 total.
