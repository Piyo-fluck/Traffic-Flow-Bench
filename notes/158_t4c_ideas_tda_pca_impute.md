# 158 Ideas from Traffic4cast / shift literature: temporal domain adaptation, PCA link embeddings, impute-first

**Idea 2 - T4C 2021 temporal domain adaptation.** Input side: free-flow q/cap levels differ by only 2-6 % per link x
hour between training and scored months (|log ratio| p90 0.03-0.07), and are anti-correlated with where queues moved
(confounded) - the shift is in where / when queues occur, not in flow levels. Output side (`occ_lab/tda_output.py`):
AU's logit + beta log(month queued share / training share) per link (+-1) x time (+-30 min), shrunk by kappa:
all grid points within +-0.002; cross-month +0.0010 / -0.0013 (all), +0.0007 / -0.0000 (days 1-5). Nothing.

**Idea 1 - data-derived link identity (PCA embeddings, 2411.11448).** `analysis/t2_cnn_pca.py` (T2CNN_PCAEMB=8):
per link, queued share + mean v/vf per hourly bin 06-20 h over the same month's last <= 4 earlier days, projected on
the panel's PCA basis; recomputed from the scored month's released days at scoring. Same recipe, seeds 30 / 31 (Ada,
occ_lab/logs/pca), ongoing tau 0.65 lab (val all / priv all):

| | ES bce | ongoing val all | ongoing priv all |
|---|---|---|---|
| control | 0.0311 / 0.0316 | 0.760 / 0.765 | 0.738 / 0.744 |
| v1 input channels | 0.0387 / 0.0409 | 0.725 / 0.703 | 0.707 / 0.688 |
| v2 whitened + clipped + 50 % dropout | 0.0370 / 0.0375 | - | 0.733 / - |
| v3 late fusion (zero-init 1x1 after the collapse) | 0.0315 / 0.0324 | 0.749 / 0.774 | 0.731 / 0.745 |

Input-level injection hurts even in distribution (a near-unique per link-day code the net memorises); late fusion
removes the harm but adds nothing (seed-level noise).

**Idea 3 - impute first (T4C 2022 winner).** `occ_lab/fill_diag.py` on the training run's ES windows: the training
histories themselves are 76.4 % observed (eligibility is ~76.4 % everywhere; truth exists for only 30 % of the
ineligible cells), the same as the official windows - there is no train / official observation mismatch to close.
Extra official-style masking (to 58 %): cnn3 ongoing 0.9207 -> 0.9079, linear-interpolation fill recovers 0.9099
(15 % of the loss); onset unchanged (0.808-0.809). A full-state imputer has no target for most missing cells and
little to win.
