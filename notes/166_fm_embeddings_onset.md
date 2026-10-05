# 166 Idea (4): frozen time-series foundation models as onset candidate feature extractors

`occ_lab/fm_embed_onset.py` (per candidate: the link's released v/vf (and q/cap) over the 512 slots before T; weights
frozen) + `occ_lab/fm_rank_eval.py` (K principal components + their window deviations added to note 152's
IoU-LambdaRank + swap; cross-month, 6 seeds). Licences: Chronos-2 / TimesFM 2.5 / TTM-R2 / Toto Apache-2.0,
MOMENT MIT; Moirai 2.0 research-only and TiRex's NXAI licence not used.

| model (embedding) | K = 4 vs ranker (val / priv) | K = 2 |
|---|---|---|
| Chronos-2, 2 variates, mean over patches | -0.0006 / -0.0030 | **+0.0035 / +0.0001** |
| Chronos-2, last patches | -0.0029 / -0.0035 | - |
| MOMENT-1-large (30 % NaN rows, zero-filled) | -0.0015 / -0.0014 | +0.0000 / -0.0022 |
| TTM-R2 (interpolated) | -0.0019 / -0.0031 | +0.0015 / -0.0014 |
| TimesFM 2.5 (last layer, mean) | +0.0006 / -0.0076 | -0.0023 / +0.0008 |

Only Chronos-2 at K = 2 is non-negative in both months, inside the seed spread (+-0.0044) and chosen after seeing
the grid. The candidate ranking gains nothing reliable from a foundation-model representation of the series.
