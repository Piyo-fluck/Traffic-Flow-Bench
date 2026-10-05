# 142 - Critical-slowing-down (early-warning) features for the onset bottleneck: no signal in this simulator

`occ_lab/bottleneck_csd.py`: per candidate peak (note 129) and its +-2 links, on the detrended history hour of v/vf,
q/cap and density proxy: variance, lag-1 AC, skewness, and the variance RISE (last 30 min minus first 30 min); plus
spatial correlation / leading-eigenvalue share (could not be formed: too few fully observed slots over 5 links).

| cross-month | top-1 (val / priv) | swap onset IoU vs production | rescue / spoil |
|---|---|---|---|
| production | 0.769 / 0.840 | 0 | |
| note 129 features | 0.817 / 0.877 | +0.0132 / +0.0124 | 4 / 0 |
| + early-warning | 0.805 / 0.852 | +0.0166 / +0.0154 | 4 / 0 |

The classifier gets WORSE at picking the bottleneck; the IoU uptick is within-window extent noise on the same 4
rescued windows. Direct check - the true first bottleneck's relative rank among its window's candidates (0 = the
highest value, 0.5 = chance, n ~ 250-315): speed variance 0.479, speed AC1 0.477, speed skewness 0.500, speed
variance rise 0.471; flow / density variance 0.34 (a level effect: busier bottlenecks), their AC1 0.45, rises 0.46;
production mass 0.068. No critical-slowing-down signature at 5-min resolution (consistent with note 133: queue speed
sd 0.5 km/h, no oscillation). Negative.
