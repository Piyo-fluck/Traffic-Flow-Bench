# 139 - Boundary-aware losses for the queue edges: within the noise of the control

`occ_lab/leap_t2.py --bloss` on p_PS s20 (whole-panel fine-tuning, 6k steps; the same term also in the daily
adaptation). Edges are taken per horizon step along the link axis.
- sdf: Kervadec boundary loss, sigmoid(logit) x signed edge distance (+-8 links), weight ramped 0.01 -> 0.3
- band: BCE x3 on a +-2-link band around the true edges AND around the edge of the last observed history queue
  (known at inference; the edge cells carry most of the error, note 55)
- blov: an extra Lovasz on the band cells only; band+sdf: both

| ongoing (val / priv) | single all | single days 1-5 | stack vs AU, all / days 1-5 |
|---|---|---|---|
| control | 0.856 / 0.829 | 0.878 / 0.879 | +0.0008 / +0.0007 |
| sdf | 0.856 / 0.829 | 0.878 / 0.880 | +0.0002 / +0.0010 |
| band | 0.857 / 0.830 | 0.879 / 0.882 | +0.0005 / +0.0009 |
| blov | 0.857 / 0.832 | 0.877 / 0.882 | +0.0004 / +0.0004 |
| band+sdf | 0.856 / 0.830 | 0.878 / 0.882 | +0.0005 / +0.0008 |

Single-member differences +-0.002, stack differences +-0.0006 against the control: no effect. Weighting the edge
cells does not change what the network can resolve there.
