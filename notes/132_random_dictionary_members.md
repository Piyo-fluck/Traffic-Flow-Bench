# 132 - Random-dictionary members (HYDRA-2D, echo state network) with an exact run head: negative

Proposal (user-supplied list, items 1 + 3): a fixed random feature extractor, so its errors differ from every learned
member, plus a run-specific linear head. `occ_lab/hydra_t2.py`:

- features: HYDRA (Dempster et al. 2023) extended to 2-D kernels (3 slots x 5 links, link dilation 1/2/4) over
  v/vf, q/cap, observed, queued and their time differences; 16 groups x 8 kernels, soft and hard counts, pooled over
  the hour and the last 20 min and over +-2 links (3,072 features), or an ESN (1,024 units, final + mean state over the
  hour, input = the link and +-2 neighbours); optionally the link's own raw history (48); time, weekend, static.
- head (a): logistic layer per horizon step (ongoing) / at T+30 (onset) on the train run (cells within 8 links of a
  queue, 5% of the rest).
- head (b): before each day, one exact Newton step of the logistic loss on every released pseudo window of the run so
  far, from sufficient statistics accumulated day by day (A = sum p(1-p) phi phi^T, b = sum (y-p) phi, per step),
  lambda 1e2..1e5.

| single member, ongoing (val / priv) | head (a) | + run head, best lambda (1e4) |
|---|---|---|
| HYDRA h0 | 0.585 / 0.581 | 0.632 / 0.611 |
| HYDRA + raw history, 16 epochs (r0) | 0.652 / 0.644 | 0.661 / 0.637 |
| ESN + raw (e0) | 0.646 / 0.630 | 0.665 / 0.637 |
| (CNN, frozen, for scale) | ~0.81 / 0.79 | |

Onset heads 0.05-0.23 alone. In AU's p0 at 0.05: every variant -0.0014..-0.0019 (val) / -0.0005..-0.0015 (priv),
more windows worse than better (e.g. 593 / 905); at 0.10-0.20 steadily worse. Onset blend: r0+run head at 0.10
+0.0086 / +0.0008 on one rescued window - noise on the onset surface (note 108).

A linear readout of a random dictionary is 0.15-0.20 IoU below the learned members; the different representation
does not survive that gap. The exact run head helps the weak member (+0.01..+0.05) but not the stack. Negative.
