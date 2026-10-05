# 159 AU's ongoing error anatomy by queue phenomenon

Script: `occ_lab/og_anatomy.py` (lab official-like pool, AU with the corrector, tau 0.5; T-queue = queued cells of
the last history slot, observed only; upstream per panel from where true queues grow - lower index on 7 panels,
higher on D7_I10_E).

| class | val loss share | priv loss share | error rate | missed cells with p >= 0.3 / p < 0.1 |
|---|---|---|---|---|
| stay | 0.067 | 0.068 | 0.009-0.010 | 0.66 / 0.07 |
| grow-tail (upstream, <= 2 links) | 0.067 | 0.063 | 0.018 | 0.58 / 0.12 |
| grow-head (downstream, <= 2 links) | 0.029 | 0.031 | 0.019-0.024 | 0.45-0.56 / 0.15-0.22 |
| **grow-new (> 2 links from any T-queue)** | **0.307** | **0.336** | 0.17-0.21 (FN) | 0.35-0.37 / 0.25-0.29 |
| clear-head | 0.168 | 0.148 | 0.21-0.23 (FP) | - |
| clear-tail / clear-inner | 0.026 / 0.006 | 0.025 / 0.005 | 0.16-0.21 | - |
| **far (false alarms away from any T-queue)** | **0.330** | **0.324** | 0.004-0.005 (FP) | - |

About two thirds of the loss is queues appearing (missed) or predicted (false) AWAY from the visible queue - an
onset-like problem inside the ongoing windows - and a sixth is the head clearing. Boundary dynamics near the visible
queue (tail / head growth, persistence) are nearly solved (3-7 % each). Unobserved links at T are not where the far /
new errors concentrate (hidden +-1 at T: 83-90 % of the cells and of the errors); links with no observation in the
last 15 min hold 21-22 % of the far false alarms vs 12 % of the cells (lab histories are 52 % observed, official 76 %).
