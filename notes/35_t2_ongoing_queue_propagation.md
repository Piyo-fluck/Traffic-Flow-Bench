# 35 — Task 2 ongoing: queue-propagation (ETA) features add only +0.0025 ongoing IoU

Date: 2026-09-15 · Script: `analysis/t2_ongoing_qprop.py` · Log: `work/t2_ongoing_qprop.log`

The new features come from the last 6 history slots of visible queue flags:
- the distance to the nearest queued link downstream and upstream, and how fast the downstream distance shrinks;
- the estimated arrival time of the queue (ETA) and whether it arrives by each horizon step h;
- for queued links, the distance to the tail and the estimated clearance time.

Two no-prior stress ongoing models (= I) were trained on identical sampled rows, without and with these
features, and scored on the blank-aware official-like ongoing windows (τ 0.65).

| ongoing IoU (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| I (stress) | 0.771 | 0.772 | 0.783 | 0.754 |
| + queue propagation | 0.775 (+0.003, 5/3) | 0.775 (+0.003, 5/3) | 0.780 (−0.003, 6/2) | 0.761 (+0.007, 7/1) |

- Most panels improve, but one panel drops enough to turn priv d1–14 negative. The mean is about +0.0025
  ongoing, i.e. +0.0004 total.
- The model still leans on `vr_last` and `kc_last` (gains 0.37 / 0.26). The best new feature, `qp_arrive_margin`,
  takes 1.4% of the gain. The clearance features are almost constant (the visible tail is short, and its
  movement is noisy under the Task 1 masks).
- Conclusion: the model already reads the queue dynamics from the latest speed and density. Not adopted.
  Together with note 34, the cheap T2 levers are exhausted. What remains is a second model family (T2 CNN,
  `analysis/t2_cnn.py`, waiting for the 4× RTX 6000 Ada machine).
