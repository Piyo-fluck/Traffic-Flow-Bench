# 31 — Task 2: per-window expected-IoU decisions lift onset by +0.05…+0.09 (ongoing keeps τ 0.65)

Date: 2026-09-15 · Script: `analysis/t2_expected_iou.py` · Log: `work/t2_expected_iou.log`
· Production: `scripts/t2_redecide_onset.py`

IoU is a set metric, and its Bayes-optimal prediction is a top-k set whose k maximises the expected IoU given the
window's probabilities (note 30). Production used one rule for every window: onset k = clip(round(1.2 Σp), 1, 6),
ongoing p ≥ 0.65. The comparison used the saved models (onset = H's 5 bags, ongoing = I's stress model) on the
blank-aware official-like windows (note 28). The rules tested:
- **ratio**: k* = argmax_k S_k / (k + P − S_k) on p' = min(1, c·p)
- **mc**: exact E[IoU_k] under independent Bernoulli labels, conditioned on a non-empty truth

| Δ onset vs current (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| **ratio c 1.0** | **+0.055 (4/0)** | **+0.046 (6/1)** | **+0.090 (5/1)** | **+0.072 (7/1)** |
| ratio c 1.2 | +0.056 (4/0) | +0.047 (6/1) | +0.093 (5/1) | +0.071 (7/0) |
| mc c 1.0 | +0.062 (5/0) | +0.045 (6/0) | +0.086 (5/1) | +0.065 (7/1) |
| Δ ongoing for the same rules | −0.002 … −0.041 | | | |

- **Onset improves in every condition and nearly every panel.** The old rule took too many links when
  probability mass was spread thinly. The expected-IoU rule keeps only the links that pay off.
- On ongoing the rule loses. The cells there are strongly correlated in space and time, which breaks the
  independence assumption, so the fixed τ 0.65 stays.
- **Adopted: ratio, c = 1.0.** It is the plain Bayes rule with no scale tuned on the scored months, it is
  deterministic, and it is within noise of the best.
- Onset-only effect on the total: +0.007 … +0.014 vs J, mean ≈ +0.010. **K = F1 + (onset: ratio c 1.0 on H's
  probabilities; ongoing: I)** ≈ 0.863 LB (range about 0.853–0.870).
