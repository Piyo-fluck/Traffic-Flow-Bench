# 34 — Task 2: contiguity-aware onset decisions and ongoing post-processing do not help

Date: 2026-09-15 · Script: `analysis/t2_decision_variants.py` (cached probs in `work/cache/t2probs_*`)
· Log: task output (see script)

These tests use the saved models (onset = H's 5 bags, ongoing = I) on the blank-aware official-like windows.
The baselines are K's rules: onset uses the expected-IoU ratio (c 1.0), ongoing uses p ≥ 0.65.

| Δ IoU vs baseline (panels better/worse) | val d1–14 | val all | priv d1–14 | priv all |
|---|---|---|---|---|
| onset: best contiguous block | −0.047 (2/4) | −0.067 (3/4) | −0.065 (0/4) | −0.060 (1/5) |
| onset: smooth p with neighbours, λ 0.25 | −0.007 (1/2) | +0.003 (3/3) | +0.008 (2/0) | +0.003 (3/2) |
| onset: smooth λ 0.5 | −0.026 | −0.026 | −0.011 | −0.029 |
| ongoing: τ 0.60 | +0.001 (5/3) | +0.001 (4/4) | +0.008 (5/3) | +0.002 (6/2) |
| ongoing: spatial smoothing w 0.25 | −0.009 | −0.006 | +0.003 | −0.003 |
| ongoing: 3×3 space-time smoothing | −0.105 | −0.100 | −0.104 | −0.090 |
| ongoing: close 1-link gaps | −0.022 | −0.023 | −0.014 | −0.021 |
| ongoing: drop isolated cells | −0.000 | −0.003 | −0.001 | −0.002 |

- The true onset cells are often not a single contiguous block, or the probability mass splits across blocks.
  Forcing contiguity loses 0.05–0.07.
- Every smoothing or morphology step on ongoing hurts. The model's cell-wise probabilities already hold the
  spatial structure it can use.
- τ 0.60 is ≥ τ 0.65 in all four conditions, but only by +0.001…+0.008 ongoing, about +0.0005 total.
- **Conclusion: decision-level tweaks are exhausted. K's rules stay.** Further T2 gains must come from better
  probabilities (features or models), not from the decision.
