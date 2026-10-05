# 29 — Task 2 ongoing: stress features confirmed; ineligible history cells add a hair; tau 0.65 stays

Date: 2026-09-15 · Script: `analysis/t2_ongoing_variants.py` · Log: `work/t2_ongoing_variants.log`

Four no-prior ongoing models were trained on identical sampled rows:
- history: eligible only (el) vs every valued cell (fin)
- demand-stress features: off vs on

Each was scored with τ ∈ {0.55, 0.65, 0.75} on official-like ongoing windows. The ongoing windows barely overlap
official blanks (note 28). The reference is el/stress/0.65, J's ongoing.

| ongoing IoU, family mean | val d1–7 | val d1–14 | priv d1–7 | priv d1–14 |
|---|---|---|---|---|
| el / stress / 0.65 (J) | 0.772 | 0.771 | 0.786 | 0.789 |
| **fin / stress / 0.65** | 0.773 (+0.001) | 0.774 (+0.003) | 0.787 (+0.002) | 0.791 (+0.003) |
| el / no stress / 0.65 (H) | 0.759 | 0.762 | 0.772 | 0.773 |

- **Stress features help ongoing in every condition**: +0.009 … +0.016 at τ 0.65, as in notes 22 and 26.
- Using ineligible valued history cells (fin) is ≥ the reference in all 4 conditions (5/3 panels each). The gain
  is tiny: about +0.0025 ongoing, i.e. +0.0004 total. It is free to adopt, and the official history carries these
  cells.
- τ 0.65 is the most stable. 0.55 and 0.75 each win one condition and lose others.
- Ongoing is close to saturated for this model family. The remaining T2 headroom is in onset.
