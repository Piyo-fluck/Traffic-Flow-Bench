# 27 — Task 2 onset: no model tweak beats H's onset on official-like windows

Date: 2026-09-15 · Script: `analysis/t2_onset_variants.py` · Log: `work/t2_onset_variants.log`

16 onset-model configurations, each with 5 bags. Every configuration uses the same prior-dropout draws. They are
scored on first-morning-onset windows (the official-like onset, note 26), validation and private, days 1–7
and 1–14. The dimensions are:
- history: eligible only (el) vs every valued cell (fin; the official history carries about 7% ineligible valued
  cells)
- columns: base, +struct, +mg, +struct+mg
- training windows: all vs morning (queue at 04–14 h)
- α ∈ {1.0, 1.2, 1.5}

The reference is el/base/all/α 1.2, i.e. H's onset.

| condition | reference | best configuration (Δ) | that configuration elsewhere |
|---|---|---|---|
| val d1–7 | 0.466 | el/base/morning/1.5 (+0.046, 5/0 panels) | val d1–14 −0.039, priv d1–7 −0.016, priv d1–14 −0.006 |
| val d1–14 | 0.652 | el/+struct/all/1.5 (+0.005) | val d1–7 +0.010, priv d1–7 −0.014, priv d1–14 −0.011 |
| priv d1–7 | 0.557 | el/base/morning/1.2 (+0.002) | priv d1–14 +0.015, val d1–7 −0.020 |
| priv d1–14 | 0.645 | el/base/morning/1.2 (+0.015) | see the row above |

- **No configuration improves all four conditions.** The reference is at or near the top in every one of them.
  The differences are at noise level: about 50 first-morning events on days 1–7 over 8 panels, and a per-event
  IoU sd of about 0.4.
- Using ineligible valued history cells does not help onset. Capacity-margin features hurt (−0.04 on d1–14).
  α = 1.2 is the most stable. Morning-only training is inconsistent.
- Next: error analysis of H's onset, location vs count, and a combination with past-day first-onset locations
  (`analysis/t2_onset_error_analysis.py`).
