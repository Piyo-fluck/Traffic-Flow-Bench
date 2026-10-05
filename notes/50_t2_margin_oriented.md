# 50 — T2 onset: capacity-margin / structure features rebuilt with the correct link orientation → orientation is not the problem

Date: 2026-09-16 · Script: `analysis/t2_margin_oriented.py` · Log: `work/t2_margin_oriented.log` (581 s, local)

**Question (from note 49).** The margin (`mg_*`) and structure (`st_*`) features and the effective-capacity estimate
(notes 21, 22, 27) treated the higher index as downstream in every panel, so they looked upstream in S/W panels. Does
re-orienting them make them work for onset?

**Setup.**
- Three orientations:
  - **old**: +1 everywhere (the old code).
  - **rule**: N/E +1, S/W −1.
  - **measured**: the note-49 signs; unlike the rule, D7_I10_E is −1 and D7_I10_W and D12_I5_S are +1.
- Queue heads are now "link free AND its upstream neighbour queued".
  - The effective capacity changes on 11/107, 17/81, 23/109 and 22/100 links of the four S/W panels.
  - It is unchanged on N/E panels.
- Features: `st_*` (11) and `mg_*` (6) added to H's onset features.
- Paired H-config LightGBM (prior dropout, 5 bags, 600 rounds, 63 leaves), 3 seeds. All variants share the
  prior-dropout draws. The margin + structure share of gain is 0.18–0.23.
- Scored on the blank-aware official-like onset windows (ratio rule). Δ is against the paired base, mean of 3 seeds
  (seeds > 0).

| onset | val d1-14 | val all | priv d1-14 | priv all | mean Δ |
|---|---|---|---|---|---|
| **alone, masked**: base | 0.787 | 0.753 | 0.810 | 0.790 | |
| old | +0.0055 (3/3) | +0.0155 (3/3) | +0.0129 (3/3) | +0.0017 (1/3) | +0.0089 |
| rule | +0.0125 (3/3) | +0.0186 (3/3) | +0.0100 (3/3) | −0.0024 (1/3) | +0.0097 |
| measured | +0.0038 (2/3) | +0.0268 (3/3) | +0.0005 (1/3) | +0.0070 (2/3) | +0.0095 |
| **alone, T1-filled**: base | 0.795 | 0.766 | 0.823 | 0.798 | |
| old | +0.0058 (3/3) | +0.0182 (3/3) | −0.0177 (0/3) | +0.0081 (2/3) | +0.0036 |
| rule | +0.0022 (2/3) | +0.0122 (3/3) | −0.0065 (1/3) | −0.0030 (1/3) | +0.0012 |
| measured | +0.0054 (3/3) | +0.0230 (3/3) | −0.0124 (0/3) | +0.0121 (3/3) | +0.0070 |
| **M blend (0.7 LGBM + 0.3 CNN×3), masked**: base | 0.797 | 0.775 | 0.839 | 0.811 | |
| old | +0.0034 (2/3) | +0.0052 (2/3) | −0.0023 (1/3) | +0.0080 (3/3) | +0.0036 |
| rule | −0.0040 (2/3) | +0.0018 (2/3) | −0.0060 (1/3) | −0.0067 (0/3) | −0.0037 |
| measured | −0.0114 (0/3) | +0.0124 (3/3) | −0.0043 (1/3) | +0.0103 (2/3) | +0.0018 |

**Findings.**
- **The correct orientation does not help.** The rule orientation is the worst variant in M's blend (−0.0037).
  The measured orientation (+0.0018) does not beat old (+0.0036).
- **Orientation effects are below the noise floor.** On panels where the features are IDENTICAL across variants,
  the per-panel scores still differ by 0.01–0.05, because the model is trained on all panels pooled. Examples
  (alone, masked):
  - D7_I10_W, old = measured: 0.664 vs 0.713.
  - D7_I210_E, all three variants the same: 0.950 / 0.938 / 0.927.
  - The per-panel orientation "wins" (e.g. D7_I210_W: rule 0.889 vs old 0.863) are within this spread.
- **The margin family gives LightGBM alone about +0.01 onset on masked histories, in every orientation.**
  - The alone gain is smaller on filled histories (+0.001 to +0.007), and priv d1-14 is negative for all three.
  - Most of it disappears in M's blend: the CNN already carries it. The best blend gain is +0.0036 onset (old,
    1 of 4 conditions negative), ≈ +0.0005 S_total. That is not robust and not adopted.
- **The failure of notes 22/27 is therefore NOT an orientation artefact.** Note 21's capacity transfer stands.
- Still open from note 49 (not tested here): `pqc_dn` orientation, and re-orienting the CNN/TFM input axis.
