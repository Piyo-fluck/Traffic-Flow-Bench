# 95. Structured link identity for the onset STID (ST-MLP's A.E beats free ID; Laplacian PE does not)

Date 2026-09-23. Code `occ_lab/graph_ident.py`, `occ_lab/train_onset_stid.py --ident`,
`occ_lab/onset_four_sweep.py`. The user's literature reading after note 93: rather than stabilising the STID,
enrich the identity it carries - ST-MLP (arXiv 2308.07496) adds a graph-smoothed node codebook A.B beside the
free embedding, STEMLP (Inf. Sci. 2025) replaces the free embedding with normalized-Laplacian eigenvectors.

## What "graph" means here

Our Task 2 panels are single corridors: the links form a chain ordered by milepost (ramps are separate data and
already enter as static input channels). On a path graph the normalized-Laplacian eigenvectors reduce to a
cosine basis, so **LapPE here is a positional encoding, not network structure**; A.E is the average of the
neighbours' identity embeddings. Both are still new to the model, whose identity was a free per-link embedding
with no notion of position or neighbourhood.

Everything else is held at candidate AN's recipe: 15 x 12 input, 96-d projection, 3 residual MLP blocks, T+30
single output, BCE + 0.5 Lovasz, 8k steps, 4 seeds. Eigenvector signs are randomly flipped during training.

## Result (onset family IoU against production 0.7862 / 0.8227)

| identity | mixtures beating production on both months | best by the worse month |
|---|---|---|
| free ID (AN) | 86 / 195 | H .30 / ad9b .35 / STID .35 -> +0.0298 / +0.0337 |
| **+ neighbourhood A.E (ST-MLP)** | 75 / 195 | **H .20 / ad9b .40 / STID .40 -> +0.0325 / +0.0373** |
| + Laplacian PE (STEMLP) | 48 / 195 | H .45 / ad9b .40 / STID .15 -> +0.0233 / +0.0222 |

- **The neighbourhood identity wins on both months** and keeps a plateau (ad9b 0.40, STID 0.40-0.50, H 0.10-0.20
  all sit at about +0.031 / +0.037).
- **LapPE is worse than the plain free embedding.** On a chain it adds a smooth positional basis and nothing
  else, and it costs capacity the free embedding was using.
- A four-member blend (H / ad9b / free-ID STID / neighbourhood STID) over 1,771 mixtures peaks exactly at
  **ID weight 0** - the two identity variants are not complementary; the neighbourhood one subsumes the free one.

## Candidate AO (built, not submitted)

`work/sub_AO_onsetnbr/submission.csv` = AN with the onset STID replaced by the neighbourhood-identity version at
H 0.20 / ad9b 0.40 / STID 0.40. **14 queue rows differ from AN.** Local estimate +0.0095 / +0.0036 onset family
over AN, i.e. about +0.001 total before the realisation factor.

## LB: AO = 0.87693 (submission 56496719), -0.00460 against AN

AO lost heavily - below AN (0.88153) and below AI (0.87902), on **14 queue rows**. Locally it was ahead of AN on
both months (+0.0325 / +0.0373 against +0.0230 / +0.0337).

Two things follow:

- **The local onset evaluation cannot rank these variants.** It has 169 / 162 pseudo onset windows per month
  against the official 40, and the decision is a top-k set rule, so a handful of windows moves the family mean.
  Every onset mixture inside the "plateau" should be treated as unranked, not as safe.
- AO changed two things at once (the identity variant and the weights), so the loss is not attributed. AN's
  point - free identity at 0.35 / 0.35 / 0.30 - is the only onset configuration with an LB result above AI.

AN stays the best submission.
