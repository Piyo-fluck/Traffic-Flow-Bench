# 127 - D2STGNN as a heterogeneous ongoing family: trains well, adds nothing to AU

Chosen after a literature search with explicit criteria (a different inductive bias; strongest in independent
benchmarks - LibCity 2023 rank 1, BasicTS+ TKDE 2024 best MAE on 5 of 6 traffic sets; small and gradient-
adaptable). Re-implemented from the VLDB 2022 paper (`analysis/t2_d2stgnn.py`): estimation-gate decoupling with
backcast residuals, localized spatio-temporal diffusion over forward / backward / self-adaptive transitions
modulated by a dynamic attention graph, GRU + self-attention inherent block. Deviations for Task 2: queue logits
with BCE + 0.5 Lovasz, linear (not autoregressive) forecast branch, time embeddings from the input's time
channels, global link-id node embeddings, Gaussian milepost adjacency (sigma 3 km, cut 0.1). 0.27M parameters.

Train-run ES bce 0.0203-0.0213 (4 seeds; the SSL CNN ~0.022). Official-like ongoing pool, via
`occ_lab/adapt_family.py --family d2` (frozen / adapted to the scored run): seeds 0.759-0.777 / 0.819-0.825
(validation), 0.735-0.748 / 0.793-0.797 (private); 4-seed mean 0.7724 frozen, 0.8257 adapted (adapted tfm8 mean
0.8462).

In AU's blend (`occ_lab/d2_stack_eval.py`, AU corrector): adapted 0.10 of p0 +0.0001 (early -0.0009), in the
tfm slot -0.0004, 0.20 of p0 -0.0017; frozen variants -0.0022..-0.0089. Not submitted.

Reading: once the members are adapted to the scored run, the architecture barely matters - the same outcome as
adapting a third to fifth family (notes 125-126). The information the adaptation supplies is shared by all of them.
