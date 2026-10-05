# 42 — Research round 2: top-team techniques not yet used, and what the competition's own pages say

Date: 2026-09-15 · Web research only (no experiments). Sources: the competition's Kaggle discussion and leaderboard, and
winning write-ups of structurally similar competitions:
- Traffic4cast 2020, 2021 and 2022: traffic state and congestion classes from detector data, under temporal shift.
- Weather4cast 2022 and 2023: rain/no-rain maps scored by IoU/CSI, under spatio-temporal shift.
- CMI sleep onset, Contrails, and the segmentation "tricks" literature.

## The competition itself

- The discussion has 3 topics, all about the Task 2 leak and the rebuild. There are no public notebooks.
- Leak history: Task 2 horizon cells were published in `mainline_states_masked`.
  - Reading them scored S_queue ≈ 0.9 on train and 0.641 on validation (organizer).
  - A second route inside `window_history.parquet` was reported after the first fix.
  - Both are closed by the 9/11 rebuild: new seeds for the val/private run; train unchanged.
- The reporter of the leak wrote that his genuine forecasting method gets S_queue ≈ 0.5 on train (pre-rebuild).
- The leaderboard mixes pre-rebuild and post-rebuild scores. Only post-rebuild ones count.
  - Post-rebuild: 0.92406 (2026-09-12), 0.92226 (2026-09-15), then **ours 0.87505 (3rd among post-rebuild entries)**.
  - Entries dated 9/9–9/10 (0.917, 0.910, 0.894, 0.878) are pre-rebuild and not comparable.

## Techniques used by winners that we have not used

| # | Technique | Where it won | Our status | Where it would go |
|---|---|---|---|---|
| 1 | **Learned location identity** (per-node/per-link embedding) | STID ([arXiv:2208.05233](https://arxiv.org/abs/2208.05233)) beats STGNNs with identity embeddings; T4c 2020 geo-embedding; T4c 2022 1st (ustc-gobbler) node embeddings | The CNN sees only static channels (lanes, capacity, vf, ramps) on 64-link crops, with no identity. LightGBM has pos/panel, and dropping it hurts (note 22) | CNN: add a link embedding (panel × link) or full-panel inputs |
| 2 | **Multi-task auxiliary targets** | T4c 2022 GongLab (congestion class + speed + volume); almost all top Ventilator teams | The CNN predicts only the 6 queue logits | CNN: add future v/vf regression heads for T+5…T+30 (train truth) |
| 3 | **Mixup** | W4C 2022 KAIST: val CSI 0.330 → 0.370 ([arXiv:2212.02059](https://arxiv.org/pdf/2212.02059)) | none | CNN training |
| 4 | **Self-distillation / EMA / SWA of weights** | W4C 2022 KAIST; standard in segmentation winners | Best early-stopping state only | CNN training (cheap) |
| 5 | **Global network-state features** (PCA of the whole current state) | T4c 2022 2nd (Bolt): PCA alone 0.8978 → 0.8877 weighted CE; "global features dominate" ([arXiv:2211.00157](https://arxiv.org/abs/2211.00157)) | LightGBM is per-link plus tod/panel/stress; no panel-wide state summary | LightGBM, local CPU |
| 6 | **Per-region thresholds** | W4C 2022 2nd (meteoai); KAIST uses 0.1/0.2/0.4 by region | One global τ 0.60 for ongoing. Note 34 tested only global τ | Cached probs: per-family or per-horizon τ, chosen on val and checked on private |
| 7 | **Calibrate before the Bayes decision** | Label-shift literature: bias-corrected calibration is hard to beat ([arXiv:1901.06852](https://arxiv.org/abs/1901.06852)) | Lovász de-calibrates the CNN (note 36), yet the ratio rule assumes calibration | Platt/isotonic fit on train, then the onset blend and decision |
| 8 | Spatio-temporal smooth loss + TTA ensemble | W4C 2022 SIANet: +0.8 and +0.9 mIoU ([arXiv:2212.02968](https://ar5iv.labs.arxiv.org/html/2212.02968)) | Post-hoc smoothing hurt (note 34). Geometric TTA is invalid because direction matters | low priority |
| 9 | Third model family (GNN / transformer) | Every winner ensembles families | LGBM + CNN; a second CNN seed family adds nothing (note 38) | Ada |

## Excluded (already known to fail here, or outside the user's rule)

- Test-time domain adaptation on the scored months (pseudo-labelling, TENT, EM label-shift):
  - T4c 2021 3rd reports that pseudo-labelling hurt under temporal shift.
  - It would also train T2 on scored months, which the user's rule forbids (evaluation yes, training no).
- Similarity/kNN analog features and target encodings (TSE, oahciy, Bolt): these amount to train priors. Those
  are run-specific and hurt here (notes 17, 19).
- The T4c 2021 mean-map ratio transform: needs per-pixel stationarity; our runs shift where queues form, not a
  global scale.

## Takeaway

- Items 1–4 are the CNN upgrades that top solutions stack. Our CNN has none of them, and its onset is the weak
  part: it takes only 30% of the onset blend.
- Item 5 is the one feature family that T4c winners report as dominant and that we lack.
- Items 6–7 are cheap checks on cached probabilities.
