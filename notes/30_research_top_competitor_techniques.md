# 30 — Research: general top-competitor techniques we have not used yet

Date: 2026-09-15. Scope: web search on general techniques only, at the user's request (no Kaggle discussion or
notebook browsing).

| Technique | What top competitors do | Our status | Where it could apply |
|---|---|---|---|
| **Expected-metric (IoU/Jaccard) maximisation** | For set metrics the Bayes-optimal prediction is a top-k set whose k maximises the expected metric given the probability vector, chosen per instance. There are exact algorithms under label independence ([arXiv:1310.4849](https://arxiv.org/pdf/1310.4849)). | One rule for every window: onset k = round(1.2 Σp), ongoing p ≥ 0.65. **Not done.** | Task 2, both conditions. No retraining needed. Tested in `analysis/t2_expected_iou.py` |
| Adversarial validation | A train-vs-test classifier; reweight train samples by similarity to test, or drop dissimilar ones ([arXiv:2112.10078](https://arxiv.org/pdf/2112.10078), [KDnuggets](https://www.kdnuggets.com/2020/02/adversarial-validation-overview.html)) | Run shift found (note 17) and handled with prior dropout; no sample weighting | T2 window weighting toward the scored run's conditions |
| Pseudo-labelling / test-time training | Retrain on test inputs with model or self-supervised labels; common in winning solutions ([PyData keynote](https://github.com/stanleyjzheng/PyData-Pseudolabelling-Keynote)) | T1 has only the month-deviation base (note 18) | **T1 is offline, so the whole month is allowed**: fine-tune the CNN on the scored month's visible cells with self-supervised hiding. T2 is limited by the past-days rule (official windows on days 1–5) |
| Diverse-model ensembles and stacking | GBDT + NN blends, hill-climbing weights, multi-seed averaging ([NVIDIA GM playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/), [arXiv:2009.07701](https://arxiv.org/pdf/2009.07701)) | T2 is LightGBM only (with bags); T1 averages 3 CNNs | A T2 neural model (spatio-temporal CNN on the history) blended with LightGBM |

Priority: expected-IoU decisions first (cheap, directly targets the metric). Then T1 test-time training on the
scored month. Then a second T2 model family.
