# 155 C: the set decoder with the new candidate evidence in the ranker

`occ_lab/onset_setdec_fit.py --plus` (models onset_rank_plus_s*, onset_du_plus_s*, tau 0.02),
`scripts/t2_onset_blend_predict.py --setdec --setdec-tag _plus` -> work/sub_t2BC_onset.

Seed-averaged cross-month: stage 1 val 0.8312 / priv 0.8972 (BA's 0.8289 / 0.8972), + stage 2 val 0.8339 / priv
0.9003 (BA 0.8315 / 0.9020): vs BA +0.0024 val, -0.0017 priv - opposite signs.
Official: vs BA 2 cells (val: D12_I5_N_003, D7_I405_S_002), 6 cells (priv: D12_I5_S_002/_003, D7_I405_S_001);
stage-1 top changed 6 / 4, stage-2 adds 2 / 3. The official cells move but the cross-month direction is mixed ->
branch closed; BA stays the set-decoder candidate.
