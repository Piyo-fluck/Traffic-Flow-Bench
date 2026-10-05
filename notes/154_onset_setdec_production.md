# 154 Onset set decoder in production (candidate BA)

Fit: `occ_lab/onset_setdec_fit.py` -> occ_lab/models/onset_rank_s0-5.txt (IoU-LambdaRank, no new features),
onset_du_s0-5.txt, onset_du_tau.txt (0.02, from 5-fold OOF on both months; OOF gain by threshold 0: +1.81,
0.02: +1.92, 0.05: +1.33, 0.1: +1.18). Seed-averaged cross-month check: stage 1 val 0.8289 / priv 0.8972
(production 0.8160 / 0.8623), stage 2 +0.0027 / +0.0048.

Official: `scripts/t2_onset_blend_predict.py --setdec` (work/run_BA_onset.sh) -> work/sub_t2BA_onset (+ per-window
log setdec_log_<split>.csv); patched into AX -> `work/sub_BA/submission.csv`; window diff `work/sub_BA/diff_windows.csv`.

Per split 40 onset windows: stage-1 top changed in 4 (val) / 4 (priv); stage-2 adds 2 windows (3 cells, val),
0 (priv; max predicted dU -0.0003). BA vs AX: 19 queue cells in 7 windows (4 of them validation); BA vs AXZ (note 145):
3 cells, the two validation stage-2 adds (Q2_D7_I405_S_validation_002 +1 cell, _004 +2 cells). Stage-1 changes that
move no cell: the swap leaves the ratio set unchanged. Submitted 2026-09-27: public 0.88621 (AU/AW 0.88635).

Public attribution (2026-09-27): AX (AU + ongoing per-window selection, note 141) = 0.88559 vs AU 0.88635 ->
the ongoing selection -0.00076; BA (AX + set-decoder onset) = 0.88621 -> the onset set decoder +0.00062 (19 cells,
4 validation windows). Candidate **BB = AU + set-decoder onset** (`work/sub_BB/submission.csv`; vs AU exactly the 19
onset cells, vs BA exactly AX's 88 ongoing cells). Not submitted.

**BB submitted 2026-09-27: public 0.88697 = new best** (+0.00062 over AU, the same as AX -> BA: the onset gain is
additive). Hence BE (BB + note 163's okall ongoing) 0.88617 puts that ongoing change at -0.0008 on public.
