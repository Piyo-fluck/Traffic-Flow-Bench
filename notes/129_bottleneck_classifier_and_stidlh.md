# 129 - Onset bottleneck classifier (works on the lab surface, candidate AW) and a link x horizon STID (negative)

## Onset: candidate bottlenecks competing, with run-invariant features (`occ_lab/bottleneck_cls.py`)

Candidates = the top 5 peaks of the production onset probability at T+30 (>= 4 links apart), each link assigned to
its nearest peak. Features without any link id: production evidence (peak, log, rank, peak / max, region mass, the
four members at the peak), current state at the peak (last / min v/vf, q/cap last and trend), structure (lanes, lane
change, ramps), same-run evidence (queued share of the visible 06-12 h cells at the peak on all earlier days and on
the last earlier day), time of day. LightGBM on candidate rows, normalised per window, fitted on one scored month and
applied to the other.

| | validation | private |
|---|---|---|
| production top-1 candidate correct | 0.769 | 0.840 |
| classifier top-1 candidate correct | **0.817** | **0.877** |
| truth in production top-3 | 0.994 | 0.975 |

Turning the choice into a decision matters: reweighting regions by P_cls / P_base (-0.05) or keeping only the chosen
region (-0.10) destroy the extent the production blend already gets right. **Swap** - only where the classifier's top
candidate differs from production's, exchange the two peaks' heights by rescaling their regions - is clean:

| onset family IoU | validation | private | swapped / rescued / spoiled |
|---|---|---|---|
| production | 0.8160 | 0.8623 | |
| **swap, margin 0** | **0.8292 (+0.0132)** | **0.8747 (+0.0124)** | 31 / 4 / 0 |
| swap, margin 0.3 | 0.8327 (+0.0166) | 0.8690 (+0.0067) | 24 / 3 / 0 |

Candidate AW: `occ_lab/bottleneck_fit.py` (both months) -> `occ_lab/models/bottleneck_lgb.txt`;
`scripts/t2_onset_blend_predict.py --bn-model` (`work/run_AW_onset.sh`); 8 of the 80 official onset windows swapped,
3 change their decision (13 cells) vs AP/AU; patched into AU -> `work/sub_AW/submission.csv`. Not yet submitted.

## Ongoing: STID with a low-rank (4) link x horizon term, adapted whole or only in that term

`occ_lab/train_stid.py --lh-rank 4`, 4 seeds (ES bce 0.043-0.044). Official-like ongoing pool, `adapt_family.py
--family stidlh`: frozen 0.669-0.674 / 0.649-0.656; adapted whole 0.724-0.728 / 0.696-0.701; adapted in the link x
horizon term only 0.677-0.680 / 0.647-0.654 (the term alone carries no run information). In AU's p0 at 0.05-0.15:
-0.0009 .. -0.0044 in every variant. Negative.

## LB

AW public **0.88635** = AU 0.88635 (identical to 5 decimals). The 3 changed onset windows (13 cells) either fall in the
private part of the leaderboard split or move the score by < 5e-6; the public LB cannot tell.
