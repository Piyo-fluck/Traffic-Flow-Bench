# 152 Onset set-decoder plan: oracles, then a candidate utility reranker (step 1)

Scripts: oracles in the session scratchpad (oracles.py); `occ_lab/onset_rerank.py`. Lab onset pool, 331 windows,
probabilities = note 145 prior4-swapped for the oracles.

Oracles (mean IoU over windows; 302 single-site, 29 with 2+ sites >= 4 links apart):

| decision | all | single | 2+ sites |
|---|---|---|---|
| ratio rule | 0.870 | 0.875 | 0.824 |
| O2 best subset of the top-5 regions (current shapes) | 0.946 | 0.943 | 0.970 |
| O2 top-10 | 0.947 | 0.943 | 0.981 |
| O3 true regions + best k per region | 0.983 | 0.981 | 1.000 |
| O4 true regions + best interval | 0.989 | 0.991 | 0.977 |

Multi-site recall of all true sites in the top-M candidates: @1 0.14, @2 0.52, @3 0.69, @5 0.86, @10 1.00. The
'2+ regions' loss of note 150 was mostly one queue split by a region boundary; true multi-site is ~12 % of the loss.
Extent is not the problem (O3 0.98); the candidate ranking is (~80 % of O2's gain is in single-site windows).

Step 1, reranker (cross-month, 6 seeds; vs production = no swap; 'single' = single-site windows):

| scorer / decision | all val | all priv | single val | single priv | top changed val/priv | better/worse val, priv |
|---|---|---|---|---|---|---|
| cls (note 145, region label) / swap | +0.0128 | +0.0248 | +0.0154 | +0.0268 | 19.8 / 13.5 | 6.0/1.0, 7.7/0.5 |
| reg (U_b regression) / swap | +0.0146 | +0.0221 | +0.0180 | +0.0239 | 19.0 / 11.7 | 5.3/0.0, 6.8/0.3 |
| **rank (lambdarank on U_b) / swap** | **+0.0146** | **+0.0355** | **+0.0170** | **+0.0403** | 22.7 / 20.0 | 6.5/1.0, 10.2/0.8 |
| rank+ (new evidence) / swap | +0.0146 | +0.0368 | +0.0137 | +0.0419 | 28.0 / 20.3 | 7.5/2.5, 10.2/0.3 |
| any scorer / pick (output the chosen region's shape only) | -0.10..-0.12 | -0.08..-0.10 | | | | ~15/53 |

rank/swap vs the current cls/swap: all +0.0018 / +0.0107, single +0.0016 / +0.0135 - same sign both ways, validation
below +0.005. The teacher (utility ranking) carries it; the new candidate evidence (width, prominence, density slope,
asymmetry) adds nothing on validation. 'pick' loses heavily: a region's own ratio set is a poor shape - the ratio set
over the whole window (R) often spans the boundary-split parts of one queue; the swap keeps it.
