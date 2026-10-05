# 108. The onset evaluation problem is run shift, not coverage - and H is the member that does not travel

Date 2026-09-24. Code `occ_lab/cache_onset_oof.py --truth-hist`, `occ_lab/onset_hicov_eval.py`,
`occ_lab/onset_runshift_check.py`.

## The fix that should have worked

Note 76 blamed the local onset evaluation on coverage: our pseudo windows' histories are 0.524 observed against
the official 0.761, because they inherit the Task 1 target mask. So build the windows on the held-out tail of the
train month (2031-01/02, out of fold for every member) with the histories taken from the TRUE layer restricted to
eligibility. Measured coverage **0.764 against the official 0.761** - the mismatch is gone.

Then score the three mixtures whose leaderboard outcome we already know (368 onset windows, 8 panels):

| mixture | this evaluation | actual leaderboard |
|---|---|---|
| T..AM 0.70 H + 0.30 ad9b | 0.9062 | reference |
| **AN 0.35 H + 0.35 ad9b + 0.30 STID** | **0.8616 (-0.0446)** | **+0.00251 over AI** |
| AO 0.20 H + 0.40 ad9b + 0.40 STID(ab) | 0.8165 (-0.0897) | -0.00460 vs AN |

AO's sign is right, **AN's is backwards**, and the level (0.91) is far above anything seen on the official
windows. Fixing the coverage exactly did not fix the ranking, so **coverage was not the primary cause**.

## The mechanism: H is a same-run memory

Each member alone, expected-IoU ratio rule at c = 1.0, family mean over panels:

| member | train tail (same run) | validation | private | drop |
|---|---|---|---|---|
| **H** (LightGBM + recurrence prior) | **0.9240** | 0.7845 | 0.8025 | **-0.1305** |
| **ad9b** (past-day adapted CNN) | 0.7069 | 0.7427 | 0.6787 | **+0.0038** |
| cnn3 | 0.7145 | 0.6894 | 0.6617 | -0.0390 |
| onset STID | 0.7896 | - | - | - |

H's recurrence prior is built from other days of the **same simulation run**, so on the train month it sits at
0.924, near the ceiling, and any diverse member only dilutes it. On a different run it falls to 0.79. **ad9b does
not move at all (+0.004), because it adapts to the run at test time.**

That is the whole story of AN and AO. On the train tail H beats ad9b by 0.217; on the scored months by only
0.083. Every local onset measurement we have ever made was taken on the run where H is inflated, which is why the
production weight 0.70 H / 0.30 ad9b was set too high and why cutting it to 0.35 (AN) gained on the leaderboard.

## What is reusable

- **No train-month evaluation can rank onset mixtures**, at any coverage. Retire that surface for onset.
- A member's **drop from the train tail to the scored months** is a transferable statistic: it needs no mixture
  ranking, and it says which members survive a run change. Adaptive members travel; memorising members do not.
- By that criterion H is still over-weighted at 0.35 and the adaptive slot is under-weighted. Candidate AP put
  the note 106 pseudo-supervised model in only half the ad9b slot (0.175) and it moved **2 rows in 87,000** -
  too timid to test the axis. The clean next step is one change from AN: give the adaptive slot to the
  strictly better adaptive member, 0.35 H + 0.35 pseudo + 0.30 STID.
- AO is not evidence against cutting H: it changed two things at once (H 0.35 -> 0.20 and the STID identity
  id -> ab), and note 93 already suspected the ab identity.
