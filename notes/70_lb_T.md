# 70 - LB: T = 0.87778, a new best, and the first onset-side change that transfers

Date: 2026-09-16 · Submission 56274029 · Candidate: note 69

| submission | content | public LB |
|---|---|---|
| M | L + the T2 CNN blend | 0.87505 |
| R | M with the ongoing ensemble (note 56) | 0.87687 |
| **T** | **R + past-day adapted onset CNN (idea 9b) + 8-seed transformer ongoing** | **0.87778** |

**+0.00091 over R.** The local estimate was +0.0012...+0.0015 family, about +0.0007 after the note-58 halving, so
this one came in at roughly 65% of the raw local estimate - inside the 0.44-0.76 band of note 58, and above the
halved figure.

## Two things worth recording

1. **An onset-side change transferred to the public LB for the first time.** Note 58 established that onset changes
   never showed there (the analog map went negative, the onset hill climb did nothing), which is why the calibration
   rule treats onset gains as unproven. Idea 9b breaks that: the onset half of this candidate is the adapted CNN,
   worth family +0.0065 (validation) locally, and the board moved.
   The likely reason the earlier onset attempts failed and this one did not: they changed the **decision** (a map, a
   gate, a re-ranking) on 169 windows of evaluation evidence, while this changes the **model weights** to match the
   run, which is a property of every window rather than of the few the evaluation happened to sample.
2. **The gain is split between two changes** - ongoing 8 seeds (local family +0.0030 val -> ~+0.00045 total) and the
   adapted onset CNN (local family +0.0065 val -> ~+0.00098 total). Their sum, halved, is what landed.

## Where this leaves the ideas

Of the ten ideas, **9b is the only one that reached the leaderboard**, and it did so through the mechanism the user
insisted on: adapt to the run using the traffic dynamics only, never the queue locations (note 59 had already shown
that using past-day locations reproduces the habitual-bottleneck error).

Still on the table: three-seed adaptation (two more reconstruction-head models are training on Ada), which should
remove the single-seed variance of the current adapted model.
