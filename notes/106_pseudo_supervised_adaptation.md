# 106. Causal pseudo-supervised adaptation for Task 2 (candidate AP, LB 0.88311)

## The lever

Production adaptation (idea 9b, note 67) fine-tunes on the scored month with a **masked reconstruction** loss and
never touches a Task 2 label. But the organizers' clarification (2026-09-20) allows every released observation
timestamped at or before a window's origin T, including earlier hours of the same day. From that we can build
**real task windows**: an origin tau with tau+30 min <= T, its history from the released layer, and its labels
from the released layer at tau+5 .. tau+30.

Those labels are not a proxy. `obs == true` to the last bit wherever a cell is visible (note 103), so at visible
future cells the pseudo label IS the label; invisible cells are simply masked out of the loss. No leak route is
touched: nothing after T is read (note 79's boundary).

## Local result (`occ_lab/pseudo_adapt.py`, single CNN, its own IoU)

| adaptation | validation onset | validation ongoing | private onset | private ongoing |
|---|---|---|---|---|
| none        | 0.6669 | 0.7939 | 0.6419 | 0.7777 |
| rec (= 9b)  | 0.7345 | 0.7990 | 0.6794 | 0.7784 |
| sup         | 0.7227 | 0.8374 | 0.7424 | 0.8217 |
| **sup+rec** | **0.7375** | **0.8394** | **0.7457** | **0.8192** |

The supervised half is where the ongoing gain lives (+0.038 / +0.043 over 9b); reconstruction still adds a little
on onset. This is the first adaptation variant since note 67 that improves **both months and both halves**.

## In the production blends (`occ_lab/pseudo_blend_eval.py`)

- ongoing: `base + new 0.30` (base = 0.2 gs1 + 0.6 tfm8 + 0.2 cnn3) gained about +0.0125 on both months - a
  plateau interior, not an edge, after AO's failure at a plateau edge (note 93).
- onset: replacing half the ad9b slot (0.35 H + 0.175 ad9b + 0.175 new + 0.30 STID) gained about +0.006 on both.
- the AI residual corrector still helps on top of the new blend: +0.0080 / +0.0100 against AI
  (`occ_lab/pseudo_corrector_check.py`), so it is kept unchanged.

## Candidate AP

`scripts/t2_pseudo_adapt_official.py` runs the same recipe on the official windows (30 steps, lr 1e-4, bs 16,
w_rec 1.0, same-day windows drawn 35% of the time with the cut at the day's first official origin), 2 minutes for
8 panels x 2 splits. Then `t2_residual_predict.py --pseudo-dir ... --w-pseudo 0.30` (the new member enters p0
*before* the corrector's features) and `t2_onset_blend_predict.py --ad2-dir ... --w-ad2 0.175`.

- **260 queue rows differ from AN**: ongoing 258 (125 are 0->1, 133 are 1->0), **onset only 2** (both 0->1).
  The expected-IoU ratio rule is a top-k set rule, so a 0.175 reweighting of one member almost never flips it.
- **LB 0.88311 against AN's 0.88153: +0.00158.** Five times AI's corrector gain (+0.00031) and the largest
  single-candidate step since candidate T. Practically all of it comes from 258 ongoing cells.

## What this says

- The ongoing local evaluation finally pointed the right way: local +0.0125 family -> +0.00158 total, a
  realisation of about 0.8 against the 0.44-0.6 of earlier candidates (note 89). Windows built from the scored
  run itself evidently match the official ones far better than our pseudo windows do (note 76's coverage gap).
- The remaining onset lever is not the mixture weight but the decision rule: at these probabilities the ratio
  rule is nearly invariant to member reweighting.
