# 66 - T2: the Task 1 trunk is a WORSE starting point than random init (idea 1, rejected)

Date: 2026-09-16 · Remote logs `work/t2_cnn_s20_base.log` and `work/t2_cnn_s20_t1i.log` on the Ada machine
· Patch: `T2CNN_T1INIT` added to `analysis/t2_cnn.py` (opt-in, backup kept)

Idea 1 of the user: Task 1 has far more supervision than Task 2, so pretrain its encoder on masked reconstruction
and transfer the trunk. The verification they asked for: same architecture, same seed, only the initialisation
differs. The Task 1 Block and the Task 2 Block have identical parameter shapes (GroupNorm(8,64), Conv2d(64,64,3)
twice; only the dilation values differ), so the first 8 Task 1 blocks were copied by name into T2Net.blocks -
**64 tensors** - leaving the input and output layers random.

Seed 20, Lovasz 0.5, 30k steps, one GPU each.

| CNN alone | onset val d1-14 / all | onset priv d1-14 / all | ongoing val d1-14 / all | ongoing priv d1-14 / all |
|---|---|---|---|---|
| random init | **0.684 / 0.654** | **0.681 / 0.623** | 0.780 / 0.782 | 0.772 / 0.761 |
| Task 1 pretrained | 0.593 / 0.588 | 0.557 / 0.528 | 0.770 / 0.772 | 0.778 / 0.756 |

Blended with the LightGBM: ongoing val all 0.795 (random) vs 0.789 (transfer), priv all 0.777 vs 0.776.

**Verdict: rejected.** The transfer costs 0.066-0.095 onset IoU and is neutral to slightly negative for ongoing.
The early training curve already showed it (ES bce 0.0868 vs 0.0922 at step 500, and 0.0251 vs 0.0254 at the end).

Why, most likely: the Task 1 trunk is optimised to **reconstruct the present** - it learns to smooth and copy the
observed cells - while Task 2 has to **extrapolate 30 minutes forward**. Initialising from it imports the
reconstruction bias, and onset, which is the extrapolation-heavy half, suffers most.

This does not condemn idea 5 (history/future contrastive pretraining): that objective is trained on Task 2 own
windows and asks the encoder to identify which future belongs to this history, which is aligned with the target
task rather than opposed to it.
