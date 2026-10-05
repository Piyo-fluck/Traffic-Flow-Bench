# 67 - T2: contrastive pretraining (idea 5) also hurts; the representation ideas keep failing the same way

Date: 2026-09-16 · Remote logs `work/t2_idea5_con.log` (and `work/t2_idea4_*.log`) on the Ada machine
· Patch: `T2CNN_CONTRAST`, `T2CNN_FUTURE`, `T2CNN_TEACHER`, `T2CNN_DISTILL`, `T2CNN_REC`, `T2CNN_PL` added to the
remote `analysis/t2_cnn.py` as opt-in flags, backups kept.

## Idea 5: history/future contrastive pretraining

3000 InfoNCE steps between a history window and its own future window (the future is repeated to 12 slots so the
collapse layer sees the same shape), against the other windows of the batch, then the normal 30k-step training.
Control: the same seed 20 random-init run of note 66.

| CNN alone | onset val d1-14 / all | onset priv d1-14 / all | ongoing val d1-14 / all | ongoing priv d1-14 / all |
|---|---|---|---|---|
| random init (control) | 0.684 / **0.654** | 0.681 / **0.623** | 0.780 / **0.782** | 0.772 / **0.761** |
| contrastive pretraining | 0.637 / 0.606 | 0.640 / 0.572 | 0.770 / 0.767 | 0.785 / 0.750 |

Blended with the LightGBM: ongoing val all 0.789 against 0.795, private all 0.773 against 0.777.

**Rejected.** Same shape of failure as idea 1 (note 66): onset loses 0.05, ongoing loses a little, and the blend
follows. Two different pretraining objectives - Task 1 reconstruction and history/future contrast - both leave the
encoder worse for this task than random initialisation.

## Idea 4: the teacher run produced NaN and saved nothing

The privileged teacher (T2CNN_FUTURE=1) trained with `train loss nan` from step 500 onward, so the early-stopping
score never improved and no checkpoint was written; the student then failed on the missing file. Cause: the horizon
channels were built as `true_flow / cap` without masking, while `make_x` zeroes the history channels at ineligible
cells - so every window with a missing future cell fed NaN into the trunk. Fixed by masking with `ctx.el` and
filling with zero; the teacher is being retrained.

## Ideas 9 and 10: an auxiliary head that keeps running beats every pretraining

Both were trained at full length with the same seed 20 and the same 30k steps as the control.

    idea 9   T2CNN_REC=0.3   a self-supervised reconstruction head on the trunk: 20% of the observed history cells
                             are hidden from the input and the head predicts their v/vf and q/cap. No Task 2 label
                             is involved, which is what the user asked for (note 59 showed that past-day QUEUE
                             LOCATIONS reinforce the habitual-bottleneck bias, so only the dynamics are used).
    idea 10  + T2CNN_PL=0.3  additionally regress the Task-4 style path load of each link (the closed-form solve
                             of note 60 as a per-window target).

| CNN alone | onset val all | onset priv all | ongoing val all | ongoing priv all |
|---|---|---|---|---|
| random init (control) | 0.654 | 0.623 | 0.782 | 0.761 |
| **idea 9, reconstruction head** | **0.667** | **0.642** | **0.791** | **0.773** |
| idea 10, + path-load head | 0.625 | 0.606 | 0.774 | 0.758 |

Blended with the LightGBM, ongoing: control 0.795 / 0.777, **idea 9 0.800 / 0.786**, idea 10 0.789 / 0.777.

- **Idea 9 is the first representation change that improves anything**: +0.013 / +0.019 onset and +0.009 / +0.012
  ongoing for the CNN alone, and +0.005 / +0.009 in the blend. Roughly +0.0008 / +0.0014 total on the ongoing side.
- **Idea 10 is negative**: adding the path-load regression on top of idea 9 costs 0.04 onset and 0.017 ongoing.
  The latent-demand target pulls the trunk away from what the queue head needs, which fits note 60, where path load
  had a 0.45 top-1 ceiling as a bottleneck ranker.

## The pattern across ideas 1, 4, 5, 9, 10

Changing the **initialisation** (idea 1 Task 1 transfer, idea 5 contrastive pretraining) makes the model worse.
Adding a **signal that keeps running during training** (idea 9 reconstruction, and ideas 2/3 as features) makes it
better. The T2 window is small (12 slots x links) and the model already has enough capacity for 30k steps on a few
thousand windows, so there is little for a better starting point to buy - while an auxiliary objective keeps
regularising the trunk for the whole run.

## Ideas 4 and 9b: the two that work, and they are the two that keep learning at the right time

**Idea 4, distillation from a privileged teacher.** The teacher (T2CNN_FUTURE=1) sees the horizon FLOW as extra
channels and reaches ES bce 0.0093; its probabilities on every training window are then the soft target of a normal
student (T2CNN_DISTILL=1.0). Same seed 20, same 30k steps as the control.

| | onset val all | onset priv all | ongoing val all | ongoing priv all |
|---|---|---|---|---|
| control, CNN alone | 0.654 | 0.623 | 0.782 | 0.761 |
| student, CNN alone | 0.660 | **0.637** | 0.779 | 0.761 |
| control, blended with the LightGBM | 0.751 | 0.739 | 0.795 | 0.777 |
| **student, blended** | **0.767** | **0.792** | 0.795 | 0.780 |

**Idea 9b, causal adaptation on past days** (`work/t2_idea9_adapt.py` on Ada). For each scored day in ascending
order the trunk is adapted on windows drawn from days strictly before it, with the reconstruction loss only - no
Task 2 label ever enters - 30 steps at lr 1e-4, weights carried forward day by day.

| | base | adapted | delta |
|---|---|---|---|
| onset, CNN alone (331 windows) | 0.7314 | **0.7799** | **+0.0485 [+0.0258, +0.0722]** |
| ongoing, CNN alone (6725 windows) | 0.8045 | 0.8089 | +0.0043 [+0.0020, +0.0066] |

**This is the largest onset improvement measured in the whole project**, and it comes from the one form of run
adaptation that ignores where the queues were. Note 59 rejected the version that uses past-day queue LOCATIONS
(it reproduces the habitual-bottleneck error); adapting to the run DYNAMICS instead is a different object, and the
user was right to insist on the distinction.

## Summary of the representation ideas

| idea | what changes | verdict |
|---|---|---|
| 1 Task 1 trunk transfer | the initialisation | rejected, onset −0.07…−0.10 |
| 5 contrastive pretraining | the initialisation | rejected, onset −0.05 |
| 10 Task-4 path-load head | an auxiliary objective pulling elsewhere | rejected, onset −0.04 vs idea 9 |
| 9a reconstruction head during training | an auxiliary objective on the same data | CNN better everywhere, **no ensemble gain** |
| **4 privileged-teacher distillation** | a soft target that runs all through training | **onset blend +0.016 val / +0.053 priv** |
| **9b past-day causal adaptation** | the weights track the run at scoring time | **onset +0.0485 CNN alone** |

### Idea 4 does not survive the production blend

The internal evaluation inside `t2_cnn.py` blends the LightGBM and the CNN with equal weight; M production onset
rule is 0.7 LightGBM + 0.3 CNN x3 followed by the expected-IoU ratio rule. Re-tested there
(`analysis/t2_onset_distill_blend.py`), on the blank-aware official-like onset windows:

| onset | validation | private |
|---|---|---|
| M production (CNN x3) | 0.8101 (family 0.7797) | 0.8327 (family 0.8181) |
| distilled CNN instead | +0.0015 [-0.0089, +0.0139] | +0.0005 [-0.0118, +0.0133] |
| CNN x3 and distilled together | +0.0022 [-0.0043, +0.0112] | -0.0028 [-0.0123, +0.0062] |

Nothing, with every CI crossing zero, and the private family mean actually falls. **At a CNN weight of 0.3 a better
CNN is diluted**; the large numbers in the training log came from the internal equal-weight blend. Idea 4 is real
but not bankable in the current recipe.

### Idea 9b in the production recipes

The adapted probabilities were saved in the cache format and fed to the two production rules.

**Onset (0.7 LightGBM + 0.3 CNN, ratio rule), blank-aware official-like windows:**

| | validation | private |
|---|---|---|
| M production (CNN x3) | 0.8101 (family 0.7797) | 0.8327 (family 0.8181) |
| **adapted CNN instead of CNN x3** | **0.8181 (family 0.7862), +0.0079 [-0.0004, +0.0198]** | **0.8516 (family 0.8227), +0.0189 [+0.0024, +0.0387]** |
| CNN x3 and adapted together | +0.0053 | +0.0070 |

Family-mean gain +0.0065 (val) and +0.0046 (priv), i.e. about **+0.001 total**, and the private CI excludes zero -
the first onset-side change in this project that manages that inside the production blend.

**Ongoing:** adding it to the note-56 ensemble is neutral to negative (family 0.8328 vs 0.8357 on validation,
0.8225 vs 0.8233 on private), like every other ongoing addition since note 54.
