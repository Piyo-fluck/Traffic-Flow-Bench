# ON HOLD — do not post. Note 14: using these ramp flows as queue flags lowers S_queue, so the "reveals the target" claim below is not supported.

# DRAFT — not posted. For the user to review and post if they choose.

**Title:** Task 2: ramp flows are not blanked over the forecast horizon and reveal the mainline queue state

Hi organizers,

Thank you for the rebuild. While auditing the new release for Task 2 I found
one more route of the same kind as the ones you closed. I have not used it
in any submission.

**What I see.** The files `corridors/<PANEL>/<split>/ramp_states/` are not
blanked over the Task 2 forecast horizon, or over the 60-minute buffer after
it, on validation and private. On the train split, where the queue state
can be checked, the released ramp flows react strongly to the state of the
attached mainline link:

- **On-ramps**: while the attached mainline link is queued (speed ≤ 0.6 ×
  free speed), the on-ramp flow falls to about zero on 6 of the 8 Task 2
  panels. The median ratio to that ramp's own weekday/weekend
  time-of-day mean is 0.000 when queued, against about 1.0 otherwise. As a
  classifier of "mainline queued", low on-ramp flow reaches AUC 0.65–0.93
  depending on the corridor.
- **Off-ramps**: flow rises by about 6–10% when the mainline is queued
  (AUC about 0.65–0.70).

So the horizon ramp observations carry the forecast target, much like the
ramp speed column that was removed in the rebuild.

**Possible fixes.** Blank the ramp cells over each window's horizon and
buffer in `ramp_states`, as done for the mainline. Or score Task 2 only on
cells not inferable this way.

Happy to share the measurement script.
