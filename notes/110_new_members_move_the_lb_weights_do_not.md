# 110. Candidate AR, and the pattern across ten candidates: new members move the leaderboard, weights do not

Date 2026-09-24. Candidate AR (LB 0.88318) against AP (0.88311).

## AR

AP put the note 106 pseudo-supervised model into the ongoing p0 at 0.30. `occ_lab/ongoing_pseudo_sweep.py` swept
that weight on the scored months with the **real** corrector features (`occ_lab/cache_og_members.py` - the first
local cache of occ_cnn, occ_tfm and the ongoing STID, without which the corrector reads substituted members and
its column is distorted):

| weight | validation, + corrector | private, + corrector |
|---|---|---|
| 0.30 (AP) | 0.8486 | 0.8385 |
| 0.50 | 0.8546 | 0.8420 |
| **0.60** | **0.8586 (+0.0099)** | **0.8424 (+0.0040)** |
| 0.70 | 0.8594 (+0.0107) | 0.8411 |
| 1.00 | 0.8510 | 0.8310 |

0.60 is the joint interior optimum. AR raises the weight to 0.60: **356 ongoing rows differ from AP**, onset
untouched. Expected from the local curve: 0.15 x 0.0070 = +0.0011. **Realised: +0.00007.** Realisation 0.07,
against AP's 0.8.

## The pattern

Ten candidates since the corrector, sorted by what kind of change they were:

| change | candidate | LB delta |
|---|---|---|
| **a genuinely new member** | AN (onset STID) | **+0.00251** |
| **a genuinely new member** | AP (pseudo-supervised adaptation) | **+0.00158** |
| a member's weight, along a plateau | AR (ongoing 0.30 -> 0.60) | +0.00007 |
| a member's weight, concentrated | AQ (adaptive slot to one member) | -0.00094 |
| a member's weight, at the edge | AO (H 0.35 -> 0.20) | -0.00460 |
| the corrector's fitting set | AJ, AK, AM | -0.0001 to -0.0008 |
| the corrector's selection | AL (stability) | -0.00022 |
| a mixture tweak | AH (occupancy) | -0.00003 |

**Adding a model that was not in the blend gains. Moving weights among models that are already in it does not**,
whatever the local curve says, and the local curve's slope along a plateau carries no information about the
leaderboard even when its sign at the plateau's entrance does.

This is consistent with note 108's mechanism: the local surfaces are the same run the members were fitted on, or
the right run at the wrong coverage. Either way they measure how members combine *here*, not *there*. What a new
member adds survives that; where the optimum sits does not.

## Reading

AR is the best submission at 0.88318, but AP at 0.88311 is the same result. The operating rule from here is to
spend submissions on **new models**, not on retuning the ones we have.
