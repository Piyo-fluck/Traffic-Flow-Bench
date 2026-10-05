# 117. What the residual actually is: two or three cars, in the most ordinary place on the road

Date 2026-09-25. Code `analysis/t1_look_at_cases.py`, `analysis/t1_transition_share.py`.

Notes 96-116 established what the residual does not correlate with, across ten directions. None of them said what
it **is**.

## In physical units

Flow is a count over a five-minute bin. Per lane, on the 7,089,289 target cells of the local test block:

| | flow, vehicles per 5 min per lane | speed, km/h |
|---|---|---|
| the truth itself | median **81.8** | median 114.3 |
| our error | RMSE **2.47**, median 1.17 | RMSE 1.31, median 0.73 |
| within +-1 | 45.2% of cells | 63.2% |
| within +-2 | 68.0% | 89.9% |
| within +-3 | 81.5% | - |

**We answer "82 cars" with "84".** Note 01's "noise floor" of 17.9 veh/h/lane is 1.5 cars; we are at 2.5. The
whole Task 1 argument is about one car per lane per five minutes. Relative: flow 3.24%, speed 1.18%.

## The worst cells are all one phenomenon

Reading the largest errors one at a time, rather than summarising them, gives the same picture every time, on
different corridors and days: the truth holds a queue (about 120 vehicles per 5 min per lane at 38 km/h), then
**flushes it in a single slot** - flow spikes to 145-190 while speed jumps to 72-100 - and settles into free flow
the slot after. We put the release one slot early and miss the discharge spike entirely.

    L405N-101  00:35 target 119.7 -> 140.4    00:40 target 147.2 -> 62.3   (error -85 vehicles)
    L405N-037  00:35 target 121.2 -> 139.4    00:40 target 145.9 -> 63.2   (error -83)
    L5N-133    18:15 target 174.5 -> 123.4    18:20 target 175.1 -> 95.3   (error -80)

A one-slot timing error is the worst possible shape for S_LWR, which scores `|e[t+1] - e[t]| L`: one large
residual and then another of the opposite sign. Transitions are 0.2% of cells (note 96), so no pooled
correlation - including every test in notes 107 and 116 - could have seen this.

## But that is not where S_LWR is lost

Splitting the S_LWR numerator by distance to the nearest congestion crossing (v < 0.6 vf):

| band | transitions | share | numerator share | denominator share | local E_LWR |
|---|---|---|---|---|---|
| at the crossing | 24,757 | 0.17% | 1.25% | **7.45%** | **0.061** |
| within 1 slot | 218,961 | 1.54% | 8.63% | 14.52% | 0.216 |
| within 3 slots | 479,916 | 3.37% | 17.43% | 21.79% | 0.291 |
| **beyond 3 slots** | 13.75M | 96.6% | **82.57%** | 78.21% | **0.384** |
| all | 14,232,858 | - | - | - | 0.364 |

The numerator is concentrated at crossings by 7.4x - **and the denominator by 44x**. At the moment a queue
releases, our relative error is six times *better* than average, because the truth's own dN is enormous there.

**82.6% of the S_LWR numerator sits in ordinary cells more than three slots from any congestion transition**,
where the road is doing nothing, the truth barely changes, and our two-car wobble is large next to it.

## Reading

The tail is a real, nameable failure - queue-release timing - and it is worth 8.1% of the flow squared error in
3.4% of cells, so it belongs to S_state, not to S_LWR. S_LWR is lost in the quiet bulk, and the quantity that
has to shrink there is the two or three cars, which notes 96-116 have now failed to predict from ten independent
directions.
