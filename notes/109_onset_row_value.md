# 109. One onset cell is worth 0.00025 of the total - and mixing both adaptive members beats either alone

Date 2026-09-24. Candidate AQ (LB 0.88217), against AP (0.88311).

## The candidate

Note 108 said H does not travel between runs (-0.1305) while the adapted CNN does (+0.0038), so the adaptive slot
should go to the strictly better adaptive member. AQ is AP with one change: the onset mixture becomes
**0.35 H + 0.35 pseudo + 0.30 STID**, dropping ad9b entirely, ongoing untouched.

It moved **4 onset rows in 87,000** and lost 0.00094.

## The constant

Two independent measurements of what an onset decision is worth:

| candidates | onset rows changed | LB delta | per row |
|---|---|---|---|
| AO vs AN | 14 | -0.00460 | 0.00033 |
| **AQ vs AP** | **4** | **-0.00094** | **0.00024** |

**One onset cell moves the total by about 0.00025.** For comparison AP moved 258 ongoing cells for +0.00158,
about 0.0000061 per cell - an onset cell is worth roughly **40 ongoing cells**.

This is the shape of the onset problem, and it explains the whole history of onset candidates. There are only
about 326 predicted onset cells in the whole upload; the blend weights barely move them (AP moved 2, AQ moved 4,
AO moved 14), and each one that does move carries 40 ongoing cells' worth of score. Onset is a small number of
very heavy, nearly unmovable decisions - with no local surface that can sign them (note 108).

## What AP's onset actually is

| onset mixture | LB |
|---|---|
| 0.70 H + 0.30 ad9b (T..AM) | reference |
| 0.35 H + 0.35 ad9b + 0.30 STID (AN) | +0.00251 |
| **0.35 H + 0.175 ad9b + 0.175 pseudo + 0.30 STID (AP)** | **+0.00158 more** |
| 0.35 H + 0.35 pseudo + 0.30 STID (AQ) | -0.00094 against AP |
| 0.20 H + 0.40 ad9b + 0.40 STID(ab) (AO) | -0.00460 against AN |

Splitting the adaptive slot between ad9b and the pseudo-supervised model beats giving it to **either one alone**.
Note 108's reasoning (give the slot to the better member) was right about which members travel and wrong about
concentration: the two adaptive members disagree on the few cells that matter, and averaging them is what wins.

AP stays the best at 0.88311.
