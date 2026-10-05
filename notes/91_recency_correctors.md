# 91. Recency for the residual corrector: same-month days, and a month-specific corrector

Date 2026-09-22/23. Code `occ_lab/causal_corrector.py`, `occ_lab/fit_corrector_recent.py`, logs
`occ_lab/logs/causal.log`, `causal_day.log`. Motivation: every measurement said the corrector is recency-bound
(tail -> March / April +0.0044 / +0.0045; adjacent month +0.0117 / +0.0079; Jan -> Feb +0.017 .. +0.025).

## A. The scored month's own earlier days (causal, as idea 9b)

For each day d of a scored month the corrector is refitted on that month's days < d and only day d is predicted.
Over all 31 days this looked decisive - `past` +0.0153 / +0.0073, `tail+past w3` +0.0140 / +0.0100 against
AI's +0.0046 / +0.0042. **That average is not the official window distribution.** The official ongoing windows
are 40 per month and sit on days 1-6 (March: 33 of 40; April: 30 of 40), where the month's own pool is empty or
tiny. Restricted to days 1-14, at AI's setting:

| variant | day 1 | days 2-3 | days 4-7 | days 8-14 |
|---|---|---|---|---|
| tail (AI), March / April | -0.0026 / -0.0088 | -0.0037 / -0.0013 | +0.0013 / +0.0030 | +0.0069 / +0.0044 |
| past only | (= tail) | +0.0021 / **-0.0641** | -0.0006 / -0.0083 | +0.0151 / +0.0135 |
| tail+past w3 | (= tail) | -0.0074 / +0.0005 | +0.0059 / +0.0028 | +0.0136 / +0.0101 |

Weighted by the official day counts: tail +0.0000 / +0.0002, tail+past w3 +0.0016 / +0.0016. The principle
survives but the size collapses, and `past` alone breaks on the thin early pools (-0.0641 on April 2-3).
**The day distribution of the official windows should have been applied before designing this, not after.**

## B. One corrector per month, recent data weighted up (candidate AM)

What is available without the thin-pool problem: every March observation precedes every April origin, so April's
corrector may use all of March. March's may only use the tail.

    March   tail (2031-01/02), February weighted 3x            3,881,167 cells
    April   tail + all of March, March weighted 3x             5,815,001 cells

Applied at AI's setting; onset and the six members unchanged. 320 queue rows differ from Z, 126 from AI; 145 /
175 corrected window cells (AI: 150 / 172).

## LB: AM = 0.87893 (submission 56476231), -0.00009 against AI

AM did not beat AI (0.87902); it sits with AH (0.87899) and above Z (0.87871). The recency effect, which is the
largest and most reproducible quantity measured anywhere in the corrector work (2-5x the tail-only gain, in both
directions, in every split tried), **did not transfer to the LB at all.**

This is now the fifth corrector variant (AJ, AK, AL, AM, plus the horizon split not submitted) that improves the
local number and does not improve the LB. AI's plain tail-fitted corrector remains the best.
