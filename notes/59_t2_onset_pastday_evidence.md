// filepath: notes/59_t2_onset_pastday_evidence.md
# 59 — T2 onset: the rare bottlenecks we miss ARE active on earlier days, but past-day activity cannot rank them

Date: 2026-09-16 · Script: `analysis/t2_onset_pastday_evidence.py` · Log: `work/t2_onset_pastday_evidence.log`
(11 s) · Data: `work/t2_onset_pastday_evidence.csv`

The user's top-priority diagnostic: note 51 showed that on the ~12% of onset windows M misses, the true bottleneck
is a median 26 links away and rare **in train**. Is it nevertheless visible as "active" in the **same simulation
run**, using only the released layer of strictly earlier days (the user's 2026-09-14 rule)? Their acceptance
criterion: ≈ 70% → large untapped information; ≈ 10% → discard the idea.

Candidates = the Voronoi cells of the train first-onset peaks (note 51). Evidence per candidate, from days < d of
the same split, 04:00–14:00: queued slots (v ≤ 0.60 vf), slow slots (v ≤ 0.85 vf), days with any queue, max q/cap,
and the longest daily queued run.

## Availability (a prior worry of mine, and it was wrong)

Note 19 warned that official windows sit on days 1–7, so past-day methods would have nothing to work with. For the
blank-aware official-like onset set that is **not** the case: every one of the 331 windows has at least 3 past
days (3 → 1 window, 4 → 4, 5 → 9, 6 → 11, 7 → 11, 8 → 12, …, median ≈ 10). **No window is starved of history.**

## Result: presence is saturated, ranking is wrong

| missed windows (40) | true bottleneck | the bottleneck M chose |
|---|---|---|
| had a queued day before | **0.90** | 0.80 |
| queued slots per past day | 58.5 | **115.3** |
| slow slots per past day | 63.4 | — |
| max q/cap | 0.96 | — |

| correct windows (291) | true bottleneck | chosen |
|---|---|---|
| had a queued day before | 0.99 | 0.98 |
| queued slots per past day | 80.4 | 79.2 |

- **The 70% bar is cleared on presence (90%), but presence carries no information**: on the correct windows it is
  99%/98%, and on the missed ones the *wrong* bottleneck is also active 80% of the time — and about **twice as
  active** (115 vs 58 queued slots per day).
- Ranking the candidates by past-day queued slots puts the true bottleneck **first in only 17%** of the missed
  windows (median rank 2 of ~9 candidates; top-3 0.93). On correct windows top-1 is 0.54.
- Past evidence favours the true bottleneck over M's choice in only **30%** of the missed windows.

**A run-specific prior built on past-day congestion would therefore re-rank toward the habitual bottleneck — the
exact error M already makes.** The idea is rejected in this form. (Train frequency check, as in note 51: true
0.265 vs chosen 0.434.)

## What it implies for the next step

The missing information is not *which locations are active in this run* — nearly all candidates are. It has to be
*where today's demand goes*. That is precisely the user's next proposal: infer path/OD demand from the window's own
history (T−60…T−5) and compute the future load of each candidate bottleneck, instead of a location prior.

This diagnostic therefore redirects effort rather than ending it: item 2 of the user's plan is closed, item 3
(causal Task-4-style path-load inference) is the remaining onset lever, and it is the only one that would explain a
bottleneck 26 links away from the usual one.
