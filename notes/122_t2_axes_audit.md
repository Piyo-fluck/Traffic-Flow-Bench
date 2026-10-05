# 122 - What the Task 2 stack is actually worth, and which axes are closed

Three questions were settled on the same day: what each layer of the ongoing stack buys, whether the answer has
a low-dimensional geometry, and whether the day's longer context can fix the onset misses.

## The stack prices out flat

`occ_lab/ongoing_ablation.py`, ongoing family IoU after tau 0.50 on the official-like pool of both scored months:

| stack | validation | private | mean |
|---|---|---|---|
| old = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 (**12 models**) | 0.8357 | 0.8233 | 0.8295 |
| **p_PS alone (1 model, adapted to the scored run)** | 0.8388 | 0.8194 | **0.8291** |
| blend = 0.40 old + 0.60 p_PS | 0.8510 | 0.8338 | 0.8424 |
| blend + corrector (what AS ships) | 0.8586 | 0.8424 | 0.8505 |
| blend + corrector without the occ/stid columns | 0.8519 | 0.8419 | 0.8469 |

**Twelve frozen models equal one adapted model.** The twelve that feed only the corrector (occ_cnn x4, occ_tfm
x4, ongoing STID x4) are worth +0.0036 locally; on the leaderboard the whole corrector was +0.00031 (AH 0.87899
-> AI 0.87902) and its five variants lost. Against the LB record, the only changes after M that paid were the
two that added a KIND of model that was not there: AN (onset link identity, +0.00251) and AP (pseudo-supervised
adaptation, +0.00158).

## The answer is not a low-dimensional geometry

A queue reads like an interval of links, and the per-window budget oracle of note 121 is an interval length, so
the obvious reformulation is to predict the geometry instead of 900 cells. `occ_lab/queue_geometry.py` refutes
it. Over 37,952 (window, step) pairs holding a queue, the truth averages 14.1 queued cells, and filling the
truth's OWN convex hull scores **IoU 0.5817 against our 0.8722**. The hull holds about 24 eligible links and the
queue occupies 58% of it: the per-cell model is picking the right 14 of those 24, which is exactly what an
interval throws away. (The link axis is genuinely spatial - `ctx.z["link_ids"]` matches the topology's
`order_index` - so this is not an ordering artefact. Link lengths run 0.12-1.42 km and sensorless links sit in
between, so adjacent eligible links can be kilometres apart.) **The per-cell representation is right.**

## The onset misses are not a context-length problem

Note 51: on 12% of onset windows the answer scores 0, those windows carry the largest top1-top2 margin, and the
chosen link sits a median of 26 links from the true one, while the true bottleneck is in the top-3 candidates
98-99% of the time. Since every member reads only the 60 minutes of `window_history.parquet` while the rules
allow every released observation at or before T, the natural hypothesis is that the model cannot see why today
is unusual. `occ_lab/onset_bottleneck_rerank.py` tests it: candidates are the fit-day first-onset peaks (>= 3
links apart, top 10, Voronoi), and everything - the window model included - is fitted on days before FIT_END and
read on the 368 held-out onset windows.

| top-1 bottleneck accuracy | |
|---|---|
| habitual prior alone | 0.5788 |
| 60-minute window model alone | 0.7473 |
| prior + 60-minute window model | **0.7636** |
| the day's evidence alone (morning build-up, 3 h and dawn-to-T levels, ramp inflow, recurrence, demand vs the same weekday) | **0.7609** |
| both | 0.7554 |

The day's longer context reaches the window model's accuracy on its own and **adds nothing on top of it**. The
60 minutes already carry it. The hypothesis is refuted.

One thing to keep from it: a LightGBM over 24 aggregate features, fitted on the fit days only, reaches 0.7636,
where note 51 measured the production ten-model onset blend at 0.72-0.80 on the scored months. **Bottleneck
selection is not gaining from the stack.**

## What this leaves

The one axis with a mechanism demonstrated on this dataset and no attempt behind it: note 119's finding that SSL
pretraining is what makes test-time adaptation pay (TTT moved S_LWR -0.0024 on a model that had never seen a
masking task and +0.0026 on a pretrained one; the mask distribution was irrelevant). Task 2's adaptation already
works - one adapted model equals twelve frozen ones - on a width-64 backbone with no pretraining at all.
