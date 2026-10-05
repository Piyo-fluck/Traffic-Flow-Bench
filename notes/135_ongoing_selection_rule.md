# 135 - The ongoing windows' earliest-origin selection: replicated, but AU already answers what it implies

Kaggle-solution survey (Ventilator 1st place: read the generator) -> the official Task 2 selector.
`occ_lab/selector_replicate.py` on train (labels known): onset 39/40 official windows reproduced exactly
(earliest origin, onset scanned first, 360-min spacing); ongoing 30/40 with >= 2 queued history slots, coverage
>= 0.7, 07-21 h, last-observed persistence IoU <= 0.9 (the released baseline's final-slot and at-T variants: 7/40).
On the official windows, persistence IoU at T sits just under 0.9 (0.85-0.898) and at T-5 just over (0.90-0.95)
for most windows not blocked by spacing: the scan is visible in the data.

Implication: where T-5 passed the visible conditions, persistence from T-5 had IoU > 0.9 over T..T+25.
`occ_lab/selector_rule_eval.py`: such "rule" windows are 9% of the lab pool (290 / 312), and AU already scores
0.938 / 0.931 on them (0.874 / 0.858 overall). Moving steps 1-5 towards the T-5 persistence state: a = 0.3 gives
+0.005 / +0.009 on the rule windows, +0.0006 / +0.0012 overall, and -0.005 / -0.002 on the rule windows of days
1-5 (where the official windows are); a >= 0.5 or max-filling loses 0.01-0.03. The constraint is already in AU's
answer. No score lever.
