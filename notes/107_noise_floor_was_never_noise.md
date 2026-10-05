# 107. The Task 1 "noise floor" was never noise - and the door it opens leads nowhere

Date 2026-09-24. Code `analysis/t1_noise_floor_doubt.py`, `analysis/t1_advection_probe.py`.

## The contradiction we were carrying

Note 01 F4 set the Task 1 ceiling at about 0.96 from the robust sigma of the truth's second time difference,
on the stated premise **"the truth is the noisy observation"**. Note 23 then built the Task 3 ceiling by
injecting density noise *at that measured size* and reading off S_LWR.

Note 103 (same day as this one) established that `obs == true` to the last bit on Task 2's layers and wrote
"this retires the premise of the earlier CTM work" - but nobody went back to notes 01 and 23. Both ceilings were
still being quoted, including to the user an hour before this note.

## The premise is false (test A)

`obs == true` on the Task 1 layers too: 4 panels x 3 fields, 38,281,239 cells, **max |obs - true| = 0.000000,
exact share 1.000000**. There is no measurement noise anywhere in the release.

## The sigma fails the noise signature (test B)

An i.i.d. error of any size forces the second difference to have rho1 = -2/3 and rho2 = +1/6, whatever the
underlying signal. Measured:

| panel | field | rho1 | rho2 |
|---|---|---|---|
| D12_I5_N | speed / flow | -0.504 / -0.540 | 0.016 / 0.048 |
| D7_I210_W | speed / flow | -0.551 / -0.593 | 0.055 / 0.102 |
| D7_I10_E | speed / flow | -0.559 / -0.593 | 0.060 / 0.098 |
| D7_I405_N | speed / flow | -0.527 / -0.578 | 0.046 / 0.073 |

All eight are far from the noise values. **17.9 veh/h/lane is the amplitude of deterministic high-frequency
dynamics, not an error bar.** Nothing in principle stops a model from reaching below it, so the "+0.002 left in
Task 1" bound and note 23's Task 3 ceiling are both void as stated.

## But the door leads nowhere (test C and the advection probe)

- Occupancy is blanked at target cells along with speed and flow (visible share 0.000000), so `q = k v` gives
  nothing for free.
- Task 4 is not the generator: `synthetic_link_counts.csv` is one row per link with no time axis and a single
  `SYN_PM_TRAIN` departure bucket. It cannot inform a 5-minute state.
- **Advection.** Adjacent links are 0.184 km apart, 5.7 s at free speed, so a one-slot lag needs ~10 km and ~50
  links. Scanning the (link offset, time lag) plane from -160 to +160 links and 0 to 8 slots, against the
  high-pass flow anomaly: the surface is flat. Max |corr| 0.0081 (D12_I5_N) and 0.0056 (D7_I405_N) over a
  million targets, no ridge, and the two panels peak in different places. **The residual has no upstream
  origin.**

Together with note 103 (no lattice, no CTM, no retrieval signal) the residual is deterministic, cell-local,
high-frequency, and not reconstructible from anything in the release we have found in six independent probes.

## Reading

The correct statement is now: **Task 1's ceiling is unknown rather than proven near, and we have no route to
it.** The practical conclusion is unchanged, but it no longer rests on a false premise - and we should stop
quoting note 01's 0.96 and note 23's Task 3 headroom as bounds.

The pool remains where note 106 left it. Onset carries 0.15 of the total and sits at about 0.74, the largest
unrealised block anywhere, and candidate AP showed its decisions are nearly invariant to the probability model
(0.175 of a member moved 2 cells in 87,000 rows). The binding constraint there is the decision rule, and the
local onset evaluation has never been trustworthy (note 76's coverage mismatch; AN +0.023 local -> +0.0025,
AO +0.033 -> -0.0046).
