# 162 Is there room for a Lagrangian ongoing member? (the room test before training one)

Measurements: link spacing 0.04-1.4 km within a panel (median 0.16-0.64 km); the co-occurrence of queue onsets u
links upstream k slots later decays monotonically in u (no propagation peak in link units) - waves need km
coordinates. Script `occ_lab/lagr_persist.py`: Q_T (queued at the last slot, or unobserved there and queued in the
last 15 min) carried upstream in km by w x elapsed time, union Q_T.

| w (km/h) | covers AU's missed new cells (val / priv) | cells added outside AU's set | precision of the added cells |
|---|---|---|---|
| 5 | 0.13 / 0.16 | 80k / 75k | 0.041 / 0.052 |
| 15 | 0.38 / 0.43 | 167k / 156k | 0.031 / 0.042 |
| 25 | 0.50 / 0.55 | 248k / 237k | 0.025 / 0.032 |

AU's logit + beta on the added cells loses at every (w, beta): best -0.0041 / -0.0057 all, -0.0003..-0.0008 d1-5.
Upstream transport covers up to half of the missed new cells but only 3-5 % of the cells it adds queue: growth is
the exception and AU already places it; the transport prior carries no information beyond AU. No room for a
Lagrangian member.
