# 164 ASDA-style moving-jam tracking (Kerner's ASDA/FOTO): room test

`occ_lab/asda_room.py`: T-clusters whose downstream front moved >= 0.3 / 0.6 / 1.0 km upstream over the last 10 min
(6-11k 'moving jams' per month) advected upstream at the measured speed (5-30 km/h), both fronts. Covers 0.9-1.3 %
of AU's missed new cells; adds 3-5k cells outside AU's set at precision 0.03-0.05; AU logit + beta on them loses at
every setting (-0.0001..-0.0017). Tracking what the history shows moving does not raise the precision of note 162's
transport - the fronts that moved over the last 10 min do not keep moving often enough.
