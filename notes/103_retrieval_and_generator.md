# 103. Retrieval memory, and reverse-engineering the generator

Date 2026-09-24. Code `analysis/t1_knn_residual.py`, `analysis/t1_quantisation.py`, plus a re-run of
`analysis/phys_eda01_ctm_check.py`. Two of the user's three proposals from the Kaggle-solution survey; the third
(auxiliary heads) was not run.

## 1. Nearest-neighbour retrieval of past residuals: no signal

Keep the CNN, and correct it with the residuals of cells whose situation resembles the query's, using the CNN's
own penultimate representation (64-d) as the search key. Database = the January target cells of the local test
block (312,706 sampled), queries = February (29,302). Both are out of fold for the CNN.

| retrieval | corr with the true residual | sd of the retrieved residual | RMSE at alpha 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| no correction | - | - | 29.452 | | |
| top-1, any panel | +0.010 | 28.893 | 30.253 | 32.671 | 41.046 |
| top-5 | +0.009 | 12.910 | 29.600 | 30.094 | 32.051 |
| top-20 | -0.002 | 6.476 | 29.500 | 29.637 | 30.170 |
| top-100 | +0.008 | 3.036 | 29.456 | 29.478 | 29.583 |
| top-5, same panel | +0.014 | 13.034 | 29.587 | 30.076 | 32.042 |
| **random neighbour (control)** | **-0.004** | 29.663 | 30.400 | 33.028 | 41.884 |

The correlation is indistinguishable from the random control at every k, and every alpha makes the error worse.
This is not a database-size artefact: averaging 100 neighbours cuts the retrieved residual's sd to 3.0 - heavy
denoising - and the correlation stays at +0.008. Real signal would surface under that much averaging.

Note 100 found residuals nearly independent between neighbouring cells in time and space (+0.04); this adds that
they are also independent between neighbours in the network's own representation space.

## 2. The generator: no lattice, no CTM, no trivial free-flow law

- **Quantisation.** On the true layer, flow takes 71% distinct values with gaps down to 6e-5, and for every
  candidate step (1, 2, 3, 4, 6, 12, 15, 20, 30, 60 veh/h) the mean offset to the nearest multiple is 0.250 of a
  step - exactly uniform. Per lane, per link, and on single-lane links alike. Speed and occupancy the same.
  There is no count lattice to exploit.
- **Measurement noise: none.** Where both layers exist, `obs == true` to the last bit (mean and max |difference|
  0.000000 over 5,954,183 cells). The released values ARE the simulator state. This retires the premise of the
  earlier CTM work (note 13: "no exact CTM from noisy flows") - the mismatch was never about observation noise.
- **CTM.** Re-running the check with that in mind changes nothing: against a baseline of predicting no change
  (E = 1.000), the link-flow balance scores 21.1, the topology fallback 22.4 and the Godunov CTM 23.3 on
  D12_I5_N, and 11.3 / 12.3 / 15.8 on D7_I210_W. The generator is not reproducible from the released topology
  and FD parameters.
- **Free-flow speed.** The FD check reports a median free-flow v/vf of 0.9997, which suggests v = vf might be
  exact. It is not: the 1st to 99th percentile of v/vf in free flow is 0.975 to 1.022 and only 8.4% of cells sit
  within 0.1% of vf. Predicting v = vf gives a speed RMSE of 2.053 against the CNN's 1.269.

## Reading

Both branches are closed. Together with notes 96-102 the picture is consistent and now quite strongly
established: **the Task 1 residual is cell-specific and carries no structure in any coordinate system tried** -
mask geometry, corridor position, day, time of day, flow level (beyond a 0.111 mean bias), low-rank factors,
neighbouring cells, or the network's own embedding space - and it is not measurement noise, since the released
values are exact. It is the part of the simulator's dynamics that the masked context does not determine.
