# 49 — Link orientation: S/W panels run downstream→upstream in our link order (teammate's EDA, checked)

Date: 2026-09-16 · Source: the teammate's `EDA_TASK1.md` §2.3 / §8.2 · Check: scratch scripts `queue_direction{,2}.py`

**Claim (teammate).** `lwr_mainline_topology.csv` `order_index` follows milepost order in every panel. In N/E panels
that is upstream → downstream; in S/W panels it is **downstream → upstream**. Our loader (`tfb/data.py`) sorts every
panel by `order_index` and never re-orients.

**Check (train truth, our link order).** At every queue-onset cell (queued at t, not at t−1), compare whether the
link 6 positions away was already queued at t−1. The queue sits downstream of the new tail, so the more likely side
is downstream:

| panel | P(s+6 queued) | P(s−6 queued) | downstream = |
|---|---|---|---|
| D7_I10_E | 0.270 | 0.327 | lower index (unexpected) |
| D7_I10_W | 0.230 | 0.198 | higher index (unexpected) |
| D7_I210_E | 0.367 | 0.316 | higher index |
| D7_I210_W | 0.146 | 0.160 | lower index |
| D7_I405_N | 0.715 | 0.652 | higher index |
| D7_I405_S | 0.315 | 0.400 | lower index |
| D12_I5_N | 0.562 | 0.540 | higher index |
| D12_I5_S | 0.377 | 0.370 | higher index (≈ tie) |

- The claim holds for 5 panels clearly and for D12_I5_S only weakly; the two I-10 panels come out the other way.
  The asymmetries are small (0.01–0.09). The first check (the direction in which a new queue spreads within 15
  min) was symmetric: queues barely move at 5-min steps (note 09).
- **Affected parts of our T2 pipeline (all panels pooled):**
  - the demand-stress feature `pqc_dn` (`t2_demand_stress.stress`: the max over the next 4 higher-index links).
    For S/W panels it is actually the UPSTREAM max;
  - the `dist_slow_up` / `dist_slow_dn` names, which are really lower- and higher-index;
  - the CNN's direction-specific kernels and the transformer's signed relative-position bias. Both learn mixed
    semantics across panels.
- Symmetric neighbour features (±1, ±2, ±4) and the panel id let LightGBM partly compensate.
- **Check of the past capacity-margin features (notes 21, 22, 27), done on the user's request; stopped after it:**
  - `analysis/t2_struct_noid.py` defines "downstream" as `at(x, k)` = the value at index s+k, for EVERY panel. All
    of these therefore look UPSTREAM in S/W panels:
    - the margins `mg_q_dn1`, `mg_dn_max4`, `mg_tight`, `mg_tight_proj`;
    - the structure features `st_lane_drop*`, `st_cap_ratio*`, `st_on_dn1`, `st_merge_dn`, `st_dist_bneck`;
    - the bottleneck flag.
  - Its `effective_capacity` (and `analysis/t2_capacity_transfer.py`) detect a queue head as "link s free-flowing,
    link s−1 queued", which assumes that the lower index is upstream. In S/W panels this picks queue TAILS, so the
    estimated discharge capacity is wrong there.
  - `t2_capacity_transfer.py` ran by default on D7_I10_E, D12_I5_N and D7_I405_N, all N/E panels. Note 21's "capacity
    transfers within 0.5%" therefore stands for those panels (D7_I10_E is the anomalous one in the table above).
  - **The margin and structure tests that were negative for onset (notes 22, 27) pooled 4 S/W panels with the wrong
    direction.** Their failure may be partly an implementation artefact, not evidence against the idea.
    **Re-tested in note 50: it is not. Re-oriented margins do no better; the rule orientation is the worst.**
- **Candidate experiment (not run):** re-orient the link axis so that every panel runs upstream → downstream,
  either by the N/E vs S/W rule or by the measured sign per panel. Then re-train the T2 LightGBM (H, I) and the CNN
  and compare paired on official-like windows.
