# 13 — Overall review (2026-09-14): where we stand, why the top is at 0.92, and what to do next

> **Correction (same day, note [14](14_t2_ramp_route_proof_negative.md))**: the direct test shows that
> horizon ramp flows used as queue flags **lower** S_queue (0.707 → 0.575, and the placebo does the same). The
> claim in §3c that the ramp route "plausibly explains" the ≈0.92 top scores is **withdrawn**, and the draft
> report is on hold. The AUC observation stands; the "leaks the target" conclusion does not.

## 1. Where we stand
Public LB best: **0.80099** (C', ref 56229889).

| Task | Weight | Estimate | Contribution | Near-ceiling contribution |
|---|---|---|---|---|
| T1 State (M3 CNN) | 0.35 | 0.952 (holdout) | 0.333 | ≈ 0.34 |
| T2 Queue (rules v2) | 0.30 | **≈ 0.56** (implied by the LB) | 0.169 | 0.30 |
| T3 Physics | 0.15 | ≈ 0.67 (implied by the LB) | 0.100 | ≈ 0.144 |
| T4 ODME (own counts) | 0.20 | ≈ 0.99 | 0.198 | 0.20 |
| **Total** | | | **0.800** | ≈ 0.98 |

## 2. What the top (0.924, post-rebuild) would need
T1 and T4 are near their ceilings, and T3 is structurally bounded (§3a). To
reach 0.924, a team needs **T2 ≈ 0.9+**, or T2 ≈ 0.82 with T3 ≈ 0.96. A
~0.9 S_queue from a genuine 30-min forecast is far beyond everything we
measured: persistence scores 0.30 and our tuned rules about 0.70 on train
windows.

## 3. What we checked in this review
### a) Is the generator an exact CTM we could assimilate against? No, not usefully
`analysis/phys_eda01_ctm_check.py`, `work/phys_eda01_ctm_check.log`
- The truth sits on the released triangular FD. Median |q − Q_FD(k)|/q is
  0.6–1.1% in free flow and 2.6–2.9% in congestion, and free-flow v/vf is
  ≈ 1.000.
- Link-level conservation built from released flows fails completely: the
  error is 11–28× |ΔN| for link-flow, topology or Godunov fluxes, at spans
  from 5 to 60 min. Links are short (median 185 m on D12_I5_N) and the flow
  noise (≈ 18 veh/h/lane) dwarfs the accumulation change.
- ⇒ Exact physics-based reconstruction is not identifiable from the release.
  S_LWR will keep tracking T1 density accuracy (notes 04/05/07), so T3 stays
  roughly where it is.

### b) Were the validation windows harder than train? Partly
`analysis/t2_eda05_split_shift.py`, `work/t2_eda05_split_shift.log`
- Onset: the time of day, the weekend share and the missing speed
  precursor look the same across splits. But at validation T+30 the prior
  mass is **less concentrated** (top-3 share 0.355 against 0.45 on train).
  One panel (D7_I10_W) has almost no recurrent signal (top-1 prior 0.02),
  which suggests incident-driven onsets.
- Ongoing: the queues are smaller on validation and private (about 23 and 22
  slow links against 36 on train), which makes IoU more sensitive at the
  queue ends.
- ⇒ Part of the gap between the local S_queue (≈ 0.70) and validation
  (≈ 0.56) is window difficulty and small-sample noise (40 + 40 windows).

### c) What remains visible around a Task 2 window (audit only, never submitted)
`analysis/t2_leak_audit.py`, `analysis/t2_ramp_audit.py`, logs in `work/`

| Information | Effect on train official windows |
|---|---|
| Slot T of the masked layer | S_queue 0.707 → 0.708: nothing |
| Masked layer from T+95 on (after the 60-min buffer) | Crude use gives 0.474 (worse): onset queues are short-lived |
| **Horizon ramp flows** (ramp files are not blanked) | **On-ramp flow falls to ≈ 0** (median ratio to its tod mean 0.000) while the attached mainline link is queued, on 6 of 8 panels. AUC of low flow → queued is 0.65–0.93. Off-ramp flows rise 6–10% (AUC 0.65–0.70 reversed). |

- ⇒ The horizon ramp observations leak the forecast target, just like the
  routes the organizers already closed (mainline horizon values, the
  window_history path, and the ramp *speed* column removed at the rebuild).
  That plausibly explains how post-rebuild scores reach ≈ 0.92.

## 4. Policy
- **We do not use horizon ramp flows, post-buffer mainline data, or slot T
  for Task 2.** They are future information, outside the online contract.
  The organizers treat such routes as release flaws, and a disqualification
  risk is not worth it.
- **Pending user decisions**:
  1. Report the ramp-flow route to the organizers? A draft is in
     `notes/draft_report_ramp_leak.md`. Only the user posts.
  2. Allow same-day masked-layer data **before** T−60 (past information,
     not explicitly in the contract)?
- If the organizers close the route (rebuild or re-score), the leaderboard
  top should fall back toward the legitimate range, and our relative
  position improves.

## 5. Legitimate plan, prioritised
| Priority | Work | Expected gain (total score) |
|---|---|---|
| P1 | **T2 learned models**: a per-cell ongoing model (dissolve/spread at the queue ends, neighbour dynamics, priors), and an onset ranker (P_on, P_q, day group, neighbours, history ramp flows up to T−5) | +0.02 to +0.05 |
| P2 | T1: a 3-CNN ensemble (the width-128 production run is in progress) | ≈ +0.002, bundled into the next real submission |
| P3 | T3: comes with T1. The ensemble helps S_LWR more than S_state (note 07) | small |
| — | T4: done (≈ 0.99) | — |

A realistic legitimate target is about **0.82–0.85**.
