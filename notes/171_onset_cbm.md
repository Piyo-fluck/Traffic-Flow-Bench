# 171 Onset as a Concept Bottleneck Model (CBM), trained on the training months only

Scripts: `occ_lab/cbm_onset.py` (training), `occ_lab/cbm_eval.py` (lab evaluation). Models `occ_lab/models/cbm_*.pt`,
lab probabilities `occ_lab/cache/cbm_<tag>_<split>_<panel>.npz`. Logs `work/cbm_*.log`, `work/cbm_eval_*.log`.

## Model

- **Concept net** (no link identity):
  - Input: the 12-slot history of v/vf, q / train q95, ok, on-ramp and off-ramp flows, plus static road channels
    and time of day.
  - Body: dilated 1-D convolutions along the corridor.
  - Output, per link: queued at T+30 and at T+15, v/vf at T+30 and T+15, q/q95 at T+30.
  - Training data: every window of the 273 training days, masked to links whose +-5 neighbours are queue-free
    over the history (the onset precondition).
- **Onset head g:** a tiny convolution over the concepts of +-3 links. It is trained on the training months' onset
  windows (2,072 of them) with the concepts frozen (sequential CBM).
- **Use:** a second opinion on production's candidate choice. The rank of a candidate = production's region mass
  x (the CBM probability maximised over the region) ^ lambda. Where the top candidate changes, the two peaks are
  swapped (note 129's rule). No scored-month data is used in training; the only choice is lambda.

## Evaluation condition

- BB's day-adapted members with K = 0..4 days of adaptation (the official windows sit on days 0-4), averaged over K.
- Note 129's classifier is fitted cross-month (3 seeds). **Its run-evidence features were restricted to the K
  earlier days** (`ONSET_SHARE_K=1` in `bottleneck_cls.py`). With the full month it read 0.805 / 0.827; under the
  fair condition it reads 0.791 / 0.819.

## Results (lab, 169 val / 162 pri windows)

| | top-1 val / pri | IoU val (SE) | IoU pri (SE) |
|---|---|---|---|
| production | 0.753 / 0.816 | | |
| note 129 swap (fair) | 0.791 / 0.819 | +0.0170 (0.0095) | +0.0066 (0.0129) |
| CBM alone, 4 seeds | 0.73 / 0.73-0.76 | -0.09 | -0.08..-0.11 |
| CBM rf x production, lambda 1, seeds 0-3 | 0.822 / 0.821 | +0.0182 (0.0094) | +0.0098 (0.0071) |
| same, seeds 4-7 (replication) | 0.780 / 0.820 | +0.0052 (0.0049) | +0.0090 (0.0068) |
| **same, 8 seeds** | **0.799 / 0.822** | **+0.0093 (0.0066)** | **+0.0098 (0.0071)** |

Findings:
- **Single CBMs vary wildly between seeds:** seed 0 scores 0.619 / 0.800, seed 1 scores 0.716 / 0.699. Yet every
  seed scores 0.91-0.93 on the training months' held-out onsets. The training run does not pin the solution down,
  and held-out days of the same run cannot rank models. Compare ensembles only.
- **The concept pretraining matters:** the same backbone trained end-to-end on onset windows only reached 0.646 / 0.652.
- **Narrowing the receptive field (dilations 1,2,4,1,2,4 vs 1..16) helps in both months:** as a 4-seed ensemble
  combined with production, +0.018 / +0.010 vs -0.000 / +0.006. Removing the static channels was not consistent.
  Reading: a wide field plus static channels lets the net fingerprint locations, i.e. rebuild the link identity.
- **Residual intervention** (today's surprise at slot T-1 fed to g) did not help (1 seed).
- **Lambda:** 1.0 is best in both months for the 8-seed ensemble.
- **Per panel** top-1, production -> CBM x production:
  - Gains: I5_S val 0.22 -> 0.57, I5_N val 0.88 -> 1.00, I405_S pri 0.41 -> 0.48.
  - Losses: I210_E pri 1.00 -> 0.96, I405_N pri 1.00 -> 0.96, I210_W val 0.88 -> 0.86.
  - The gains sit where the scored run's active heads differ from the training run's (note 169).

## On top of BB's set decoder (`occ_lab/cbm_setdec_check.py`, production-like: fitted on month A, applied to early month B)

The CBM swap is applied after stage 1. Where the top changes, the stage-1 score is re-ordered so stage 2 sees the
new top.

| | vs setdec, mean over K (SE) |
|---|---|
| validation | +0.0007 (0.0078) |
| private | -0.0001 (0.0011) |

- By K: positive at K = 0-1 (val +0.008 / +0.008, pri +0.004 / 0), negative at K = 3-4 (val -0.005 / -0.009,
  pri -0.002 / -0.002).
- The set decoder already makes most of the corrections the CBM makes.

## Candidate CE = BB + CBM after set decoder stage 1: public 0.88663 (BB 0.88697, -0.00034)

- Build: `scripts/t2_onset_blend_predict.py ... --setdec --cbm occ_lab/models/cbm_cbm_rf_s{0..7}.pt` ->
  `work/sub_t2CE_onset`, then patched into BB -> `work/sub_CE/submission.csv`.
- The CBM swapped 6 of the 80 official windows (val 4, pri 2).
- 10 cells differ from BB, in 3 windows:
  - I5_N val 003
  - I405_S val 002
  - I405_S pri 001 (7 cells)
- -0.00034 is about 1.4 onset cells' worth.

## Remaining loss after the set decoder, and CBM proposals (`cbm_gap.py`, `cbm_cand_oracle.py`, `cbm_override.py`, `cbm_rule.py`, `cbm_loc_oracle.py`)

Loss per window after the set decoder:

| loss | val | pri |
|---|---|---|
| choice | 0.053 | 0.048 |
| missed site | 0.043 | 0.043 |
| extent | 0.047 | 0.031 |
| multi-site | 0.014 | 0.013 |

- **CBM proposals:**
  - Adding the CBM's peaks as candidates raises the oracle by +0.10 / +0.08.
  - A learned override (LightGBM on proposals, about 165 windows a month) learns nothing (tau = inf).
  - The rule "replace when the CBM is >= 0.2 surer elsewhere" gives val +0.018: 3 I5_S val windows, 0 -> 1 each. It
    fires in 1 official window (I405_S pri), so CF was not submitted (the team's quota was used).
- **Re-localising the chosen site** (shift / width within +-3 links) has an oracle of only +0.027 / +0.016. The
  missed-site loss is a wrong-site loss, not a misplacement.

## MixCEM-style gate (`cbm_gate.py`): the CBM swap only where the window looks out of the training run's distribution

- **Ungated CBM swap after stage 1:**
  - lambda 1: +0.0007 / -0.0001.
  - **lambda 2: +0.0071 / +0.0022.** lambda 2 is best in both months.
  - lambda 4: +0.0033 / -0.0038.
- **Gates, cross-month threshold:**
  - Demand-level deviation, lambda 1 (CBM when today's demand < the training profile; the same direction from both
    months): val +0.0067 (SE 0.0060), pri +0.0008 (SE 0.0006).
  - Best at lambda 2: spatial-pattern deviation (dev_mad < 0.02), val +0.0098 (SE 0.0087), pri +0.0035 (SE 0.0025).
    But this direction means "use the CBM when the pattern is LIKE training", the opposite of the MixCEM logic.
  - No score isolates out-of-distribution windows clearly. The gains stay within 1-1.5 SE, and the gate adds little
    over the ungated lambda 2.

## OceanCBM-style joint training (`cbm_onset.py --mode joint`, rf dilations, 1 free concept, onset weight 0.5)

Stopped after 4 seeds, on the user's call.

| | top-1 val | top-1 pri |
|---|---|---|
| single seeds, joint | 0.58-0.67 | 0.746-0.758 |
| single seeds, sequential rf (for comparison) | 0.69-0.73 | 0.71-0.77 |
| 4-seed ensemble, joint | 0.583 | 0.736 |

- The private spread narrows, but validation (the month further from the training run) drops.
- The onset loss pulls the concepts toward the training run's site habits (leakage).
- Not pursued.
