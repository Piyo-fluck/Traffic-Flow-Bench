# 172 Blind judging of production's close two-way onset calls (can a reasoning model beat production?)

## Setup

- **Windows:** lab windows, K = 0 condition.
- **Close call:** after BB's set decoder stage 1 (production-like, cross-month rankers), the 2nd candidate's region
  mass is >= 0.5 x the top's. There are 103 such windows; 20 per month were drawn (seed 171).
- **Dossier per window** (`occ_lab/llm_dossier.py` -> `work/llm_judge/dossiers/`), data released at or before T
  only. Per candidate site:
  - production's region share
  - training-run habit
  - train discharge capacity
  - today's upstream on-ramp demand / training profile
  - 12-slot v/vf and q/capacity at the site +-3 links
  - the forward CTM density / critical at T+5..T+30
  - The candidates appear as X / Y in random order.
- **Judging:** the judge (Claude, in session) read all 40 dossiers and recorded calls in `work/llm_judge/judgments.txt`
  before opening `key.json`.

## Result

| | judge | production | disagreements (judge won / production won / neither) |
|---|---|---|---|
| validation (20) | 13 | 11 | 3 / 1 / 1 |
| private (20) | 12 | 15 | 1 / 4 / 1 |
| total (40) | 25 | 26 | 4 / 5 / 2 |

- The judge is no better than production on the close calls; the judging gate of the plan is not passed, so no
  open-source LLM step.
- **The dossiers carry no discriminating signal in many close calls:**
  - D7_I10_E: links 5 vs 9/10.
  - D7_I210_E: links 0 vs 4.
  - Both sites are in free flow at q/cap ~0.3-0.6 until T, so the judge fell back on production's share.
- **Where the judge overrode production on physics, the record is mixed:**
  - Forward-CTM density higher: won 2, lost 3.
  - "Flow at the site's capacity" (I405_S link 32 / 11): won 1, lost 2. A link at capacity for an hour does not
    break first.
