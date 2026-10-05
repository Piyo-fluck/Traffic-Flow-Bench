# 128 - Two new onset identities (link x lag, pattern memory): both spoil, none rescue

User-supplied proposal (2026-09-26): after STID's link identity (AN, +0.0025 LB), find the next unexpressed
identity. Two of its four items were already expressible by STID (an ADDITIVE link x lag input embedding collapses
into the link embedding after STID's linear series projection; a low-rank link x horizon bias is what the head
already is). The two genuinely new ones were added to `occ_lab/train_onset_stid.py`:

    mult   per-link multiplicative weights (1 + tanh w) on every history slot x channel before the projection
    pm     pattern memory: the link's history encoded, attended over 48 learned prototypes, value appended

4 seeds each, the production recipe (T+30 only, 8000 steps); train-run ES bce 0.0139-0.0146 (as the production
onset STID). Official-like onset pool (`occ_lab/onset_new_eval.py`), production blend 0.35 H + 0.175 ad9b + 0.175
p_PS + 0.30 STID = 0.8160 / 0.8623:

| member | alone val / priv | + at 0.05 | + at 0.10 | rescue / spoil at 0.05 |
|---|---|---|---|---|
| mult | 0.7524 / 0.7836 | -0.0029 / -0.0015 | -0.0066 / -0.0045 | 0 / 2 |
| pm | 0.7862 / 0.8312 | -0.0029 / -0.0015 | -0.0226 / -0.0045 | 0 / 2 |

No rescues at any weight. The finer link-specific structure (mult) transfers worst across runs.
