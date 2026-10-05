# 161 Where the 'new' ongoing queues come from, and literature of that problem type

Lab pool, AU (tau 0.5). Signed distance of queued-in-the-horizon cells that are > 2 links from the nearest observed
T-queue cell (+ = upstream; upstream per note 159):

| | -20.. | -20..-10 | -10..-3 | 3..6 up | 6..10 up | 10..20 up | 20.. up |
|---|---|---|---|---|---|---|---|
| true new T+5 | 0.08 | 0.01 | 0.16 | 0.33 | 0.12 | 0.07 | 0.05 |
| true new T+30 | 0.09 | 0.03 | 0.09 | 0.27 | 0.16 | 0.15 | 0.13 |
| missed T+30 | 0.16 | 0.06 | 0.10 | 0.16 | 0.16 | 0.15 | 0.19 |

New queued cells sit upstream and move further upstream with the horizon; 15-17 % of the misses are independent
queues > 20 links downstream. Of the upstream cells 3-20 links away, 83-85 % are CONTIGUOUS with the T-queue in the
truth (tail extension or a queue unobserved at T; missed 4-7 %) and 15-17 % DETACHED (stop-and-go waves travelling
upstream; missed 12-14 %). Detached waves are roughly 40 % of the upstream misses, ~1/4 of the new-queue misses.

Literature of this type: Lagrangian CNN (Ritvanen et al. 2023, IEEE JSTARS; code fmidev/lagrangian-convolutional-
neural-network; differentiable version LUPIN 2402.10747) separates motion from growth/decay by moving the inputs
into a frame advected with the field; convective-initiation nowcasting (object-based probabilistic CI, AIES 2024);
STGNPP (AAAI 2023), a neural point process for the time and duration of the next congestion event.
