"""What does the official S_LWR actually compare our dN against?

`tfb/physics.py` scores |dN_pred - dN_true| / |dN_true|, i.e. it assumes the conservation right-hand side IS the
truth's own dN. The official scorer does not:

    rhs   = dt * (inflow + on_ramp - outflow - off_ramp)          score_task3.py:394
    resid = |dN - rhs|,   E = sum resid / sum |rhs|

with inflow/outflow the flows on the topology's incoming/outgoing links and the ramps the released observations.
BASELINES.md says a perfect answer scores S_physics 0.9636, so S_LWR is about 0.945 rather than 1.0 - the truth
itself does not satisfy this identity, and our proxy returns exactly 1.0 there by construction. The two are
different targets.

This measures the three quantities on the transitions the scorer would keep, using only released observations
for the right-hand side:

    a   |dN_true - rhs_obs|   the structural floor - what a perfect answer still pays
    b   |dN_ours - dN_true|   our error against the truth, which is what we have been minimising
    c   |dN_ours - rhs_obs|   our error against what the metric actually asks for

If b is much larger than a, moving dN towards rhs_obs is worth doing whatever it costs in accuracy, because
rhs_obs is computable and dN_true is not.

    .venv/Scripts/python.exe analysis/t3_official_rhs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, RELEASE, load  # noqa: E402
from tfb.physics import fd_params, load_ramps, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = "t1_m3all"
DT = 5.0 / 60.0


def neighbour_maps(panel, link_ids):
    """For each link, the indices of its incoming and outgoing mainline links inside the panel."""
    topo = pd.read_csv(RELEASE / "corridors" / panel / "network" / "lwr_mainline_topology.csv", dtype=str)
    idx = {l: i for i, l in enumerate(map(str, link_ids))}
    inc, out = [[] for _ in link_ids], [[] for _ in link_ids]
    for row in topo.itertuples(index=False):
        li = idx.get(str(row.link_id))
        if li is None:
            continue
        for col, dst in (("incoming_link_ids", inc), ("outgoing_link_ids", out)):
            for v in str(getattr(row, col, "") or "").split(";"):
                j = idx.get(v.strip())
                if j is not None:
                    dst[li].append(j)
    return inc, out


def agg(q, groups, own):
    """Sum of q over each link's neighbour set; the link's own flow where the set is empty (score_task3:205)."""
    outp = np.empty_like(q)
    for i, g in enumerate(groups):
        outp[..., i] = q[..., g].sum(-1) if g else own[..., i]
    return outp


def main() -> None:
    rows = []
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm = fd_params(panel, z["link_ids"])
        L = prm["length"].astype(np.float64)
        ok = ramp_ok(panel, "train", z)[days]
        r = load_ramps(panel, "train", z)
        on_q, off_q = r["on_q"][days].astype(np.float64), r["off_q"][days].astype(np.float64)
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        vA, qA = p["pv"].astype(np.float64), p["pq"].astype(np.float64)

        k_true = tq / np.maximum(ts, 1.0)
        k_ours = np.where(tgt, qA / np.maximum(vA, 1.0), k_true)
        dN_true = (k_true[:, 1:] - k_true[:, :-1]) * L
        dN_ours = (k_ours[:, 1:] - k_ours[:, :-1]) * L

        inc, out = neighbour_maps(panel, z["link_ids"])
        # the right-hand side from RELEASED observations only: visible mainline flows plus the ramps
        vis = use & ~tgt
        q_obs = np.where(vis, tq, np.nan)
        q_fill = np.where(vis, tq, 0.0)
        have = vis.astype(np.float64)
        inflow = agg(q_fill, inc, q_fill)
        outflow = agg(q_fill, out, q_fill)
        n_in = agg(have, inc, have)
        n_out = agg(have, out, have)
        need_in = np.array([max(len(g), 1) for g in inc], float)
        need_out = np.array([max(len(g), 1) for g in out], float)
        full = (n_in >= need_in) & (n_out >= need_out) & vis          # every term of the rhs observed
        rhs = DT * (inflow + on_q - outflow - off_q)

        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
        touch = trans & (tgt[:, :-1] | tgt[:, 1:])
        base = trans & full[:, :-1]
        a = np.abs(dN_true - rhs[:, :-1])
        b = np.abs(dN_ours - dN_true)
        c = np.abs(dN_ours - rhs[:, :-1])
        for nm, m in (("all scored", base), ("target-touching", base & touch), ("both visible", base & ~touch)):
            if not m.any():
                continue
            rows.append(dict(panel=panel, set=nm, n=int(m.sum()),
                             rhs=float(np.abs(rhs[:, :-1])[m].mean()), dN=float(np.abs(dN_true)[m].mean()),
                             a=float(a[m].mean()), b=float(b[m].mean()), c=float(c[m].mean())))
        print(f"  {panel}: rhs observable on {float(base.sum()) / max(int(trans.sum()), 1) * 100:.1f}% "
              f"of scored transitions", flush=True)
        del z, q_obs

    d = pd.DataFrame(rows)
    print(f"\nmean per transition, averaged over panels\n")
    print(f"{'set':<17} {'n':>12} {'|rhs_obs|':>10} {'|dN_true|':>10} "
          f"{'a: true vs rhs':>15} {'b: ours vs true':>16} {'c: ours vs rhs':>15}")
    for nm in ("all scored", "target-touching", "both visible"):
        g = d[d.set == nm]
        if g.empty:
            continue
        print(f"{nm:<17} {int(g.n.sum()):>12,} {g.rhs.mean():>10.4f} {g.dN.mean():>10.4f} "
              f"{g.a.mean():>15.4f} {g.b.mean():>16.4f} {g.c.mean():>15.4f}")
    print("\nE_LWR-style ratios on the target-touching set (sum |.| / sum |rhs_obs|):")
    g = d[d.set == "target-touching"]
    print(f"   a perfect answer   {float((g.a * g.n).sum() / (g.rhs * g.n).sum()):.4f}")
    print(f"   our answer         {float((g.c * g.n).sum() / (g.rhs * g.n).sum()):.4f}")


if __name__ == "__main__":
    main()
