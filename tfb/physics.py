"""Local Task 3 scoring on dense arrays, following src/task3/score_task3.py.

S_FD is exact: it needs only the state (truth at non-target cells, the
submission at targets) and the fundamental-diagram parameters.

S_LWR needs the organizer boundary flows, which are never released. They are
projected onto the released observations, so the truth satisfies conservation
(up to a practical ceiling of about 0.96). The proxy therefore takes the
truth's own accumulation change as the right-hand side:

    E_proxy = sum |dN_pred - dN_true| / sum |dN_true|,  S_LWR_proxy = max(0, 1 - E_proxy)

which is computable on train, where the truth is released.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfb.data import CACHE, REGIMES, RELEASE, T


def fd_params(panel: str, link_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per-station FD parameters, resolved like score_task3.network_parameters()."""
    net = RELEASE / "corridors" / panel / "network"
    links = pd.read_csv(net / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str})
    have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in fd.columns]
    fdm = fd.groupby("link_id")[have].mean()
    idx = pd.Index(link_ids)
    length = links.length_km.reindex(idx).fillna(1.0).to_numpy(float)
    vf = fdm.free_speed_kmh.reindex(idx).fillna(links.free_speed_kmh.reindex(idx)).fillna(100.0).to_numpy(float)
    lanes = fdm.lanes.reindex(idx).fillna(links.lanes.reindex(idx)).fillna(1.0).clip(lower=1.0).to_numpy(float)
    cap = fdm.capacity_vph.reindex(idx).fillna(links.capacity_vph.reindex(idx)).fillna(pd.Series(1800.0 * lanes, index=idx)).to_numpy(float)
    kcrit = cap / np.maximum(vf, 1.0)
    kjam = (fdm.k_jam.reindex(idx).to_numpy(float) if "k_jam" in fdm else np.full(len(idx), np.nan))
    kjam = np.where(np.isfinite(kjam), kjam, 100.0 * lanes)
    kjam = np.maximum(kjam, kcrit * 1.05)
    return dict(length=length, vf=vf, lanes=lanes, cap=cap, kcrit=kcrit, kjam=kjam)


def load_ramps(panel: str, split: str, z: dict) -> dict[str, np.ndarray]:
    """Valid released ramp flows aggregated onto mainline stations, as score_task3.load_ramp_flows does.

    on_q/off_q (D,T,S): summed valid on/off-ramp flow attached to the station (0 where none).
    on_v/off_v (D,T,S): at least one valid ramp cell of that type. on_att/off_att (S,):
    the topology attaches a ramp of that type. Cached per panel and split.
    """
    path = CACHE / f"{panel}_{split}_ramps.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    net = RELEASE / "corridors" / panel / "network"
    amap = pd.read_csv(net / "ramp_attachment_map.csv", dtype=str)
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
    link_s = {l: i for i, l in enumerate(z["link_ids"])}
    S = len(link_s)
    on_att, off_att = np.zeros(S, bool), np.zeros(S, bool)
    for att, col in ((on_att, "on_ramp_link_ids"), (off_att, "off_ramp_link_ids")):
        for link in topo.loc[topo[col].fillna("").str.len() > 0, "link_id"]:
            if link in link_s:
                att[link_s[link]] = True
    ramp_s = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id.map(link_s)))
    day = {d: i for i, d in enumerate(z["dates"])}
    D = len(day)
    on_v, off_v = np.zeros((D, T, S), bool), np.zeros((D, T, S), bool)
    on_q, off_q = np.zeros((D, T, S), np.float32), np.zeros((D, T, S), np.float32)
    for f in sorted((RELEASE / "corridors" / panel / split / "ramp_states").glob("**/*.parquet")):
        x = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph",
                                        "pct_observed", "is_score_eligible"])
        x = x[x.is_score_eligible.astype(bool) & (x.pct_observed >= 75) & x.flow_vph.notna()]
        s = x.ramp_link_id.astype(str).map(ramp_s)
        x, s = x[s.notna()], s[s.notna()].astype(int).to_numpy()
        ts = x.timestamp.astype(str)
        d = ts.str.slice(0, 10).map(day)
        keep = d.notna().to_numpy()
        d = d[keep].astype(int).to_numpy()
        t = (ts.str.slice(11, 13).astype(int) * 12 + ts.str.slice(14, 16).astype(int) // 5).to_numpy()[keep]
        rt = x.ramp_type.astype(str).str.upper().to_numpy()[keep]
        q = x.flow_vph.to_numpy(np.float32)[keep]
        s = s[keep]
        on = np.isin(rt, ["OR", "ON"])
        off = np.isin(rt, ["FR", "OFF"])
        on_v[d[on], t[on], s[on]] = True
        off_v[d[off], t[off], s[off]] = True
        np.add.at(on_q, (d[on], t[on], s[on]), q[on])
        np.add.at(off_q, (d[off], t[off], s[off]), q[off])
    out = dict(on_q=on_q, off_q=off_q, on_v=on_v, off_v=off_v, on_att=on_att, off_att=off_att)
    np.savez(path, **out)
    return out


def ramp_ok(panel: str, split: str, z: dict) -> np.ndarray:
    """(D,T,S) True where a transition starting at that cell passes the Mode A ramp rule."""
    r = load_ramps(panel, split, z)
    return (r["on_v"] | ~r["on_att"]) & (r["off_v"] | ~r["off_att"])


def physics_scores(z: dict, pred_speed: np.ndarray, pred_flow: np.ndarray, prm: dict, ok: np.ndarray,
                   days: np.ndarray | None = None) -> dict:
    """S_FD (exact) and S_LWR proxy per regime, plus their macro means, over the given days."""
    tgt = z["target"]
    ts, tq = z["true_speed"], z["true_flow"]
    use = z["elig"] & np.isfinite(ts) & np.isfinite(tq)
    sv = np.where(tgt, pred_speed, ts)
    sq = np.where(tgt, pred_flow, tq)
    k_pred = sq / np.maximum(sv, 1.0)
    k_true = tq / np.maximum(ts, 1.0)
    L, lanes = prm["length"], prm["lanes"]
    # FD, per lane, on the submitted state.
    kl = k_pred / lanes
    q_fd = np.where(kl <= prm["kcrit"] / lanes, prm["vf"] * kl,
                    (prm["cap"] / lanes) / np.maximum((prm["kjam"] - prm["kcrit"]) / lanes, 1e-9)
                    * np.maximum(prm["kjam"] / lanes - kl, 0))
    fd_abs = np.abs(sq / lanes - q_fd)
    ql_abs = np.abs(sq / lanes)
    # Conservation transitions t -> t+1 within a day, both cells in the state, ramp rule at t.
    trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
    dn_pred = (k_pred[:, 1:] - k_pred[:, :-1]) * L
    dn_true = (k_true[:, 1:] - k_true[:, :-1]) * L
    res = np.abs(dn_pred - dn_true)
    rhs = np.abs(dn_true)
    touched = trans & (tgt[:, :-1] | tgt[:, 1:])
    out = {}
    sel_days = np.ones(len(z["dates"]), bool) if days is None else days
    for r in REGIMES:
        dm = (z["regime"] == r) & sel_days
        if not dm.any():
            out[r] = dict(S_FD=0.0, S_LWR=0.0, S_phys=0.0, touched=0.0)
            continue
        u, tr = use[dm], trans[dm]
        s_fd = max(0.0, 1.0 - fd_abs[dm][u].sum() / (ql_abs[dm][u].sum() + 1e-9))
        e = res[dm][tr].sum() / (rhs[dm][tr].sum() + 1e-9)
        s_lwr = max(0.0, 1.0 - min(1.0, e))
        out[r] = dict(S_FD=s_fd, S_LWR=s_lwr, S_phys=(s_fd + 2 * s_lwr) / 3, touched=float(touched[dm][tr].mean()))
    out["mean"] = {k: float(np.mean([out[r][k] for r in REGIMES])) for k in ("S_FD", "S_LWR", "S_phys", "touched")}
    return out


def fmt_phys(r: dict) -> str:
    return (" | ".join(f"{g}: FD={r[g]['S_FD']:.3f} LWR={r[g]['S_LWR']:.3f}" for g in REGIMES)
            + f" || S_FD={r['mean']['S_FD']:.4f} S_LWR~{r['mean']['S_LWR']:.4f} S_phys~{r['mean']['S_phys']:.4f}")
