"""Note 169: calibrate per-head breakdown factors so each head is active on the same share of days as in the truth
(train only). Start: the truth-only rule (ctm_rule0_<panel>.npz). Heads: sensored h with h queued (< 0.6 vf) and
h+1 not queued (> 0.6 vf) on >= 2 slots of 05-21 h on >= 10 % of train days, plus the downstream end (queued) when
the corridor has a boundary capacity. Each round replays half the train days (observed demand, no draws) and moves
ap_h <- ap_h * exp(0.5 * (share_replay - share_truth) / max(share_truth, 0.1)) (over-active -> breaks later),
clipped to [0.85, 1.3]; the end uses out_ap. 5 rounds, best round by sum |share gap|. Also prints the onset-site
overlap and, on D7_I405_N, how often z349 is queued at T+30 of end (z351) onsets.
Saves work/ctm_rule_share_<panel>.npz (heads, caps, sub, ap_cells, ap, out_ap)."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from onset_sites_util import day_sites
from tfb.data import load
warnings.simplefilter("ignore")
A, B = 60, 252


def shares(V, vfo, heads, n):
    r = V / vfo
    out = {}
    for h in heads:
        if h < n - 1:
            act = (r[:, A:B, h] <= 0.6) & (r[:, A:B, h + 1] > 0.6)
        else:
            act = r[:, A:B, h] <= 0.6
        out[h] = float(((act.sum(1)) >= 2).mean())
    return out


def site_stats(Vs, z, days, vf, rank, o):
    T, R, co_t, co_r = [], [], [], []
    for i, d in enumerate(days):
        el = z["elig"][d]
        for tag, V, L, co in (("t", z["true_speed"][d], T, co_t), ("r", Vs[i], R, co_r)):
            for t, s in day_sites(V, el & np.isfinite(V), vf, rank):
                L.append(s)
                if s == o[-1] and len(o) > 2:                          # end onset: is the link two up queued too?
                    co.append(bool(np.nan_to_num(V[t + 6, o[-3]], nan=1e9) <= 0.6 * vf[o[-3]]))
    S = len(vf); ht = np.bincount(T, minlength=S) / max(len(T), 1); hr = np.bincount(R, minlength=S) / max(len(R), 1)
    return np.minimum(ht, hr).sum(), (np.mean(co_t) if co_t else np.nan), (np.mean(co_r) if co_r else np.nan)


def main():
    for panel in sys.argv[1:]:
        z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
        g = CTMGen(panel, z); g.fit_profiles(z)
        ch = dict(np.load(ROOT / "work" / os.environ.get("RULE_OUT", "ctm_rule0_{}.npz").format(panel)))
        g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
        g.ap[ch["ap_cells"]] = ch["ap"]; g.out_ap = float(ch["out_ap"])
        o = g.order; vfo = vf[o]; n = g.n; rank = np.full(len(vf), -1); rank[o] = np.arange(n)
        days = np.arange(0, len(z["dates"]), 2)
        Vt = z["true_speed"][days][:, :, o]
        cand = list(range(n - 1)) + ([n - 1] if g.out_cap < 1e8 else [])
        st = shares(Vt, vfo, cand, n)
        heads = [h for h in cand if st[h] >= 0.10]
        hc = getattr(g, "head_cell", {})
        cell = {h: (hc.get(h, g.sens_cell[h] + 1) if h < n - 1 else None) for h in heads}
        best = None
        for it in range(6):
            Vr = np.empty((len(days), 288, len(vf)))
            for i, d in enumerate(days):
                up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
                Vr[i] = g.simulate(up, ent)[0]
            sr = shares(Vr[:, :, o], vfo, heads, n)
            gap = sum(abs(sr[h] - st[h]) for h in heads)
            ov, co_t, co_r = site_stats(Vr, z, days, vf, rank, o)
            print(f"  {panel} round {it}: sum |share gap| {gap:.2f}, site overlap {ov:.2f}, end-onset co-queue truth {co_t:.2f} replay {co_r:.2f} | "
                  + " ".join(f"{h}:{st[h]:.2f}/{sr[h]:.2f}" for h in heads), flush=True)
            state = (g.ap.copy(), g.out_ap)
            if best is None or gap < best[0]:
                best = (gap, state, it)
            if it == 5:
                break
            for h in heads:
                f = np.exp(0.5 * (sr[h] - st[h]) / max(st[h], 0.1))
                if cell[h] is None:
                    g.out_ap = float(np.clip(g.out_ap * f, 0.85, 1.3))
                else:
                    g.ap[cell[h]] = float(np.clip(g.ap[cell[h]] * f, 0.85, 1.3))
        ap, out_ap = best[1]
        cells = np.flatnonzero(np.abs(ap - 1.0) > 1e-9)
        print(f"{panel}: best round {best[2]} (gap {best[0]:.2f}); ap set on {len(cells)} cells; out_ap {out_ap:.3f}", flush=True)
        np.savez(ROOT / "work" / f"ctm_rule_share_{panel}.npz", heads=ch["heads"], caps=ch["caps"], sub=ch["sub"],
                 ap_cells=cells, ap=ap[cells], out_ap=out_ap)


if __name__ == "__main__":
    main()
