"""Note 169: identify the storage length between adjacent sensors from queue-tail propagation (train truth).

For an adjacent sensor pair (a upstream, b downstream, travel order): events where b turns congested while a is
free and a turns congested within 60 min after (a queue growing upstream). Crossing times of v = 0.6 vf are
interpolated linearly between 5-min slots. The tail speed u = (q_up - q_queue) / (k_up - k_queue) (Rankine-
Hugoniot; k = q / v, exact on the released FD) with the upstream state = a just before the tail arrives and the
queue state = b after it turned. Distance between the two sensor mid-points = |u| x (t_a - t_b). Per gap: median
over events, count, and the current assumption (L_a + L_b) / 2 + n_unsensored x median length.

    .venv/Scripts/python.exe analysis/dlsim_gap_length.py [panels]
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np  # noqa: E402

from dlsim_gen2 import Gen, corridor  # noqa: E402
from tfb.data import load  # noqa: E402


def crossing(v, thr, t):
    """fractional slot in (t-1, t] where v crosses thr downward (v[t-1] > thr >= v[t])"""
    a, b = v[t - 1], v[t]
    return t - 1 + (a - thr) / max(a - b, 1e-6)


def estimate(panel, z):
    g = Gen(panel, z)
    order, spos, *_ = corridor(panel, z)
    vf = g.vf
    v, q = z["true_speed"][:, :, order], z["true_flow"][:, :, order]
    thr = 0.6 * vf
    D_, T_, n = v.shape
    est = {k: [] for k in range(1, n)}
    for d in range(D_):
        vv, qq = v[d], q[d]
        ok = np.isfinite(vv) & np.isfinite(qq)
        for b in range(1, n):
            a = b - 1
            for ta in range(2, T_):
                if not (ok[ta - 1, a] and ok[ta, a] and vv[ta - 1, a] > thr[a] >= vv[ta, a]):
                    continue
                tb = None
                for s in range(ta, max(1, ta - 12), -1):              # b congested throughout [tb, ta]
                    if not (ok[s, b] and vv[s, b] <= thr[b]):
                        break
                    if ok[s - 1, b] and vv[s - 1, b] > thr[b]:
                        tb = s; break
                if tb is None:
                    continue
                tb_f, ta_f = crossing(vv[:, b], thr[b], tb), crossing(vv[:, a], thr[a], ta)
                if ta_f <= tb_f:
                    continue
                q_up, k_up = qq[ta - 1, a], qq[ta - 1, a] / max(vv[ta - 1, a], 1)
                qs, vs = qq[tb:ta + 1, b], vv[tb:ta + 1, b]
                q_q, k_q = np.nanmean(qs), np.nanmean(qs / np.maximum(vs, 1))
                if k_q - k_up < 5:
                    continue
                u = (q_up - q_q) / (k_up - k_q)
                if u >= 0:
                    continue
                est[b].append(abs(u) * (ta_f - tb_f) * 5 / 60)
    med_len = float(np.median(g.len))
    rows = []
    for b in range(1, n):
        cur = 0.5 * (g.len[b - 1] + g.len[b]) + (spos[b] - spos[b - 1] - 1) * med_len
        e = np.array(est[b])
        rows.append((b, len(e), np.median(e) if len(e) else np.nan, np.std(e) if len(e) > 1 else np.nan, cur,
                     spos[b] - spos[b - 1] - 1))
    return rows, med_len


def main():
    warnings.simplefilter("ignore")
    for panel in sys.argv[1:] or ["D12_I5_S"]:
        z = load(panel, "train")
        rows, ml = estimate(panel, z)
        r = np.array(rows, float)
        ok = r[:, 1] >= 5
        print(f"{panel}: gaps {len(r)}, with >=5 events {ok.sum()}, median sensored length {ml:.3f} km")
        print(f"  estimated / assumed distance (median over gaps with >=5 events): "
              f"{np.nanmedian(r[ok, 2] / r[ok, 4]):.2f}; corr(estimate, n_unsensored) "
              f"{np.corrcoef(r[ok, 2], r[ok, 5])[0, 1]:+.2f}; per-event sd / median {np.nanmedian(r[ok, 3] / r[ok, 2]):.2f}")
        for b, n_, m, sd, cur, nv in rows:
            if n_ >= 5:
                print(f"    gap {int(b):3d}: events {int(n_):4d}  est {m:.2f} km (sd {sd:.2f})  assumed {cur:.2f} km  unsensored {int(nv)}")


if __name__ == "__main__":
    main()
