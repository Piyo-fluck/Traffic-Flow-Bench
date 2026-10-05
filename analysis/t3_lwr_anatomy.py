"""Where does S_LWR lose, and what would it take to get it back?

S_phys = (S_FD + 2 S_LWR) / 3 per regime, averaged over regimes, and Task 3 carries 0.15 of the total, so one
point of S_LWR is worth 0.1 of the total score - two orders of magnitude more than the margins the Task 1 work
has been fighting over. The production model sits at S_FD about 0.98 and S_LWR about 0.65.

S_LWR = 1 - sum |dN_pred - dN_true| / sum |dN_true| over the transitions t -> t+1, where the submitted state
carries the TRUTH at non-target cells and our prediction at target cells. So with e = k_pred - k_true,

    the residual of a transition is |(e[t+1] - e[t]) L|

**Temporally correlated errors cancel; independent ones are amplified by sqrt(2).** Note 100 measured our
residuals to be nearly independent in time (+0.04), which is the worst case for this metric.

This decomposes the ratio by regime, by how many of the two cells are targets, and against three references:

    truth        the state with no prediction error at all (S_LWR = 1 by construction, a sanity check)
    level-only   our prediction with its error made CONSTANT within each (day, link) run of targets, i.e. the
                 same RMSE but perfectly correlated errors - the ceiling if only the correlation changed
    smoothed     our prediction smoothed in time over 3 slots, which lowers the error correlation the other way

    py analysis/t3_lwr_anatomy.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import REGIMES  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3A"


def lwr_parts(k_pred, k_true, tgt, trans, L):
    dn_p = (k_pred[:, 1:] - k_pred[:, :-1]) * L
    dn_t = (k_true[:, 1:] - k_true[:, :-1]) * L
    return np.abs(dn_p - dn_t), np.abs(dn_t), trans


def main() -> None:
    acc, by_kind, corr = {}, {}, [0.0, 0.0, 0.0, 0]
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm = fd_params(panel, z["link_ids"])
        ok = ramp_ok(panel, "train", z)[days]
        L, lanes = prm["length"], prm["lanes"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        reg = np.asarray(z["regime"], str)[days]
        sv = np.where(tgt, p["pv"], ts)
        sq = np.where(tgt, p["pq"], tq)
        k_true = tq / np.maximum(ts, 1.0)
        k_pred = sq / np.maximum(sv, 1.0)
        e = np.where(tgt, k_pred - k_true, 0.0)
        # level-only: replace the error by its mean over each run of consecutive target slots on the link
        e_lvl = np.zeros_like(e)
        run_id = np.cumsum(~tgt, axis=1)
        for d in range(e.shape[0]):
            flat = run_id[d].T.ravel() + 10_000 * np.repeat(np.arange(e.shape[2]), e.shape[1])
            vals = e[d].T.ravel()
            m = tgt[d].T.ravel()
            if not m.any():
                continue
            u, inv = np.unique(flat[m], return_inverse=True)
            s = np.bincount(inv, weights=vals[m])
            n = np.bincount(inv)
            rep = (s / n)[inv]
            out = np.zeros_like(vals)
            out[m] = rep
            e_lvl[d] = out.reshape(e.shape[2], e.shape[1]).T
        k_lvl = k_true + e_lvl
        k_sm = k_pred.copy()
        k_sm[:, 1:-1] = np.where(tgt[:, 1:-1], (k_pred[:, :-2] + k_pred[:, 1:-1] + k_pred[:, 2:]) / 3,
                                 k_pred[:, 1:-1])
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
        n_t = tgt[:, :-1].astype(np.int8) + tgt[:, 1:].astype(np.int8)
        for nm, kk in (("prediction", k_pred), ("level-only error", k_lvl), ("smoothed 3", k_sm)):
            res, rhs, tr = lwr_parts(kk, k_true, tgt, trans, L)
            for r in REGIMES:
                dm = (reg == r)[:, None, None] & tr
                a = acc.setdefault((nm, r), [0.0, 0.0])
                a[0] += float(res[dm].sum()); a[1] += float(rhs[dm].sum())
            for k in (0, 1, 2):
                dm = tr & (n_t == k)
                b = by_kind.setdefault((nm, k), [0.0, 0.0, 0])
                b[0] += float(res[dm].sum()); b[1] += float(rhs[dm].sum()); b[2] += int(dm.sum())
        m2 = trans & (n_t == 2)
        a, b = e[:, :-1][m2], e[:, 1:][m2]
        corr[0] += float((a * b).sum()); corr[1] += float((a ** 2).sum()); corr[2] += float((b ** 2).sum())
        corr[3] += int(m2.sum())
        print(f"  {panel}", flush=True)

    print(f"\n{'state':<18} " + "  ".join(f"{r:>7}" for r in REGIMES) + f"{'mean':>9}")
    for nm in ("prediction", "level-only error", "smoothed 3"):
        s = []
        for r in REGIMES:
            a = acc[(nm, r)]
            s.append(max(0.0, 1 - min(1.0, a[0] / max(a[1], 1e-9))))
        print(f"{nm:<18} " + "  ".join(f"{v:7.4f}" for v in s) + f"{np.mean(s):9.4f}")

    print(f"\n{'state':<18} {'transition':<16} {'count':>12} {'residual':>12} {'|dN true|':>12} {'ratio':>8}")
    for nm in ("prediction", "level-only error", "smoothed 3"):
        for k, lab in ((0, "no target"), (1, "one target"), (2, "both targets")):
            b = by_kind[(nm, k)]
            print(f"{nm:<18} {lab:<16} {b[2]:12,} {b[0]:12,.0f} {b[1]:12,.0f} "
                  f"{b[0] / max(b[1], 1e-9):8.3f}")
    print(f"\nerror correlation between the two cells of a both-target transition: "
          f"{corr[0] / max(np.sqrt(corr[1] * corr[2]), 1e-9):+.3f} ({corr[3]:,} transitions)")


if __name__ == "__main__":
    main()
