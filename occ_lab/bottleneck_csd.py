"""D of the 2026-09-27 survey: early-warning (critical slowing down) features for the onset bottleneck choice.

Near a tipping point a system recovers slowly from small perturbations, so variance and lag-1 autocorrelation rise
before the transition (Scheffer et al.; for traffic: arXiv 2401.09364, and Royal Society Open Science 2025 on
multivariate measures). The onset question here is WHICH candidate bottleneck breaks first at T+30 - the one
closest to its tipping point should show the strongest early-warning signature over the visible hour.

Per candidate peak (note 129's candidates) and its +-2 links, on the detrended (linear fit on observed slots)
history of v/vf, q/cap and the density proxy (q/cap) / (v/vf):
    level      variance, lag-1 autocorrelation, skewness (max over the 5 links)
    rise       variance over the last 30 min minus the first 30 min (CSD is a RISE, not a level)
    spatial    mean pairwise correlation of the 5 links' v/vf residuals, largest eigenvalue share of their covariance
The candidate classifier (LightGBM, cross-month) is fitted with note 129's 21 features and with them + these; the
decision is note 129's swap.

    .venv/Scripts/python.exe occ_lab/bottleneck_csd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402


def detrend(a, ok):
    """(H, L) series with observed mask -> residuals of a per-link linear fit (nan where unobserved)."""
    Hh = a.shape[0]
    t = np.arange(Hh, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    for l in range(a.shape[1]):
        m = ok[:, l] & np.isfinite(a[:, l])
        if m.sum() < 6:
            continue
        c = np.polyfit(t[m], a[m, l], 1)
        out[m, l] = a[m, l] - np.polyval(c, t[m])
    return out


def stats(r):
    """Residuals (H, L) -> max over links of variance, lag-1 AC, skewness, and the variance rise."""
    var, ac1, skw, rise = [], [], [], []
    for l in range(r.shape[1]):
        x = r[:, l]
        m = np.isfinite(x)
        if m.sum() < 6:
            continue
        xs = x[m]
        v = xs.var()
        var.append(v)
        pair = m[1:] & m[:-1]
        if pair.sum() >= 4 and v > 1e-12:
            a, b = x[1:][pair], x[:-1][pair]
            ac1.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0)
        skw.append(((xs - xs.mean()) ** 3).mean() / (xs.std() ** 3 + 1e-12))
        h = len(x) // 2
        f, s = x[:h][np.isfinite(x[:h])], x[h:][np.isfinite(x[h:])]
        if len(f) >= 3 and len(s) >= 3:
            rise.append(s.var() - f.var())
    g = lambda z: float(np.max(z)) if len(z) else np.nan  # noqa: E731
    return [g(var), g(ac1), g(skw), g(rise)]


def csd_features(v, qf, ok, vf, cap, j):
    S = v.shape[1]
    nb = slice(max(0, j - 2), min(S, j + 3))
    vr = np.where(ok, v / vf, np.nan)[:, nb]
    qc = np.where(ok, qf / cap, np.nan)[:, nb]
    kd = qc / np.maximum(vr, 0.05)
    okn = ok[:, nb]
    out = []
    rv = detrend(vr, okn)
    for a in (vr, qc, kd):
        out += stats(detrend(a, okn) if a is not vr else rv)
    cols = [rv[:, l] for l in range(rv.shape[1]) if np.isfinite(rv[:, l]).sum() >= 6]
    if len(cols) >= 2:
        R = np.stack(cols, 1)
        both = np.isfinite(R).all(1)
        if both.sum() >= 6:
            Rb = R[both]
            cc = np.corrcoef(Rb.T)
            iu = np.triu_indices(len(cols), 1)
            ev = np.linalg.eigvalsh(np.cov(Rb.T) + 1e-12 * np.eye(len(cols)))
            out += [float(np.nanmean(cc[iu])), float(ev[-1] / max(ev.sum(), 1e-12))]
            return out
    return out + [np.nan, np.nan]


def main() -> None:
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        for s, D in B.items():
            on = D["on"]
            extra = []
            for n, cand, reg, mass in D["meta"]:
                for j in cand:
                    extra.append(csd_features(on["v"][n], on["qf"][n], on["ok"][n], vf, cap, int(j)))
            D["Xc"] = np.asarray(extra, np.float32)
            assert len(D["Xc"]) == len(D["X"])
        data[panel] = B
        print(f"  {panel}", flush=True)
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    fam, acc, cnt = {}, {}, {}
    for feats in ("note 129 features", "+ early-warning"):
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            def X_of(D):
                return D["X"] if feats == "note 129 features" else np.concatenate([D["X"], D["Xc"]], 1)
            X = np.concatenate([X_of(d[s_fit]) for d in data.values() if s_fit in d])
            y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
            m = lgb.train(par, lgb.Dataset(X, y), 200)
            hit = tot = hb = 0
            for panel, d in data.items():
                if s_app not in d:
                    continue
                D = d[s_app]
                prob = m.predict(X_of(D))
                on = D["on"]
                b_sc = C.onset_iou(D["base"], on)
                fam.setdefault((s_app, "production"), {})[panel] = b_sc.mean()
                P2 = D["base"].copy()
                off = 0
                for (n, cand, reg, mass), gsize in zip(D["meta"], D["g"]):
                    pc = prob[off:off + gsize]
                    yy = D["y"][off:off + gsize]
                    off += gsize
                    if yy.any():
                        tot += 1
                        hit += int(np.argmax(pc) == np.argmax(yy))
                        hb += int(np.argmax(mass) == np.argmax(yy))
                    P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
                sc = C.onset_iou(P2, on)
                fam.setdefault((s_app, feats), {})[panel] = sc.mean()
                c = cnt.setdefault((s_app, feats), [0, 0])
                c[0] += int(((b_sc == 0) & (sc > 0)).sum()); c[1] += int(((b_sc > 0) & (sc == 0)).sum())
            acc[(s_app, feats)] = (hit / max(tot, 1), hb / max(tot, 1))
    print(f"\n{'top-1 bottleneck':<22}" + "".join(f"{s:>14}" for s in C.SPLITS))
    print(f"{'production':<22}" + "".join(f"{acc[(s, 'note 129 features')][1]:>14.3f}" for s in C.SPLITS))
    for feats in ("note 129 features", "+ early-warning"):
        print(f"{feats:<22}" + "".join(f"{acc[(s, feats)][0]:>14.3f}" for s in C.SPLITS))
    ref = {s: C.family_mean(fam[(s, "production")]) for s in C.SPLITS}
    print(f"\n{'onset IoU, swap':<22}" + "".join(f"{s:>22}" for s in C.SPLITS) + "   rescue / spoil")
    for nm in ("production", "note 129 features", "+ early-warning"):
        cells = "".join(f"{C.family_mean(fam[(s, nm)]):>12.4f} ({C.family_mean(fam[(s, nm)]) - ref[s]:+.4f})"
                        for s in C.SPLITS)
        rs = sum(cnt.get((s, nm), [0, 0])[0] for s in C.SPLITS); sp = sum(cnt.get((s, nm), [0, 0])[1] for s in C.SPLITS)
        print(f"{nm:<22}{cells}" + (f"   {rs} / {sp}" if nm != "production" else ""))


if __name__ == "__main__":
    main()
