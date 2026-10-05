"""Onset idea 1: yesterday's first bottleneck, inferred as the official setting allows, as a candidate feature.

Within a run the first morning bottleneck repeats from one day to the next 0.84-0.88 of the time when the two first
breakdowns are within 30 min of each other (train truth, 1,546 day pairs), 0.68 when they are 2 h+ apart. For an
official onset window on day d (days 1-5), day d-1's first breakdown is the official window of day d-1 itself: its
horizon and the next hour are blanked, so the location has to be INFERRED from what is released:
    yesterday's production onset probability (the model's own answer for that window, issued before the blank)
    x the queued links visible at the first slot after the blank (breakdown + 65 min), dilated by 2 links
    -> argmax = yesterday's inferred first bottleneck; the breakdown time is known exactly (origin + 30 min).
The lab reproduces this: the previous day's first onset event (lab pool) is used only through the production
probability of its window and the queue visible 65 min after it.

Per candidate peak of today's window (note 129): distance to yesterday's inferred bottleneck (links, capped 30),
near (<= 2 links), |today's - yesterday's breakdown time| (slots), near x close in time, yesterday available.
The candidate classifier (cross-month LightGBM) with and without these; note 129's swap decision.

    .venv/Scripts/python.exe occ_lab/bottleneck_memory.py
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

F = C.F
AFTER = 13                      # breakdown at t = origin + 6; the blank ends at origin + 18 -> first slot t + 13


def memory_features(D, z, vf):
    """(rows, 5) aligned with D["X"]; yesterday = the lab onset window on day d-1 (if any)."""
    on, base = D["on"], D["base"]
    ov = z["obs_speed"]
    okz = z["elig"] & np.isfinite(ov)
    qz = okz & (np.nan_to_num(ov, nan=1e9) <= 0.6 * vf)
    by_day = {int(dd): n for n, dd in enumerate(on["d"])}
    rows = []
    n_avail = 0
    for n, cand, reg, mass in D["meta"]:
        dd, t_today = int(on["d"][n]), int(on["t0"][n]) + F
        ny = by_day.get(dd - 1)
        jy, dt = None, np.nan
        if ny is not None:
            t_y = int(on["t0"][ny]) + F
            slot = min(t_y + AFTER, qz.shape[1] - 1)
            vis = qz[dd - 1, slot]
            if vis.any():
                vis2 = vis.copy()
                for r in (1, 2):
                    vis2[r:] |= vis[:-r]; vis2[:-r] |= vis[r:]
                score = base[ny] * vis2
                if score.max() > 0:
                    jy = int(np.argmax(score))
                    dt = abs(t_today - t_y)
                    n_avail += 1
        for j in cand:
            if jy is None:
                rows.append([np.nan, np.nan, np.nan, np.nan, 0.0])
            else:
                dist = min(abs(int(j) - jy), 30)
                near = float(dist <= 2)
                rows.append([dist, near, dt, near * float(dt <= 6), 1.0])
    return np.asarray(rows, np.float32), n_avail


def main() -> None:
    data, avail = {}, 0
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        for s, D in B.items():
            D["Xm"], na = memory_features(D, W[s]["z"], vf)
            avail += na
            assert len(D["Xm"]) == len(D["X"])
        data[panel] = B
        print(f"  {panel}", flush=True)
    print(f"windows with an inferred yesterday: {avail}")
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    fam, acc, cnt = {}, {}, {}
    for feats in ("note 129 features", "+ yesterday memory"):
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            def X_of(D):
                return D["X"] if feats == "note 129 features" else np.concatenate([D["X"], D["Xm"]], 1)
            X = np.concatenate([X_of(d[s_fit]) for d in data.values() if s_fit in d])
            y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
            m = lgb.train(par, lgb.Dataset(X, y), 200)
            hit = tot = hb = hit_y = tot_y = hb_y = 0
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
                    pc, yy, has_y = prob[off:off + gsize], D["y"][off:off + gsize], D["Xm"][off, 4] > 0
                    off += gsize
                    if yy.any():
                        ok_ = int(np.argmax(pc) == np.argmax(yy)); okb = int(np.argmax(mass) == np.argmax(yy))
                        tot += 1; hit += ok_; hb += okb
                        if has_y:
                            tot_y += 1; hit_y += ok_; hb_y += okb
                    P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
                sc = C.onset_iou(P2, on)
                fam.setdefault((s_app, feats), {})[panel] = sc.mean()
                c = cnt.setdefault((s_app, feats), [0, 0])
                c[0] += int(((b_sc == 0) & (sc > 0)).sum()); c[1] += int(((b_sc > 0) & (sc == 0)).sum())
            acc[(s_app, feats)] = (hit / max(tot, 1), hb / max(tot, 1), hit_y / max(tot_y, 1), hb_y / max(tot_y, 1), tot_y)
    print(f"\n{'top-1 bottleneck':<22}{'val all':>10}{'priv all':>10}{'val w/ yday':>14}{'priv w/ yday':>14}")
    a0 = acc[("validation", "note 129 features")]; a1 = acc[("private", "note 129 features")]
    print(f"{'production':<22}{a0[1]:>10.3f}{a1[1]:>10.3f}{a0[3]:>14.3f}{a1[3]:>14.3f}")
    for feats in ("note 129 features", "+ yesterday memory"):
        a0, a1 = acc[("validation", feats)], acc[("private", feats)]
        print(f"{feats:<22}{a0[0]:>10.3f}{a1[0]:>10.3f}{a0[2]:>14.3f}{a1[2]:>14.3f}   (n w/ yday {a0[4]} / {a1[4]})")
    ref = {s: C.family_mean(fam[(s, "production")]) for s in C.SPLITS}
    print(f"\n{'onset IoU, swap':<22}" + "".join(f"{s:>22}" for s in C.SPLITS) + "   rescue / spoil")
    for nm in ("production", "note 129 features", "+ yesterday memory"):
        cells = "".join(f"{C.family_mean(fam[(s, nm)]):>12.4f} ({C.family_mean(fam[(s, nm)]) - ref[s]:+.4f})"
                        for s in C.SPLITS)
        rs = sum(cnt.get((s, nm), [0, 0])[0] for s in C.SPLITS); sp = sum(cnt.get((s, nm), [0, 0])[1] for s in C.SPLITS)
        print(f"{nm:<22}{cells}" + (f"   {rs} / {sp}" if nm != "production" else ""))


if __name__ == "__main__":
    main()
