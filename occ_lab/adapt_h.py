"""Idea 1: adapt the onset LightGBM H to the scored run by boosting continuation on released pseudo onset windows.

H carries the largest onset weight (0.35) and the largest run-shift loss of any member (0.924 on the train tail ->
0.79 on the scored months, note 108): its features include the train run's recurrence tables (pq30, pon30, ...),
which describe where the TRAIN run queues. Every neural member was adapted to the scored run (+0.03..+0.05 alone,
notes 106 / 125); H never was.

Per panel and split, days in order. Before day d:
    pseudo onset windows   every slot t of the released layer, on days < d and on day d before its first origin,
                           where some link is queued at t and no link of the panel is queued in [t - 90 min, t)
                           (first_onsets' event definition, but every event of the day and any time of day);
                           origin t0 = t - 30 min, features = H's own onset_X on the released history,
                           label = queued at t, per link (eligible cells only)
    correction             LightGBM on those rows with init_score = logit(H), R rounds (small trees, lr 0.05),
                           refitted from scratch each day on all the run's past rows
    prediction             p = sigmoid(logit(H) + f(X)) for the day's official-like onset windows
The adapted H then replaces H in the production onset blend (0.35 H + 0.175 ad9b + 0.175 p_PS + 0.30 STID).

    .venv/Scripts/python.exe occ_lab/adapt_h.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from t2_m1_predict import onset_X, table_prior  # noqa: E402

CACHE = LAB / "cache"
PAST_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else None   # the official onset days 1-5 have 0-4 past days
H, F, T = C.H, C.F, 288
ROUNDS = (150, 300, 600, 1000)
PAR = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 100,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
       "num_threads": 8}
H_MODELS = [lgb.Booster(model_file=str(f)) for f in sorted((PROJ / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def h_feats(ctx, pri, z, d, t0, v, qf, ok):
    grp = (z["weekday"][d] >= 5).astype(int)
    _, X = onset_X(ctx, v, qf, ok, t0, grp, pri, np.stack([d, grp], 1))
    return X                                                             # (N, S, K)


def h_prob(X):
    return np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in H_MODELS], 0).reshape(X.shape[:2])


def pseudo_onsets(q, d, t_lim):
    """Event slots t (queued at t, nothing queued in [t - H - F, t)) of day d with t < t_lim."""
    qa = q[d].any(1)
    out = []
    for t in range(H + F + 1, min(t_lim, T)):
        if qa[t] and not qa[t - H - F:t].any():
            out.append(t)
    return out


def day_rows(ctx, pri, z, pool, d, t_lim):
    ov, oq, ok, q = pool
    ts = pseudo_onsets(q, d, t_lim)
    if not ts:
        return None
    t0 = np.array(ts) - F
    dd = np.full(len(t0), d)
    hi = t0[:, None] + np.arange(-H, 0)
    X = h_feats(ctx, pri, z, dd, t0, ov[dd[:, None], hi], oq[dd[:, None], hi], ok[dd[:, None], hi])
    y, m = q[dd, t0 + F], ok[dd, t0 + F]
    return X[m], y[m].astype(np.float32)


def main() -> None:
    t_start = time.time()
    fam, cnt = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel, need_train=True)
        pri = table_prior(ctx)
        for s in C.SPLITS:
            z, on, ref = W[s]["z"], W[s]["on"], W[s]["ref"]
            if not len(on["d"]):
                continue
            ov, oq = z["obs_speed"], z["obs_flow"]
            ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
            pool = (ov, oq, ok, ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf))
            Xw = h_feats(ctx, pri, z, on["d"], on["t0"], on["v"], on["qf"], on["ok"])
            pH = h_prob(Xw).astype(np.float32)
            assert np.abs(pH - ref["on_H"]).max() < 1e-4, "H recomputation differs from the cache"
            P = {f"Hadapt r{r}": pH.copy() for r in ROUNDS}
            past_X, past_y, past_day, done = [], [], [], 0
            n_rows = []
            for d in np.unique(on["d"]):
                d = int(d)
                while done < d:
                    r = day_rows(ctx, pri, z, pool, done, T)
                    if r is not None:
                        past_X.append(r[0]); past_y.append(r[1]); past_day.append(done)
                    done += 1
                first = int(on["t0"][on["d"] == d].min())
                today = day_rows(ctx, pri, z, pool, d, first)
                keep = [k for k in range(len(past_X)) if PAST_DAYS is None or past_day[k] >= d - PAST_DAYS]
                Xs = [past_X[k] for k in keep] + ([today[0]] if today is not None else [])
                ys = [past_y[k] for k in keep] + ([today[1]] if today is not None else [])
                idx = np.flatnonzero(on["d"] == d)
                if not Xs:
                    continue
                Xa, ya = np.concatenate(Xs), np.concatenate(ys)
                n_rows.append((len(ya), int(ya.sum())))
                if ya.sum() < 5:
                    continue
                init = logit(h_prob(Xa[:, None, :])[:, 0])
                booster = lgb.train(PAR, lgb.Dataset(Xa, ya, init_score=init), max(ROUNDS))
                Xd = Xw[idx].reshape(-1, Xw.shape[-1])
                base = logit(pH[idx]).reshape(-1)
                for r in ROUNDS:
                    corr = booster.predict(Xd, num_iteration=r, raw_score=True)
                    P[f"Hadapt r{r}"][idx] = (1 / (1 + np.exp(-(base + corr)))).reshape(len(idx), -1)
            ps = np.load(CACHE / f"pseudo_s20_{s}_{panel}.npz")["on"].astype(np.float32)
            st = np.load(CACHE / f"onset_adapt_stid30_{s}_{panel}.npz")["prob"].astype(np.float32)
            rest = 0.175 * ref["on_ad9b"] + 0.175 * ps + 0.30 * st
            b_sc = C.onset_iou(0.35 * pH + rest, on)
            early = on["d"] <= 4                               # the official onset windows sit on days 1-5
            for sub, msk in (("", np.ones(len(b_sc), bool)), (" [days 1-5]", early)):
                if not msk.any():
                    continue
                fam.setdefault((s, "production" + sub), {})[panel] = b_sc[msk].mean()
                fam.setdefault((s, "H alone" + sub), {})[panel] = C.onset_iou(pH, on)[msk].mean()
                for nm, pa in P.items():
                    fam.setdefault((s, f"{nm} alone" + sub), {})[panel] = C.onset_iou(pa, on)[msk].mean()
                    sc = C.onset_iou(0.35 * pa + rest, on)
                    fam.setdefault((s, f"{nm} in the blend" + sub), {})[panel] = sc[msk].mean()
                    c = cnt.setdefault((s, nm + sub), [0, 0])
                    c[0] += int(((b_sc == 0) & (sc > 0) & msk).sum()); c[1] += int(((b_sc > 0) & (sc == 0) & msk).sum())
                np.savez(CACHE / f"onset_adapt_Hadapt{nm.split()[-1]}_k{PAST_DAYS}_{s}_{panel}.npz", prob=pa)
            rows = n_rows[-1] if n_rows else (0, 0)
            print(f"  {panel} {s}: rows {rows[0]:,} (pos {rows[1]:,}) at the last day ({time.time() - t_start:.0f}s)",
                  flush=True)
    for sub in ("", " [days 1-5]"):
        names = ["production", "H alone"] + [f"Hadapt r{r} alone" for r in ROUNDS] + \
            [f"Hadapt r{r} in the blend" for r in ROUNDS]
        ref = {s: C.family_mean(fam[(s, "production" + sub)]) for s in C.SPLITS}
        print(f"\n{'onset family IoU' + sub:<40}{'validation':>20}{'private':>20}   rescue / spoil (val, priv)")
        for nm in names:
            k = nm + sub
            cells = "".join(f"{C.family_mean(fam[(s, k)]):>10.4f} ({C.family_mean(fam[(s, k)]) - ref[s]:+.4f})"
                            for s in C.SPLITS)
            kk = k.replace(" in the blend", "")
            rs = "   " + "  ".join(f"{cnt[(s, kk)][0]}/{cnt[(s, kk)][1]}" for s in C.SPLITS) if "blend" in nm else ""
            print(f"{k:<40}{cells}{rs}")


if __name__ == "__main__":
    main()
