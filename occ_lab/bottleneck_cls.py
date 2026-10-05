"""Onset as a competition between candidate bottlenecks, learned with run-invariant candidate features.

The onset loss is concentrated in windows where the production blend picks the wrong bottleneck while the right one
sits among its next peaks (notes 44, 51, 73). Link-identity refinements do not survive the run shift (note 128), so
the classifier never sees a link id. Per window:

    candidates   the top 5 peaks of the production onset probability at T+30, >= 4 links apart; every link is
                 assigned to its nearest peak (its region)
    features     production evidence   peak height, log, rank, peak / max, region mass share, the four members at
                                       the peak (H, ad9b, p_PS, STID)
                 current state          last and minimum v/vf at the peak and +-2 links, q/cap last and its trend
                 structure              lanes, lane change downstream, on-ramp / off-ramp attached
                 same-run evidence      the queued share of the VISIBLE morning cells (06-12 h) at the peak (max over
                                       +-2 links) on all earlier days of the month and on the last earlier day -
                                       released data at or before T (today's measurement: next-day correlation +0.67)
                 time of day
    label        the candidate whose region holds the most true queued cells at T+30
    model        LightGBM on candidate rows, normalised within the window; fitted on one scored month and applied
                 to the other (as note 56's cross-month weights)
    decision     'reweight': p_i * (P_cls(b(i)) / P_base(b(i)))^alpha, then the ratio rule;
                 'pick'    : only the arg-max candidate's region keeps its probability

    .venv/Scripts/python.exe occ_lab/bottleneck_cls.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
K, SEP = 5, 4
MO = slice(6 * 12, 12 * 12)


def peaks(p):
    q = p.copy()
    out = []
    for _ in range(K):
        j = int(np.argmax(q))
        if q[j] <= 1e-4:
            break
        out.append(j)
        q[max(0, j - SEP + 1):j + SEP] = -1
    return np.array(out)


def last_obs(a, ok):
    """(H, S) values, (H, S) ok -> last observed value per link (nan if none)."""
    idx = np.where(ok[::-1], np.arange(ok.shape[0])[::-1][:, None], -1).max(0)
    out = np.full(a.shape[1], np.nan, np.float32)
    m = idx >= 0
    out[m] = a[idx[m], np.arange(a.shape[1])[m]]
    return out


def month_share(z, vf):
    """per day: queued and visible counts of the morning cells, for the same-run evidence features."""
    v = z["obs_speed"]
    okz = z["elig"] & np.isfinite(v)
    qz = okz & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    return qz[:, MO].sum(1).astype(np.float32), okz[:, MO].sum(1).astype(np.float32)


def cand_rows(p, mem, v, qf, ok, vf, cap, lanes, stat, qd, nd, d, t0):
    """Candidate peaks, regions, region masses and the feature rows of ONE window.
    p (S,) production onset probability; mem 4 x (S,) = H, ad9b, p_PS, STID; v, qf, ok (H, S) history."""
    S = len(p)
    cand = peaks(p)
    if len(cand) < 2:
        return None
    reg = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1)
    vv = np.where(ok, v / vf, np.nan)
    qq = np.where(ok, qf / cap, np.nan)
    vlast, qlast = last_obs(np.nan_to_num(vv, nan=0), ok), last_obs(np.nan_to_num(qq, nan=0), ok)
    with np.errstate(all="ignore"):
        vmin = np.nanmin(vv, 0)
        qslope = np.nanmean(qq[-3:], 0) - np.nanmean(qq[:3], 0)
        share_all = qd[:d].sum(0) / np.maximum(nd[:d].sum(0), 1) if d else np.full(S, np.nan)
        share_last = qd[d - 1] / np.maximum(nd[d - 1], 1) if d else np.full(S, np.nan)
    mass = np.bincount(reg, weights=p, minlength=len(cand))
    rows = []
    for r, j in enumerate(cand):
        nb = slice(max(0, j - 2), j + 3)
        with np.errstate(all="ignore"):
            rows.append([p[j], np.log(p[j] + 1e-6), r, p[j] / p.max(), mass[r] / mass.sum(),
                         mem[0][j], mem[1][j], mem[2][j], mem[3][j]] + ([mem[4][j]] if len(mem) > 4 else []) + [
                         vlast[j], np.nanmin(vmin[nb]) if np.isfinite(vmin[nb]).any() else np.nan,
                         qlast[j], qslope[j],
                         lanes[j], lanes[min(j + 1, S - 1)] - lanes[j], stat[3, j], stat[4, j],
                         np.nanmax(share_all[nb]) if d else np.nan, np.nanmax(share_last[nb]) if d else np.nan,
                         float(d), t0 / 12.0])
    return np.asarray(rows, np.float32), cand, reg, mass


def swap(p, cand, reg, mass, pc):
    """Where the classifier's top candidate differs from production's, swap the two peaks' heights by
    rescaling their regions; elsewhere p is untouched (note 129)."""
    cb, cc = int(np.argmax(mass)), int(np.argmax(pc))
    if cb == cc:
        return p
    q = p.copy()
    hb, hc = p[cand[cb]], p[cand[cc]]
    q[reg == cc] *= hb / max(hc, 1e-6)
    q[reg == cb] *= hc / max(hb, 1e-6)
    return np.clip(q, 0, 1)


def build(i, panel):
    ctx, stat, W = C.windows(i, panel)
    vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
    lanes = np.asarray(ctx.lanes, np.float32)
    S = len(vf)
    out = {}
    for s in C.SPLITS:
        on, ref, z = W[s]["on"], W[s]["ref"], W[s]["z"]
        if not len(on["d"]):
            continue
        ek = os.environ.get("ONSET_EARLY_K", "")                 # note 169: official-window condition, K days adapted
        ps = np.load(CACHE / (f"pseudo_h{ek}_supprec_{s}_{panel}.npz" if ek else f"pseudo_s20_{s}_{panel}.npz"))["on"].astype(np.float32)
        st = np.load(CACHE / f"onset_adapt_stid30_{s}_{panel}.npz")["prob"].astype(np.float32)
        if os.environ.get("ONSET_ST", "") == "film":         # note 165: the FiLM STID supervised by recovered onsets
            st = np.load(CACHE / f"onset_film_recov_{s}_{panel}.npz")["prob"].astype(np.float32)
        elif os.environ.get("ONSET_ST", "") == "filmhalf":   # half AP STID, half FiLM STID
            st = 0.5 * st + 0.5 * np.load(CACHE / f"onset_film_recov{os.environ.get('FILM_TAG', '')}_{s}_{panel}.npz")["prob"].astype(np.float32)
        Hq, ad = ref["on_H"], ref["on_ad9b"]
        if ek:
            ad = np.load(CACHE / f"pseudo_h{ek}_rec_{s}_{panel}.npz")["on"].astype(np.float32)
        wts = os.environ.get("ONSET_W", "")                   # note 169: "H,ad,ps,G,st" weights; G = w192 + generated
        if wts:
            wH, wA, wP, wG, wS = (float(x) for x in wts.split(","))
            g_tag, g_suf = ("ctrl", ("", "_rb")) if (panel == "D7_I405_N" or os.environ.get("ONSET_G", "") == "G0")                 else ("ct10d", ("_q1", "_q2"))
            if ek and os.environ.get("ONSET_G", "") == "G0":      # w192 ctrl under the official condition (K days)
                Gm = np.mean([np.load(CACHE / f"pseudo_h{ek}_ctrl_s{k}_{s}_{panel}.npz")["on"] for k in (20, 21)],
                             0).astype(np.float32)
            else:
                Gm = np.mean([np.load(CACHE / f"fam_leap_{g_tag}_s{k}{x}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)
                              for x in g_suf], 0).astype(np.float32)
            base = wH * Hq + wA * ad + wP * ps + wG * Gm + wS * st
        else:
            Gm = None
            base = 0.35 * Hq + 0.175 * ad + 0.175 * ps + 0.30 * st
        v = z["obs_speed"]; okz = z["elig"] & np.isfinite(v)
        qz = okz & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
        qd, nd = qz[:, MO].sum(1).astype(np.float32), okz[:, MO].sum(1).astype(np.float32)
        rows, labels, groups, meta = [], [], [], []
        for n in range(len(on["d"])):
            p = base[n]
            cand = peaks(p)
            if len(cand) < 2:
                continue
            reg = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1)
            y30 = on["fut"][n, -1] & on["efut"][n, -1]
            hits = np.bincount(reg[y30], minlength=len(cand))
            label = int(np.argmax(hits)) if hits.sum() else -1
            vv = np.where(on["ok"][n], on["v"][n] / vf, np.nan)
            qq = np.where(on["ok"][n], on["qf"][n] / cap, np.nan)
            vlast, qlast = last_obs(np.nan_to_num(vv, nan=0), on["ok"][n]), last_obs(np.nan_to_num(qq, nan=0), on["ok"][n])
            with np.errstate(all="ignore"):
                vmin = np.nanmin(vv, 0)
                qslope = np.nanmean(qq[-3:], 0) - np.nanmean(qq[:3], 0)
            d = int(on["d"][n])
            lo = 0
            if ek and os.environ.get("ONSET_SHARE_K", "") == "1":    # CBM study: only the K earlier days an
                lo = max(0, d - int(ek))                              # official window (day K) can see
            share_all = qd[lo:d].sum(0) / np.maximum(nd[lo:d].sum(0), 1) if d > lo else np.full(S, np.nan)
            share_last = qd[d - 1] / np.maximum(nd[d - 1], 1) if d > lo else np.full(S, np.nan)
            mass = np.bincount(reg, weights=p, minlength=len(cand))
            for r, j in enumerate(cand):
                nb = slice(max(0, j - 2), j + 3)
                rows.append([p[j], np.log(p[j] + 1e-6), r, p[j] / p.max(), mass[r] / mass.sum(),
                             Hq[n, j], ad[n, j], ps[n, j], st[n, j]] + ([Gm[n, j]] if Gm is not None else []) + [
                             vlast[j], np.nanmin(vmin[nb]) if np.isfinite(vmin[nb]).any() else np.nan,
                             qlast[j], qslope[j],
                             lanes[j], lanes[min(j + 1, S - 1)] - lanes[j], stat[3, j], stat[4, j],
                             np.nanmax(share_all[nb]) if d else np.nan, np.nanmax(share_last[nb]) if d else np.nan,
                             float(d - lo), on["t0"][n] / 12.0])
                labels.append(int(r == label))
            groups.append(len(cand))
            meta.append((n, cand, reg, mass))
        out[s] = dict(X=np.asarray(rows, np.float32), y=np.asarray(labels), g=np.asarray(groups), meta=meta,
                      base=base, on=on)
    return out


def main() -> None:
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        data[panel] = build(i, panel)
        print(f"  {panel}", flush=True)
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    fam, cnt = {}, {}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        X = np.concatenate([d[s_fit]["X"] for d in data.values() if s_fit in d])
        y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
        m = lgb.train(par, lgb.Dataset(X, y), 200)
        for panel, d in data.items():
            if s_app not in d:
                continue
            D = d[s_app]
            prob = m.predict(D["X"])
            on = D["on"]
            b_sc = C.onset_iou(D["base"], on)
            fam.setdefault((s_app, "production"), {})[panel] = b_sc.mean()
            o, off = {}, 0
            for (n, cand, reg, mass), gsize in zip(D["meta"], D["g"]):
                pc = prob[off:off + gsize]; off += gsize
                pc = pc / max(pc.sum(), 1e-9)
                pb = mass / max(mass.sum(), 1e-9)
                p = D["base"][n]
                for a in (0.5, 1.0):
                    o.setdefault(f"reweight a{a}", {})[n] = np.clip(p * (pc / np.maximum(pb, 1e-6))[reg] ** a, 0, 1)
                keep = reg == int(np.argmax(pc))
                o.setdefault("pick", {})[n] = np.where(keep, p, 0.0)
            for nm, rep in o.items():
                P = D["base"].copy()
                for n, q in rep.items():
                    P[n] = q
                sc = C.onset_iou(P, on)
                fam.setdefault((s_app, nm), {})[panel] = sc.mean()
                c = cnt.setdefault((s_app, nm), [0, 0])
                c[0] += int(((b_sc == 0) & (sc > 0)).sum()); c[1] += int(((b_sc > 0) & (sc == 0)).sum())
    names = ["production", "reweight a0.5", "reweight a1.0", "pick"]
    ref = {s: C.family_mean(fam[(s, "production")]) for s in C.SPLITS}
    print(f"\n{'onset decision':<16}" + "".join(f"{s:>22}" for s in C.SPLITS) + "   rescue / spoil")
    for nm in names:
        cells = "".join(f"{C.family_mean(fam[(s, nm)]):>12.4f} ({C.family_mean(fam[(s, nm)]) - ref[s]:+.4f})"
                        for s in C.SPLITS)
        rs = sum(cnt.get((s, nm), [0, 0])[0] for s in C.SPLITS)
        sp = sum(cnt.get((s, nm), [0, 0])[1] for s in C.SPLITS)
        print(f"{nm:<16}{cells}   {rs} / {sp}" if nm != "production" else f"{nm:<16}{cells}")


if __name__ == "__main__":
    main()
