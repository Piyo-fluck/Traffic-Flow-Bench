"""Experiment A: adapted Task 1 residual (idea 2 x idea 9b) as onset LightGBM features, paired.

See ../EXPERIMENTS.md section A. Key points:
  * CAUSAL Task 1 input: for each window only its own 12 history slots (T-60..T-5) are visible in the day image;
    every other slot is blank. (The cached work/sub_m3* day reconstructions see ~107 slots either side, i.e. the
    pseudo-window labels and the post-buffer - note 63/64 features leak.)
  * glob  = the three Task 1 CNNs as trained;  adapt = the same nets adapted day by day on the scored month's
    strictly-past days (masked layer, self-supervised hide-and-reconstruct, as t1 TTT / idea 9b).
  * train features always come from the global nets (on the train run the global nets ARE the run-adapted nets),
    so base / +t1res models are shared and glob vs adapt differ only in the scored-month features.
  * paired LightGBM (H config, prior dropout), --seeds x --bags, identical dropout draws for base and +t1res.

    python tta/run_A_t1res.py --name A_sf                 # scale-free 4 columns, visible residual
    python tta/run_A_t1res.py --name A_all --cols all
    python tta/run_A_t1res.py --name A_hid --resid hidden
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.pop("M3_V2", None)
os.environ.pop("M3_DNLOSS", None)
os.environ.pop("M3_KHEAD", None)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import t1_m3_cnn as m3  # noqa: E402
import t2_prior_robust  # noqa: E402
from t2_m1_models import hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import T, load  # noqa: E402
from tfb.features import profile_table, truth_valid  # noqa: E402

T1 = [("m3v1_seed0_w64.pt", 64), ("m3v1_seed1_w64.pt", 64), ("m3w128_seed0_w128.pt", 128)]
NAMES = ["t1_rv3", "t1_rv12", "t1_av3", "t1_av12", "t1_rq3", "t1_sd6", "t1_sig", "t1_mix",
         "t1_av3_rel", "t1_mix_rel", "t1_av3_rk", "t1_mix_rk"]
H, F = C.H, C.F
CHUNK = 16
DEV = C.DEV


# ------------------------------------------------------------------------------------------------ Task 1 side
def load_t1():
    nets = []
    for f, w in T1:
        n = m3.Net(10, w).to(DEV)
        n.load_state_dict(torch.load(C.ASSETS / "t1" / f, map_location=DEV))
        n.eval()
        nets.append(n)
    return nets


def tables(zt):
    alld = np.ones(len(zt["dates"]), bool)
    valid = truth_valid(zt)
    return (profile_table(zt["true_speed"], valid, zt["weekday"], alld),
            profile_table(zt["true_flow"], valid, zt["weekday"], alld))


def causal_input(lanes, fs, tv, tq, wd, t0, v, qf, ok, occ):
    """Day images with ONLY the window history visible. Returns X (N,10,T,S) and the history base (N,H,S) x2."""
    N, _, S = v.shape
    ov = np.full((N, T, S), np.nan, np.float32)
    oq, oo = ov.copy(), ov.copy()
    rows = np.arange(N)[:, None]
    hi = t0[:, None] + np.arange(-H, 0)
    ov[rows, hi] = np.where(ok, v, np.nan)
    oq[rows, hi] = np.where(ok, qf, np.nan)
    oo[rows, hi] = np.where(ok, occ, np.nan)
    zz = dict(lanes=lanes, free_speed=fs, obs_speed=ov, obs_flow=oq, obs_occ=oo)
    bv, bq = tv[wd], tq[wd]
    return m3.make_inputs(zz, bv, bq), bv[rows, hi], bq[rows, hi]


@torch.no_grad()
def recon_hist(nets, X, t0, mode, n_masks=4, seed=0):
    """(K, N, H, S) x2: the nets' speed / flow-per-lane ANOMALY outputs at the history slots."""
    x = torch.from_numpy(X.astype(np.float32)).to(DEV)
    N = len(x)
    ar = torch.arange(N, device=DEV)[:, None]
    hi = torch.from_numpy(t0[:, None] + np.arange(-H, 0)).to(DEV)
    outv, outq = [], []
    for net in nets:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(x).float()
        if mode == "hidden":
            gen = torch.Generator(device=DEV).manual_seed(seed)
            acc = torch.zeros_like(o)
            cnt = torch.zeros_like(o[:, :1])
            vis = x[:, 2] > 0.5
            for _ in range(n_masks):
                hide = vis & (torch.rand(vis.shape, device=DEV, generator=gen) < 0.2)
                xin = x.clone()
                xin[:, 0:4] = xin[:, 0:4] * (~hide).unsqueeze(1).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    oh = net(xin).float()
                acc += oh * hide.unsqueeze(1)
                cnt += hide.unsqueeze(1).float()
            o = torch.where(cnt > 0, acc / cnt.clamp(min=1), o)
        g = o[ar, :, hi]                                   # (N, H, 2, S)
        outv.append(g[:, :, 0].cpu().numpy())
        outq.append(g[:, :, 1].cpu().numpy())
    return np.stack(outv), np.stack(outq)


def t1_resid(nets, lanes, fs, tv, tq, wd, t0, v, qf, ok, occ, mode):
    """Predicted history (K,N,H,S) speed and flow, in km/h and veh/h, chunked."""
    pvs, pqs = [], []
    for a in range(0, len(t0), CHUNK):
        s = slice(a, a + CHUNK)
        X, bv, bq = causal_input(lanes, fs, tv, tq, wd[s], t0[s], v[s], qf[s], ok[s], occ[s])
        ov, oq = recon_hist(nets, X, t0[s], mode, seed=a)
        pvs.append(bv[None] + ov * m3.SCALE_V)
        pqs.append(bq[None] + oq * m3.SCALE_Q * lanes)
    if not pvs:
        K, S = len(nets), v.shape[-1]
        return np.zeros((K, 0, H, S), np.float32), np.zeros((K, 0, H, S), np.float32)
    return np.concatenate(pvs, 1), np.concatenate(pqs, 1)


def wrank(x):
    S = x.shape[1]
    r = np.argsort(np.argsort(np.where(np.isfinite(x), x, -np.inf), 1), 1) / max(S - 1, 1)
    return np.where(np.isfinite(x), r, 0.5).astype(np.float32)


def t1_feats(PV, PQ, v, qf, ok):
    """(N, S, 12) exactly as analysis/t2_onset_t1res_feats.py, from given reconstructions."""
    if PV.shape[1] == 0:
        return np.zeros((0, v.shape[-1], len(NAMES)), np.float32)
    sig = PV.std(0)
    rv, rq = v - PV.mean(0), qf - PQ.mean(0)
    with np.errstate(all="ignore"):
        m3_, m12, m6 = ok[:, -3:], ok, ok[:, -6:]
        rv3 = np.nanmean(np.where(m3_, rv[:, -3:], np.nan), 1)
        rv12 = np.nanmean(np.where(m12, rv, np.nan), 1)
        av3 = np.nanmean(np.where(m3_, np.abs(rv[:, -3:]), np.nan), 1)
        av12 = np.nanmean(np.where(m12, np.abs(rv), np.nan), 1)
        rq3 = np.nanmean(np.where(m3_, rq[:, -3:], np.nan), 1)
        sd6 = np.nanstd(np.where(m6, rv[:, -6:], np.nan), 1)
        sg3 = np.nanmean(np.where(m3_, sig[:, -3:], np.nan), 1)
        mix = np.abs(rv3) * sg3
        med_a = np.nanmedian(av3, 1, keepdims=True)
        med_m = np.nanmedian(mix, 1, keepdims=True)
        cols = [rv3, rv12, av3, av12, rq3, sd6, sg3, mix,
                av3 / np.maximum(med_a, 1e-3), mix / np.maximum(med_m, 1e-6), wrank(av3), wrank(mix)]
    return np.stack([np.nan_to_num(np.asarray(c, np.float32), nan=0.0) for c in cols], -1)


def adapt_t1(nets, Xday, E, pool, steps, lr, rng, bs=8, crop=64):
    """Self-supervised adaptation of every Task 1 net on full day images of the given past days."""
    if len(pool) == 0 or steps == 0:
        return
    for net in nets:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        net.train()
        for _ in range(steps):
            days = rng.choice(pool, size=bs)
            S = Xday.shape[-1]
            s0 = rng.integers(0, S - crop + 1) if S > crop else 0
            sl = slice(s0, s0 + crop)
            x = torch.from_numpy(np.ascontiguousarray(Xday[days][..., sl]).astype(np.float32)).to(DEV)
            e = torch.from_numpy(np.ascontiguousarray(E[days][..., sl])).to(DEV)
            hide = (x[:, 2] > 0.5) & e & (torch.rand(x[:, 2].shape, device=DEV) < 0.2)
            xin = x.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hide).unsqueeze(1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(xin).float()
            w = hide.float()
            n = w.sum().clamp(min=1)
            loss = (m3.LOSS_W[0] * ((o[:, 0] - x[:, 0]) ** 2 * w).sum()
                    + m3.LOSS_W[1] * ((o[:, 1] - x[:, 1]) ** 2 * w).sum()) / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cols", default="sf", choices=("sf", "all"), help="sf = the 4 scale-free columns")
    ap.add_argument("--resid", default="visible", choices=("visible", "hidden"))
    ap.add_argument("--steps", type=int, default=30, help="Task 1 adaptation steps per day")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--order", default="predict-then-adapt", choices=("predict-then-adapt", "adapt-then-predict"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bags", type=int, default=5)
    ap.add_argument("--max-win", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()
    out = C.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    cols = slice(8, 12) if args.cols == "sf" else slice(0, 12)
    rng = np.random.default_rng(29)
    t_all = time.time()
    nets0 = load_t1()
    state0 = [{k: v.detach().clone() for k, v in n.state_dict().items()} for n in nets0]

    P = []                       # per panel: everything needed later
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        zt = ctx.z
        tv, tq = tables(zt)
        lanes, fs = zt["lanes"], zt["free_speed"]
        # ---- train windows, global-net features
        d, t0 = ctx.eo
        if len(d) > args.max_win:
            s = np.sort(rng.choice(len(d), args.max_win, replace=False))
            d, t0 = d[s], t0[s]
        v, qf, ok = hist_from_truth(ctx, d, t0)
        occ = zt["true_occ"][d[:, None], t0[:, None] + np.arange(-H, 0)]
        for n, sd in zip(nets0, state0):
            n.load_state_dict(sd)
        PV, PQ = t1_resid(nets0, lanes, fs, tv, tq, zt["weekday"][d], t0, v, qf, ok, occ, args.resid)
        Etr = t1_feats(PV, PQ, v, qf, ok)
        rec = dict(panel=panel, i=i, ctx=ctx, tr=(d, t0, v, qf, ok), Etr=Etr, y=ctx.q[d, t0 + F].reshape(-1),
                   sc={})
        # ---- scored months
        for split in C.SPLITS:
            z = load(panel, split)
            blank = C.official_blank(panel, split, z)
            on = C.first_onsets(ctx.vf, z, blank=blank)
            c = np.load(C.CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            wd = z["weekday"][on["d"]].astype(int)
            hi = on["t0"][:, None] + np.arange(-H, 0)
            occ = z["obs_occ"][on["d"][:, None], hi]
            for n, sd in zip(nets0, state0):
                n.load_state_dict(sd)
            PVg, PQg = t1_resid(nets0, lanes, fs, tv, tq, wd, on["t0"], on["v"], on["qf"], on["ok"], occ, args.resid)
            Eg = t1_feats(PVg, PQg, on["v"], on["qf"], on["ok"])
            # adapted: day by day, strictly past days only
            Xday = m3.make_inputs(z, tv[z["weekday"]], tq[z["weekday"]])
            PVa = np.zeros_like(PVg)
            PQa = np.zeros_like(PQg)
            for day in sorted(set(on["d"].tolist())):
                if args.order == "adapt-then-predict":
                    adapt_t1(nets0, Xday, z["elig"], np.arange(day), args.steps, args.lr, rng)
                m = np.flatnonzero(on["d"] == day)
                a, b = t1_resid(nets0, lanes, fs, tv, tq, wd[m], on["t0"][m], on["v"][m], on["qf"][m], on["ok"][m],
                                occ[m], args.resid)
                PVa[:, m], PQa[:, m] = a, b
                if args.order == "predict-then-adapt":
                    adapt_t1(nets0, Xday, z["elig"], np.arange(day), args.steps, args.lr, rng)
            Ea = t1_feats(PVa, PQa, on["v"], on["qf"], on["ok"])
            del Xday
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            _, Xh = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key, wd)
            ad9b = C._cnn_cache(C.AD9B, split, panel, "on_p")
            np.savez_compressed(out / f"feats_{split}_{panel}.npz", Eg=Eg, Ea=Ea, on_d=on["d"])
            rec["sc"][split] = dict(on=on, Xh=Xh, Eg=Eg, Ea=Ea, H=c["on_p"].astype(np.float32), ad9b=ad9b)
            with np.errstate(all="ignore"):
                lv = np.nanmean(np.abs(on["v"] - PVg.mean(0))[on["ok"]]) if len(on["d"]) else np.nan
                la = np.nanmean(np.abs(on["v"] - PVa.mean(0))[on["ok"]]) if len(on["d"]) else np.nan
            print(f"[{panel} {split}] {len(on['d'])} onset windows; mean |dv| global {lv:.2f} -> adapted {la:.2f} "
                  f"({time.time() - t_all:.0f}s)", flush=True)
        P.append(rec)

    # ---- paired LightGBM
    names_on = None
    res = {}
    for sd in range(args.seeds):
        Xb = [[] for _ in range(args.bags)]
        for rec in P:
            ctx = rec["ctx"]
            d, t0, v, qf, ok = rec["tr"]
            t2_prior_robust.rng = np.random.default_rng(1000 * (sd + 1) + rec["i"])
            for b in range(args.bags):
                names_on, X = feats_on(ctx, False, v, qf, ok, t0, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d,
                                       ctx.z["weekday"][d])
                Xb[b].append(X.reshape(-1, X.shape[-1]))
        y = np.concatenate([rec["y"] for rec in P])
        Et = np.concatenate([rec["Etr"][..., cols].reshape(-1, cols.stop - cols.start) for rec in P])
        nm = names_on + NAMES[cols]
        mb, mt = [], []
        for b in range(args.bags):
            Xp = np.concatenate(Xb[b])
            prm = dict(PARAMS, num_leaves=args.leaves, seed=b + 100 * sd)
            mb.append(lgb.train(prm, lgb.Dataset(Xp, y, feature_name=names_on), args.rounds))
            mt.append(lgb.train(prm, lgb.Dataset(np.concatenate([Xp, Et], 1), y, feature_name=nm), args.rounds))
        del Xb
        imp = mt[0].feature_importance("gain")
        print(f"seed {sd}: T1 block share of gain {imp[-len(NAMES[cols]):].sum() / imp.sum():.3f} "
              f"({time.time() - t_all:.0f}s)", flush=True)
        for rec in P:
            for split, s in rec["sc"].items():
                on = s["on"]
                if not len(on["d"]):
                    continue
                Xh = s["Xh"]
                pr = lambda ms, X: np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in ms], 0).reshape(X.shape[:2])  # noqa: E731
                arms = {"base": pr(mb, Xh),
                        "glob": pr(mt, np.concatenate([Xh, s["Eg"][..., cols]], -1)),
                        "adapt": pr(mt, np.concatenate([Xh, s["Ea"][..., cols]], -1))}
                if sd == 0:
                    arms["REF cachedH"] = s["H"]
                for arm, p in arms.items():
                    for mode, pp in (("alone", p), ("T-blend", C.W_ON * p + (1 - C.W_ON) * s["ad9b"])):
                        iou = C.onset_iou(pp, on)
                        res.setdefault((split, mode, arm, sd if arm != "REF cachedH" else -1), {})[rec["panel"]] = iou
    # ---- report
    lines = []
    for split in C.SPLITS:
        for mode in ("alone", "T-blend"):
            lines.append(f"\n=== {split} | onset {mode} | (seed-mean; delta vs base of the SAME seed) ===")
            ref = res.get((split, mode, "REF cachedH", -1))
            if ref:
                lines.append(f"  REF cached H{'+ad9b (=T)' if mode == 'T-blend' else ''}: IoU "
                             f"{np.concatenate(list(ref.values())).mean():.4f}  fam "
                             f"{C.family_mean({k: v.mean() for k, v in ref.items()}):.4f}")
            for arm in ("base", "glob", "adapt"):
                fams, dfam, dwin = [], [], []
                for sd in range(args.seeds):
                    r = res[(split, mode, arm, sd)]
                    rb = res[(split, mode, "base", sd)]
                    fa = C.family_mean({k: v.mean() for k, v in r.items()})
                    fb = C.family_mean({k: v.mean() for k, v in rb.items()})
                    fams.append(fa)
                    dfam.append(fa - fb)
                    dwin.append(np.concatenate([r[k] - rb[k] for k in r]))
                pos = sum(x > 1e-9 for x in dfam)
                lines.append(f"  {arm:<6} fam {np.mean(fams):.4f}  dfam vs base {np.mean(dfam):+.4f} ({pos}/{args.seeds}+)"
                             f"  window delta {C.ci(np.mean(dwin, 0))}")
            da = [C.family_mean({k: v.mean() for k, v in res[(split, mode, 'adapt', sd)].items()})
                  - C.family_mean({k: v.mean() for k, v in res[(split, mode, 'glob', sd)].items()})
                  for sd in range(args.seeds)]
            lines.append(f"  adapt - glob: dfam {np.mean(da):+.4f} ({sum(x > 1e-9 for x in da)}/{args.seeds}+)")
    txt = "\n".join(lines)
    print(txt)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
