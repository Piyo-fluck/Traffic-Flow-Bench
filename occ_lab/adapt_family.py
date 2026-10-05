"""Causal pseudo-supervised adaptation for any ongoing family, on any window set (notes 106, 125).

AU showed the transformer family adapted to the scored run moves the LB (+0.00178). This generalises the recipe
to every frozen member and to the three window sets the stack needs:

    families  tfm (T2Tfm, CNN channels)          cnn3 (T2Net w64, CNN channels, no rec head)
              occ_cnn (T2Net, + occupancy)       occ_tfm (T2Tfm, + occupancy)
              stid (STID: + occupancy, link id, origin slot, weekday)
    modes     lab       the official-like ongoing pool of both scored months (occ_lab cache, for evaluation)
              oof       the 2031-01/02 ongoing windows of the TRAIN run, exactly cache_oof.py's windows and seed,
                        adapted along the same causal trajectory (for refitting the residual corrector)
              official  the official windows (probabilities for the submission)

Per day, before predicting it: 30 steps of the supervised Task 2 loss (BCE + 0.5 Lovasz) on real ongoing windows
of the same run built from the released layer, whose horizon ends before the day's first origin (earlier days,
and the same day before that origin); weights carried forward to the next day. Nothing at or after a window's
origin enters its prediction.

    python occ_lab/adapt_family.py --family cnn3 --ckpt work/models/t2_cnn_lov_s1.pt --tag s1 --mode lab
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(PROJ / "analysis"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import pseudo_adapt as PA  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID  # noqa: E402

C = PA.C
cnn = C.cnn
H, F = C.H, C.F
CACHE = LAB / "cache"
PROFS = np.load(PROJ / "work" / "models_remote" / "occ_profiles.npz")
OCC = {"occ_cnn", "occ_tfm", "stid", "stidlh"}
CUR_A = {}          # d2: the current panel's corridor adjacency (analysis/t2_d2stgnn.corridor_adjacency)


def link_offsets():
    off, o = {}, 0
    for panel in C.T2_PANELS:
        off[panel] = o
        o += len(C.load(panel, "validation")["link_ids"])
    return off


def load_model(family, ckpt):
    if family == "d2":
        from t2_d2stgnn import D2STGNN
        d = torch.load(ckpt, map_location=C.DEV)
        net = D2STGNN(d["cin"], d["n_links"], d["d"], d["layers"]).to(C.DEV)
        net.load_state_dict(d["state"])
        return net.eval()
    if family == "stidlh":
        from train_stid import STIDLH
        d = torch.load(ckpt, map_location=C.DEV)
        net = STIDLH(d["cin"], d["n_links"], d["n_slots"], rank=d["lh_rank"]).to(C.DEV)
        net.load_state_dict(d["state"])
        return net.eval()
    if family == "stid":
        d = torch.load(ckpt, map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        return net.eval()
    sd = torch.load(ckpt, map_location=C.DEV)
    if family in ("tfm", "occ_tfm"):
        net = cnn.T2Tfm(sd["gru.weight_ih_l0"].shape[1], sd["gru.weight_hh_l0"].shape[1])
    else:
        net = cnn.T2Net(sd["inp.weight"].shape[1], sd["inp.weight"].shape[0])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("rec.") for k in missing), (missing, unexpected)
    return net.to(C.DEV).eval()


def build_x(family, ctx, stat, v, qf, ok, occ, t0, wk, prof):
    x = cnn.make_x(ctx, stat, v, qf, ok, t0, wk)
    if family in OCC:
        x = np.concatenate([x, occ_channels(occ, ok, prof)], 1)
    return x


def forward(family, net, x, lid, t0, dow):
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(C.DEV, dt)  # noqa: E731
    if family == "d2":
        return net(tt(x, torch.float32), tt(lid, torch.long), CUR_A["A"])
    if family in ("stid", "stidlh"):
        L = np.broadcast_to(lid, (len(x), len(lid))).copy()
        return net(tt(x, torch.float32), tt(L, torch.long), tt(t0, torch.long), tt(dow, torch.long))
    return net(tt(x, torch.float32))[:, :F]


@torch.no_grad()
def predict(family, net, x, lid, t0, dow, bs=64):
    net.eval()
    out = [torch.sigmoid(forward(family, net, x[a:a + bs], lid, t0[a:a + bs], dow[a:a + bs]).float()).cpu().numpy()
           for a in range(0, len(x), bs)]
    return np.concatenate(out) if out else np.zeros((0, F, len(lid)), np.float32)


def sup_batch(family, ctx, stat, z, lab, days, cuts, bs, rng, prof):
    ov, oq, ok_all, q = lab
    cand = []
    for _ in range(bs * 8):
        d = int(days[rng.integers(len(days))])
        lim = min(cuts.get(d, 10 ** 6), ov.shape[1] - 1)
        if lim <= H + F + 1:
            continue
        t0 = int(rng.integers(H + 1, lim - F))
        if q[d, t0 - H:t0].sum() >= 2 and q[d, t0 + 1:t0 + 1 + F].any():
            cand.append((d, t0))
        if len(cand) >= bs:
            break
    if not cand:
        return None
    d = np.array([c[0] for c in cand]); t0 = np.array([c[1] for c in cand])
    hi, fi = t0[:, None] + np.arange(-H, 0), t0[:, None] + np.arange(1, F + 1)
    occ = z["obs_occ"][d[:, None], hi] if family in OCC else None
    x = build_x(family, ctx, stat, ov[d[:, None], hi], oq[d[:, None], hi], ok_all[d[:, None], hi], occ, t0,
                z["weekday"][d] >= 5, prof)
    return (x, q[d[:, None], fi].astype(np.float32), ok_all[d[:, None], fi].astype(np.float32), t0.astype(np.int64),
            z["weekday"][d].astype(np.int64))


ADAPT_PARAMS = "all"


def adapt(family, net, lid, batches, steps, lr):
    if ADAPT_PARAMS == "lh":          # only the low-rank link x horizon term of STIDLH, the run-specific part
        for q in net.parameters():
            q.requires_grad_(False)
        params = [net.lh_a.weight, net.lh_b]
        for q in params:
            q.requires_grad_(True)
    else:
        params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, y, m, t0, dow = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(family, net, x, lid, t0, dow)
        logit = out.float()
        yt, mt = torch.from_numpy(y).to(C.DEV), torch.from_numpy(m).to(C.DEV)
        loss = cnn.loss_fn(logit, yt, mt) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yt, mt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    net.eval()


def released_labels(ctx, z):
    ov, oq = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
    return ov, oq, ok, ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)


def windows_for(mode, i, panel, family, max_oof):
    """Yields (split, ctx, stat, z, X, d, t0, dow, firsts, extra) per split; extra carries what the caller saves."""
    prof = PROFS[panel]
    if mode == "lab":
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            if not len(og["d"]):
                continue
            occ = np.stack([z["obs_occ"][a, b - H:b] for a, b in zip(og["d"], og["t0"])]) if family in OCC else None
            X = build_x(family, ctx, stat, og["v"], og["qf"], og["ok"], occ, og["t0"], z["weekday"][og["d"]] >= 5, prof)
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in np.unique(og["d"])}
            yield s, ctx, stat, z, X, og["d"], og["t0"], firsts, {"og": og}
    elif mode == "oof":
        from t2_official_like_eval import ongoing_official
        from tfb.t1eval import FIT_END
        ctx = C.Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        og, _ = ongoing_official(ctx, z, None)
        rng_w = np.random.default_rng(0)                      # cache_oof.py's seed, so the windows match
        keep = np.flatnonzero(z["dates"][og["d"]] >= FIT_END)
        if len(keep) > max_oof:
            keep = np.sort(rng_w.choice(keep, max_oof, replace=False))
        og = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(og["d"]) else v) for k, v in og.items()}
        occ = np.stack([z["obs_occ"][a, b - H:b] for a, b in zip(og["d"], og["t0"])]) if family in OCC else None
        X = build_x(family, ctx, stat, og["v"], og["qf"], og["ok"], occ, og["t0"], z["weekday"][og["d"]] >= 5, prof)
        firsts = {int(d): int(og["t0"][og["d"] == d].min()) for d in np.unique(og["d"])}
        yield "oof", ctx, stat, z, X, og["d"], og["t0"], firsts, {"og": og}
    else:
        from t2_m1_predict import hist_from_release
        from t2_occ_blend_predict import occ_from_release
        from tfb.data import RELEASE
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        stat = cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        for s in C.SPLITS:
            z = C.load(panel, s)
            base = RELEASE / "task2" / panel / s
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            occ = occ_from_release(ctx, Hh, Wi, t0) if family in OCC else None
            X = build_x(family, ctx, stat, v, qf, ok, occ, t0, grp.astype(bool), prof)
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            d = np.array([d_of[str(x)] for x in Wi.date], int)
            firsts = {int(k): int(t0[d == k].min()) for k in np.unique(d)}
            yield s, ctx, stat, z, X, d, t0, firsts, {"Wi": Wi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("tfm", "cnn3", "occ_cnn", "occ_tfm", "stid", "d2", "stidlh"))
    ap.add_argument("--adapt-params", default="all", choices=("all", "lh"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("lab", "oof", "official"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-oof", type=int, default=4000)
    args = ap.parse_args()
    global ADAPT_PARAMS
    ADAPT_PARAMS = args.adapt_params
    t_start = time.time()
    off = link_offsets()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        prof = PROFS[panel]
        for s, ctx, stat, z, X, d_all, t0_all, firsts, extra in windows_for(args.mode, i, panel, args.family,
                                                                             args.max_oof):
            lid = np.arange(ctx.S) + off[panel]
            if args.family == "d2":
                from t2_d2stgnn import corridor_adjacency
                CUR_A["A"] = torch.from_numpy(corridor_adjacency(C.load(panel, "validation")["milepost"])).to(C.DEV)
            dow_all = z["weekday"][d_all].astype(np.int64)
            lab = released_labels(ctx, z)
            modes = ("none", "sup") if args.mode == "lab" else ("sup",)
            for mode in modes:
                net = load_model(args.family, args.ckpt)
                rng = np.random.default_rng(7 + i)
                prob = np.zeros((len(d_all), F, ctx.S), np.float32)
                for d in sorted(firsts):
                    m = d_all == d
                    if mode == "sup":
                        past = np.arange(d)
                        first = firsts[d]
                        pool_days = np.append(past, d) if first > H + F + 2 else past
                        cuts = {int(k): (first if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            adapt(args.family, net, lid,
                                  lambda: sup_batch(args.family, ctx, stat, z, lab, pool_days, cuts, args.bs, rng, prof),
                                  args.steps, args.lr)
                    prob[m] = predict(args.family, net, X[m], lid, t0_all[m].astype(np.int64), dow_all[m])
                if args.mode == "lab":
                    res.setdefault((s, mode), {})[panel] = C.ongoing_iou(prob, extra["og"], 0.5).mean()
                    np.savez(CACHE / f"fam_{args.family}_{args.tag}_{mode}_{s}_{panel}.npz", og=prob.astype(np.float16))
                elif args.mode == "oof":
                    np.savez(CACHE / f"fam_{args.family}_{args.tag}_oof_{panel}.npz", og=prob.astype(np.float16),
                             d=d_all, t0=t0_all)
                else:
                    Wi = extra["Wi"]
                    args.out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(args.out_dir / f"prob_{s}_{panel}.npz", prob=prob,
                             is_onset=(Wi.condition == "queue_onset").to_numpy(),
                             window_id=Wi.window_id.to_numpy().astype(str), t0=t0_all)
            msg = "  ".join(f"{m} {res[(s, m)][panel]:.4f}" for m in modes) if args.mode == "lab" else "done"
            print(f"  {panel} {s}: {msg} ({time.time() - t_start:.0f}s)", flush=True)
            if args.mode == "oof":
                ctx.z = ctx.q = ctx.el = None
    if args.mode == "lab":
        for s in C.SPLITS:
            print(f"\n=== {s}: {args.family} {args.tag}, ongoing family IoU ===")
            for m in ("none", "sup"):
                print(f"  {m:<5} {C.family_mean(res[(s, m)]):.4f}")


if __name__ == "__main__":
    main()
