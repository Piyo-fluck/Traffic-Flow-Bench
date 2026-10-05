"""Causal pseudo-SUPERVISED adaptation of the ongoing STID family (note 106's recipe, as for the transformers).

STID (occ_lab/train_stid.py) reads the CNN channels plus three occupancy channels, a global link id (the T2 panels'
links numbered in panel order), the origin slot and the weekday. In AT it feeds the residual corrector only.
Adapted per day on real Task 2 windows of the scored run strictly before the day's first origin (earlier days and
the same day before it), labels from the released layer, BCE + Lovasz, weights carried forward.

    lab       --mode lab : official-like pool, frozen vs adapted, caches occ_lab/cache/stidps_<tag>_<mode>_<split>_<panel>.npz
    official  --mode official --out-dir DIR : the official windows, prob_<split>_<panel>.npz as t2_pseudo_adapt_official.py

    .venv/Scripts/python.exe occ_lab/pseudo_adapt_stid.py --ckpt occ_lab/models/stid_s0.pt --tag s0 --mode lab
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import pseudo_adapt as PA  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID  # noqa: E402

C = PA.C
cnn = C.cnn
H, F = C.H, C.F
PROFS = np.load(PROJ / "work" / "models_remote" / "occ_profiles.npz")


def link_offsets():
    off, o = {}, 0
    for panel in C.T2_PANELS:
        off[panel] = o
        o += len(C.load(panel, "validation")["link_ids"])
    return off


def load_stid(p):
    d = torch.load(p, map_location=C.DEV)
    net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
    net.load_state_dict(d["state"])
    return net.eval()


def x_occ(ctx, stat, v, qf, ok, occ, t0, wk, prof):
    x = cnn.make_x(ctx, stat, v, qf, ok, t0, wk)
    return np.concatenate([x, occ_channels(occ, ok, prof)], 1)


def fwd(net, x, lid, t0, dow):
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(C.DEV, dt)  # noqa: E731
    L = np.broadcast_to(lid, (len(x), len(lid))).copy()
    return net(tt(x, torch.float32), tt(L, torch.long), tt(t0, torch.long), tt(dow, torch.long))


@torch.no_grad()
def predict(net, x, lid, t0, dow, bs=64):
    net.eval()
    out = [torch.sigmoid(fwd(net, x[a:a + bs], lid, t0[a:a + bs], dow[a:a + bs])).float().cpu().numpy()
           for a in range(0, len(x), bs)]
    return np.concatenate(out) if out else np.zeros((0, F, len(lid)), np.float32)


def sup_batch(ctx, stat, z, lab, days, cuts, bs, rng, prof):
    """Real ongoing windows of the scored run whose horizon ends before the day's cut (PA.sup_batch's selector)."""
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
    x = x_occ(ctx, stat, ov[d[:, None], hi], oq[d[:, None], hi], ok_all[d[:, None], hi],
              z["obs_occ"][d[:, None], hi], t0, z["weekday"][d] >= 5, prof)
    return x, q[d[:, None], fi].astype(np.float32), ok_all[d[:, None], fi].astype(np.float32), t0, \
        z["weekday"][d].astype(np.int64)


def adapt(net, lid, batches, steps, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, y, m, t0, dow = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(net, x, lid, t0, dow)
        logit = out.float()
        yt, mt = torch.from_numpy(y).to(C.DEV), torch.from_numpy(m).to(C.DEV)
        loss = cnn.loss_fn(logit, yt, mt) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yt, mt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()


def labels(ctx, z):
    ov, oq = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
    return ov, oq, ok, ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", choices=("lab", "official"), default="lab")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    t_start = time.time()
    off = link_offsets()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        prof = PROFS[panel]
        if args.mode == "lab":
            ctx, stat, W = C.windows(i, panel)
        else:
            ctx = C.Ctx(panel, i)
            ctx.fitd = np.ones_like(ctx.fitd)
            stat = cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        lid = np.arange(ctx.S) + off[panel]
        for s in C.SPLITS:
            if args.mode == "lab":
                z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
                if not len(og["d"]):
                    continue
                d_all, t0_all = og["d"], og["t0"]
                occ = np.stack([z["obs_occ"][a, b - H:b] for a, b in zip(og["d"], og["t0"])])
                X = x_occ(ctx, stat, og["v"], og["qf"], og["ok"], occ, og["t0"], z["weekday"][og["d"]] >= 5, prof)
                firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                          og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in np.unique(og["d"])}
            else:
                from t2_m1_predict import hist_from_release
                from t2_occ_blend_predict import occ_from_release
                from tfb.data import RELEASE
                z = C.load(panel, s)
                base = RELEASE / "task2" / panel / s
                Wi = pd.read_csv(base / "window_index.csv")
                Hh = pd.read_parquet(base / "window_history.parquet")
                v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
                X = x_occ(ctx, stat, v, qf, ok, occ_from_release(ctx, Hh, Wi, t0), t0, grp.astype(bool), prof)
                d_of = {str(x): k for k, x in enumerate(z["dates"])}
                d_all = np.array([d_of[str(x)] for x in Wi.date], int)
                t0_all = t0
                firsts = {int(d): int(t0[d_all == d].min()) for d in np.unique(d_all)}
            dow_all = z["weekday"][d_all].astype(np.int64)
            lab = labels(ctx, z)
            modes = ("none", "sup") if args.mode == "lab" else ("sup",)
            for mode in modes:
                net = load_stid(args.ckpt)
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
                            adapt(net, lid, lambda: sup_batch(ctx, stat, z, lab, pool_days, cuts, args.bs, rng, prof),
                                  args.steps, args.lr)
                    prob[m] = predict(net, X[m], lid, t0_all[m].astype(np.int64), dow_all[m])
                if args.mode == "lab":
                    res.setdefault((s, mode), {})[panel] = C.ongoing_iou(prob, og, 0.5).mean()
                    np.savez(LAB / "cache" / f"stidps_{args.tag}_{mode}_{s}_{panel}.npz", og=prob.astype(np.float16))
                else:
                    args.out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(args.out_dir / f"prob_{s}_{panel}.npz", prob=prob,
                             is_onset=(Wi.condition == "queue_onset").to_numpy(),
                             window_id=Wi.window_id.to_numpy().astype(str), t0=t0)
            msg = "  ".join(f"{m} {res[(s, m)][panel]:.4f}" for m in modes) if args.mode == "lab" else "done"
            print(f"  {panel} {s}: {msg} ({time.time() - t_start:.0f}s)", flush=True)
    if args.mode == "lab":
        for s in C.SPLITS:
            print(f"\n=== {s}: STID {args.tag}, ongoing family IoU ===")
            for m in ("none", "sup"):
                print(f"  {m:<5} {C.family_mean(res[(s, m)]):.4f}")


if __name__ == "__main__":
    main()
