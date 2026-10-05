"""Causal pseudo-SUPERVISED adaptation on the official Task 2 windows (note 106).

The production adaptation (idea 9b, `scripts/t2_adapt_official.py`) fine-tunes on the scored month with a masked
reconstruction loss and never uses a Task 2 label. The organizers' clarification allows every released
observation at or before a window's origin, so real task windows can be built from the scored run itself: an
origin tau with tau + 30 min at or before the cut-off, its history from the released layer, and its labels from
the released layer at tau+5 .. tau+30 - exact wherever a cell is visible, since obs == true (note 103).

Locally this beat the reconstruction-only recipe on both months and both halves: onset 0.7375 / 0.7457 against
0.7345 / 0.6794, ongoing 0.8394 / 0.8192 against 0.7990 / 0.7784 (single model, note 106).

Same interface as the 9b script: writes prob_<split>_<panel>.npz with prob (N, F, S), is_onset, window_id, t0.

    T2CNN_REC=0.3 T2CNN_LOVASZ=0.5 .venv/Scripts/python.exe scripts/t2_pseudo_adapt_official.py \
        --out-dir work/sub_t2pseudo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

DEFAULT_CKPT = ROOT / "work" / ("models_remote" if (ROOT / "work" / "models_remote").exists() else "models") \
    / "t2_cnn_lov_rec_s20.pt"
H = cnn.H
rng = np.random.default_rng(3)


def rec_batch(ctx, stat, z, d_max, bs):
    """History windows from strictly earlier days, for the reconstruction half of the loss."""
    if d_max <= 0:
        return None
    days = rng.integers(0, d_max, size=bs)
    t0 = rng.integers(H + 1, z["obs_speed"].shape[1] - F - 1, size=bs)
    hi = t0[:, None] + np.arange(-H, 0)
    v, q = z["obs_speed"][days[:, None], hi], z["obs_flow"][days[:, None], hi]
    ok = z["elig"][days[:, None], hi] & np.isfinite(v) & np.isfinite(q)
    x = cnn.make_x(ctx, stat, v, q, ok, t0, z["weekday"][days] >= 5)
    return torch.from_numpy(np.ascontiguousarray(x)).to(cnn.DEV, torch.float32)


def sup_batch(ctx, stat, z, lab, d_max, same_day, cut, bs):
    """Pseudo Task 2 windows whose whole horizon is at or before the cut-off: history -> queue labels."""
    ov, oq, ok_all, q = lab
    T = z["obs_speed"].shape[1]
    cand = []
    for _ in range(bs * 8):
        use_today = same_day is not None and rng.random() < 0.35 and cut > H + F + 2
        d = int(same_day) if use_today else int(rng.integers(0, max(d_max, 1)))
        lim = min(cut if use_today else T - 1, T - 1)
        if lim <= H + F + 1 or (not use_today and d_max <= 0):
            continue
        t0 = int(rng.integers(H + 1, lim - F))
        hist = q[d, t0 - H:t0]
        fut = q[d, t0 + 1:t0 + 1 + F]
        if hist.sum() >= 2 and fut.any():                    # the ongoing selector, the one that carries signal
            cand.append((d, t0))
        if len(cand) >= bs:
            break
    if not cand:
        return None
    d = np.array([c[0] for c in cand]); t0 = np.array([c[1] for c in cand])
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, ov[d[:, None], hi], oq[d[:, None], hi], ok_all[d[:, None], hi], t0,
                   z["weekday"][d] >= 5)
    to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)  # noqa: E731
    return (to(x, torch.float32), to(q[d[:, None], fi].astype(np.float32), torch.float32),
            to(ok_all[d[:, None], fi].astype(np.float32), torch.float32))


def adapt(net, sup, rec, steps, lr, w_rec):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        loss = None
        b = sup()
        if b is not None:
            x, y, m = b
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            logit = out[:, :F].float()
            loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        if w_rec > 0:
            xb = rec()
            if xb is not None:
                hid = torch.rand_like(xb[:, 2]) < 0.2
                xin = xb.clone()
                xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    r = net.rec(net.blocks(net.inp(xin))).float()
                w = (xb[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
                lr_ = ((r - xb[:, 0:2].float()) ** 2 * w).sum() / w.sum().clamp(min=1)
                loss = w_rec * lr_ if loss is None else loss + w_rec * lr_
        if loss is None:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--solid3", action="store_true",
                    help="note 141: after the day's adaptation, each ONGOING window gets its own head fitted on the "
                         "released pseudo windows closest to it (earlier days + the same day with the horizon "
                         "ended before its origin), occ_lab/adapt_variants.solid2_predict")
    ap.add_argument("--arch", choices=("cnn", "tfm"), default="cnn",
                    help="tfm: the ongoing transformer family, adapted with the supervised loss only (it has no "
                         "reconstruction head; note 75's reconstruction-only adaptation hurt it)")
    args = ap.parse_args()
    if args.arch == "tfm":
        args.w_rec = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    sd = torch.load(args.ckpt, map_location=cnn.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if args.arch == "tfm" else sd["inp.weight"].shape[1]
    width = sd["gru.weight_hh_l0"].shape[1] if args.arch == "tfm" else cnn.WIDTH
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        for split in ("validation", "private"):
            z = load(panel, split)
            ov, oq = z["obs_speed"], z["obs_flow"]
            ok_all = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
            lab = (ov, oq, ok_all, ok_all & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf))
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            d = np.array([d_of[str(x)] for x in W.date], int)
            net = (cnn.T2Tfm if args.arch == "tfm" else cnn.T2Net)(cin, width).to(cnn.DEV)
            net.load_state_dict(sd)
            net.eval()
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            for day in sorted(set(d.tolist())):
                m = d == day
                cut = int(t0[m].min())                        # the first official origin of the day
                adapt(net,
                      lambda: sup_batch(ctx, stat, z, lab, day, day, cut, args.bs),
                      lambda: rec_batch(ctx, stat, z, day, args.bs),
                      args.steps, args.lr, args.w_rec)
                x = cnn.make_x(ctx, stat, v[m], qf[m], ok[m], t0[m], grp[m].astype(bool))
                prob[m] = cnn.predict(net, x, None)
                og_m = m & (W.condition == "queue_ongoing").to_numpy()
                if args.solid3 and og_m.any():
                    sys.path.insert(0, str(ROOT / "occ_lab"))
                    import adapt_variants as AV                    # noqa: E402  (same t2_cnn module)
                    days3 = np.arange(day + 1)
                    cuts3 = {int(k): (int(t0[og_m].max()) if int(k) == day else 10 ** 6) for k in days3}
                    P3 = AV.pool_windows(ctx, stat, z, lab, days3, cuts3, np.random.default_rng(7 + i), cap=900)
                    if P3 is not None:
                        xo = cnn.make_x(ctx, stat, v[og_m], qf[og_m], ok[og_m], t0[og_m], grp[og_m].astype(bool))
                        prob[og_m] = AV.solid2_predict(net, xo, t0[og_m], P3, day=day)
                    for p_ in net.parameters():                  # solid2_predict froze all but the head
                        p_.requires_grad_(True)
            np.savez(args.out_dir / f"prob_{split}_{panel}.npz", prob=prob,
                     is_onset=(W.condition == "queue_onset").to_numpy(),
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            print(f"[{split} {panel}] {len(W)} windows over {len(set(d.tolist()))} days "
                  f"({time.time() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
