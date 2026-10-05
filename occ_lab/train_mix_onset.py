"""Onset with a joint head: a mixture over configurations instead of per-link marginals.

Note 51 put 12% of the onset windows at IoU 0, with the LARGEST top1-top2 margin on exactly those windows and
the chosen link a median of 26 links from the true one. A model that is confidently wrong on a tenth of the
windows, while holding the true bottleneck in its top 3 on 98-99% of them, is a model whose posterior is
MULTI-MODAL and whose head can only report the average of the modes. Per-link marginals cannot say "either the
habitual bottleneck or that one, as a coherent set"; a mixture can.

The backbone is untouched (`OnsetSTID`, initialised from the production seed) so this isolates the head and the
decision from everything else. Compared on the same windows:

    marginal, ratio rule    the production decision on this backbone
    mixture, MC prefix      k chosen to maximise the expected IoU under sampled configurations

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_mix_onset.py --k 8 --steps 8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "cnn")
os.environ.setdefault("OMP_NUM_THREADS", "8")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import t2_cnn as cnn  # noqa: E402
from mixture import MixtureHead, decide, mcl_loss, mixture_nll, sample_configs, spread  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_hazard import index_windows  # noqa: E402
from train_onset_stid import OnsetSTID  # noqa: E402

MODELS = LAB / "models"


class MixOnset(nn.Module):
    """OnsetSTID's trunk with a mixture head in place of the single logit map."""

    def __init__(self, base: OnsetSTID, k: int):
        super().__init__()
        self.net = base
        self.head = MixtureHead(base.head.in_features, 1, k)

    def forward(self, x, lid, tod, dow):
        n = self.net
        B, C, Hh, S = x.shape
        h = n.series(x.permute(0, 3, 2, 1).reshape(B, S, Hh * C))
        h = torch.cat([h, n.e_link(lid), n.e_tod(tod)[:, None].expand(-1, S, -1),
                       n.e_dow(dow)[:, None].expand(-1, S, -1)], -1)
        for blk in n.blocks:
            h = h + blk(h)
        return self.head(n.norm(h))                           # (B, K, 1, S), (B, K)


def to_dev(a, d):
    return torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, d)


def build(ctx, stat, prof, d, t0, true_layer=True, z=None):
    z = z if z is not None else ctx.z
    hi = t0[:, None] + np.arange(-H, 0)
    if true_layer:
        ok = ctx.el[d[:, None], hi]
        v, q, oc = z["true_speed"], z["true_flow"], z["true_occ"]
    else:
        v, q, oc = z["obs_speed"], z["obs_flow"], z["obs_occ"]
        ok = z["elig"][d[:, None], hi] & np.isfinite(v[d[:, None], hi]) & np.isfinite(q[d[:, None], hi])
    x = cnn.make_x(ctx, stat, v[d[:, None], hi], q[d[:, None], hi], ok, t0, z["weekday"][d] >= 5)
    return np.concatenate([x, occ_channels(oc[d[:, None], hi], ok, prof)], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--init", type=Path, default=MODELS / "onset_stid_s0.pt")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="mcl", choices=("mcl", "mixture"))
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cnn.rng = rng
    torch.manual_seed(args.seed)
    t0_ = time.time()
    profs = np.load(MODELS / "occ_profiles.npz")
    sd = torch.load(args.init, map_location=cnn.DEV)
    parts, loff = [], 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = loff
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        d, t0 = index_windows(ctx, "earliest")
        es = ctx.z["dates"][d] >= FIT_END
        parts.append({"ctx": ctx, "stat": stat, "prof": profs[panel], "lid": lid,
                      "d": d, "t0": t0, "fit": np.flatnonzero(~es), "ev": np.flatnonzero(es), "panel": panel})
        print(f"[{panel}] {int((~es).sum())} fit, {int(es.sum())} held-out ({time.time() - t0_:.0f}s)",
              flush=True)
    base = OnsetSTID(sd["cin"], sd["n_links"], sd["n_slots"], ident=sd.get("ident", "id"))
    base.load_state_dict(sd["state"])
    net = MixOnset(base, args.k).to(cnn.DEV)
    print(f"mixture head K={args.k}, backbone from {args.init.name}, "
          f"{sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    def infer(p, d, t0, z=None, true_layer=True):
        x = build(p["ctx"], p["stat"], p["prof"], d, t0, true_layer, z)
        zz = z if z is not None else p["ctx"].z
        with torch.no_grad():
            lg, pi = net(to_dev(x, torch.float32),
                         to_dev(np.broadcast_to(p["lid"], (len(d), len(p["lid"]))).copy(), torch.long),
                         to_dev(t0.astype(np.int64), torch.long),
                         to_dev(zz["weekday"][d].astype(np.int64), torch.long))
        return torch.sigmoid(lg).float().cpu().numpy(), torch.softmax(pi, -1).float().cpu().numpy()

    def spread_report(P, W):
        r = np.random.default_rng(1)
        j_, i_ = [], []
        for j in range(min(len(P), 40)):
            a, b = spread(P[j], W[j], 400, r)
            j_.append(a)
            i_.append(b)
        return float(np.mean(j_)), float(np.mean(i_))

    def score(P, W, y, e):
        """marginal ratio rule vs the Monte Carlo prefix under the mixture."""
        a, b = [], []
        r = np.random.default_rng(0)
        for j in range(len(y)):
            m = e[j]
            if not m.any() or not y[j][m].any():
                continue
            marg = (W[j][:, None, None] * P[j]).sum(0)[0]
            a.append(int((select(marg[m], "ratio", 1.0) & y[j][m]).sum())
                     / max(int((select(marg[m], "ratio", 1.0) | y[j][m]).sum()), 1))
            S = sample_configs(P[j], W[j], args.samples, r)[:, 0]
            pr = decide(marg[None], S[:, None], m[None])[0]
            b.append(int((pr[m] & y[j][m]).sum()) / max(int((pr[m] | y[j][m]).sum()), 1))
        return float(np.mean(a)), float(np.mean(b))

    def evaluate():
        net.eval()
        ma, mb = [], []
        for p in parts:
            ev = p["ev"]
            if not len(ev):
                continue
            P, W = infer(p, p["d"][ev], p["t0"][ev])
            ctx = p["ctx"]
            tg = p["t0"][ev] + F
            a, b = score(P, W, ctx.q[p["d"][ev], tg], ctx.el[p["d"][ev], tg])
            ma.append(a)
            mb.append(b)
        net.train()
        return float(np.mean(ma)), float(np.mean(mb))

    best, best_state = -1.0, None
    for step in range(1, args.steps + 1):
        p = parts[rng.integers(len(parts))]
        j = p["fit"][rng.integers(len(p["fit"]), size=args.bs)]
        d, t0 = p["d"][j], p["t0"][j]
        x = build(p["ctx"], p["stat"], p["prof"], d, t0)
        lid = np.broadcast_to(p["lid"], (args.bs, len(p["lid"]))).copy()
        if rng.random() < 0.5:
            lid[:] = p["ctx"].loff * 0 + sd["n_links"]
        lg, pi = net(to_dev(x, torch.float32), to_dev(lid, torch.long),
                     to_dev(t0.astype(np.int64), torch.long),
                     to_dev(p["ctx"].z["weekday"][d].astype(np.int64), torch.long))
        tg = t0 + F
        y = to_dev(p["ctx"].q[d, tg].astype(np.float32)[:, None], torch.float32)
        m = to_dev(p["ctx"].el[d, tg].astype(np.float32)[:, None], torch.float32)
        loss = (mcl_loss(lg.float(), pi.float(), y, m, args.eps) if args.loss == "mcl"
                else mixture_nll(lg.float(), pi.float(), y, m))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            a, b = evaluate()
            p0 = parts[0]
            Pd, Wd = infer(p0, p0["d"][p0["ev"]], p0["t0"][p0["ev"]])
            sj, si = spread_report(Pd, Wd)
            flag = ""
            if b > best:
                best, flag = b, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"step {step}: loss {loss.item():.3f}  marginal {a:.4f}  mixture {b:.4f}{flag}  "
                  f"sd|Y| joint {sj:.2f} vs independent {si:.2f} ({time.time() - t0_:.0f}s)", flush=True)

    torch.save({"state": best_state, "k": args.k, "cin": sd["cin"], "n_links": sd["n_links"],
                "n_slots": sd["n_slots"]}, MODELS / f"mix_onset_k{args.k}_s{args.seed}.pt")
    net.load_state_dict(best_state)
    net.eval()
    print(f"\nscored months, official-like onset windows, released inputs\n", flush=True)
    out = {"marginal": {}, "mixture": {}}
    for p in parts:
        ctx = p["ctx"]
        for split in ("validation", "private"):
            zs = load(p["panel"], split)
            w = first_onsets(ctx.vf, zs, blank=official_blank(p["panel"], split, zs))
            if not len(w["d"]):
                continue
            P, W = infer(p, w["d"], w["t0"], zs, true_layer=False)
            a, b = score(P, W, w["fut"][:, F - 1], w["efut"][:, F - 1])
            out["marginal"].setdefault(split, {})[p["panel"]] = a
            out["mixture"].setdefault(split, {})[p["panel"]] = b
    print(f"{'decision':<12}{'validation':>12}{'private':>11}{'mean':>10}")
    for k, v in out.items():
        m = [family_mean(v[s]) for s in ("validation", "private") if s in v]
        print(f"{k:<12}" + "".join(f"{x:>12.4f}" for x in m) + f"{float(np.mean(m)):>10.4f}")


if __name__ == "__main__":
    main()
