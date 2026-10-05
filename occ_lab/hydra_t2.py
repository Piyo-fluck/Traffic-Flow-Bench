"""A random-dictionary Task 2 member: HYDRA features (Dempster et al. 2023, DMKD) + a run-specific linear head solved
exactly every day from the released pseudo labels.

Every member so far is a LEARNED feature extractor (LightGBM on hand-built stats, CNN, transformer, STID) and the run
adaptation is 30 SGD steps on the whole network. This member changes both halves:

    features    HYDRA-2D: G = 16 groups x K = 8 random kernels (3 slots x 5 links, link dilation 1 / 2 / 4) over the
                window image (v/vf, q/cap, observed, queued, and their first differences in time); at every cell the
                kernels of a group compete - soft count (the winner's response) and hard count (the loser) - pooled
                over the whole hour and the last 20 min, and over +-2 links. Nothing is learned: a fixed random pattern
                dictionary, 3,072 features per link. Plus time of day, weekend and the 5 static link channels.
    head (a)    a logistic layer per horizon step (ongoing) / at T+30 (onset), fitted on the train run
    head (b)    the run: before each day, one exact Newton step of the logistic loss on EVERY released pseudo window
                of the scored run so far, from head (a): A_h = sum p(1-p) phi phi^T, b_h = sum (y - p) phi, both
                additive, so they are accumulated day by day; w_h = (A_h + lambda I)^-1 b_h, logit += phi . w_h.
                Pseudo windows (note 106): origins whose horizon ends before the day's first origin, history and labels
                from the released layer. All of the past, not 30 x 16 samples; no optimiser state, no step size.

    python occ_lab/hydra_t2.py --stage fit --tag h0        (train run -> work/models/hydra_h0.pt)
    python occ_lab/hydra_t2.py --stage lab --tag h0        (official-like pool of both scored months)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import longctx as LC  # noqa: E402  (T2CNN environment, lab kit, train-run loader)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C, cnn = LC.C, LC.cnn
H, F, T, DEV = LC.H, LC.F, LC.T, LC.DEV
CACHE, MODELS = LC.CACHE, LC.MODELS
G, K, DILS = 16, 8, (1, 2, 4)
NFEAT = 2 * len(DILS) * G * K * 2 * 2          # (x, dx) x dilations x G x K x (soft, hard) x (hour, last 20 min)
NEXT = 3 + 5                                   # time of day (sin, cos), weekend, static link channels
RAW = 4 * H                                    # --raw: the link's own history image (v/vf, q/cap, observed, queued)
LAMS = (1e2, 1e3, 1e4, 1e5)


# ------------------------------------------------------------------------------------------------ features
class Hydra2D:
    def __init__(self, seed=0, cin=4, raw=False):
        self.use_raw = raw
        g = torch.Generator().manual_seed(seed)
        self.W = {}
        for src in ("x", "dx"):
            for d in DILS:
                w = torch.randn(G * K, cin, 3, 5, generator=g)
                w = w - w.mean((1, 2, 3), keepdim=True)
                w = w / w.abs().sum((1, 2, 3), keepdim=True)
                self.W[(src, d)] = w.to(DEV)
        self.mu = self.sd = None

    @torch.no_grad()
    def raw(self, x):                                           # x (N, 4, H, S) float -> (N, NFEAT, S)
        dx = torch.diff(x, dim=2, prepend=x[:, :, :1])
        out = []
        for src, inp in (("x", x), ("dx", dx)):
            for d in DILS:
                r = Fn.conv2d(inp, self.W[(src, d)], padding=(1, 2 * d), dilation=(1, d))   # (N, GK, H, S)
                N, _, Hh, S = r.shape
                r = r.view(N, G, K, Hh, S)
                mx, am = r.max(2)
                soft = torch.zeros_like(r).scatter_(2, am.unsqueeze(2), mx.unsqueeze(2))
                hard = torch.zeros_like(r).scatter_(2, r.argmin(2).unsqueeze(2), 1.0)
                for pool in (slice(0, Hh), slice(Hh - 4, Hh)):
                    for v in (soft, hard):
                        f = v[:, :, :, pool].mean(3).reshape(N, G * K, S)
                        out.append(Fn.avg_pool1d(f, 5, 1, 2, count_include_pad=False))
        return torch.cat(out, 1)

    def fit_norm(self, f):
        self.mu, self.sd = f.mean((0, 2), keepdim=True), f.std((0, 2), keepdim=True).clamp(min=1e-4)

    def __call__(self, xw):
        """xw: make_x output (N, 12, H, S) float16 numpy -> phi (N, NFEAT + NEXT, S) float32 on DEV."""
        x = torch.from_numpy(np.ascontiguousarray(xw)).to(DEV, torch.float32)
        f = self.raw(x[:, :4])
        if self.mu is not None:
            f = ((f - self.mu) / self.sd).clamp(-6, 6)
        ext = [x[:, 9:12, -1], x[:, 4:9, -1]]                                  # time (3), static (5)
        if self.use_raw:
            ext.append(x[:, :4].reshape(len(x), RAW, -1))
        return torch.cat([f] + ext, 1)


class ESN(Hydra2D):
    """Echo state network over each link's hour: input = the link and +-2 neighbours (4 channels x 5 links),
    a fixed random leaky reservoir of 1,024 units (spectral radius 0.9, leak 0.5, input scale 0.5); features =
    the final state and the mean state over the hour. Only the readout is fitted (heads a and b)."""
    NR = 1024

    def __init__(self, seed=0, cin=4, raw=False):
        self.use_raw = raw
        g = torch.Generator().manual_seed(1000 + seed)
        Wr = torch.randn(self.NR, self.NR, generator=g) * (torch.rand(self.NR, self.NR, generator=g) < 0.02)
        Wr = Wr / torch.linalg.eigvals(Wr).abs().max() * 0.9
        self.Wr = Wr.to(DEV)
        self.Win = (torch.randn(cin * 5, self.NR, generator=g) * 0.5).to(DEV)
        self.bias = (torch.randn(self.NR, generator=g) * 0.1).to(DEV)
        self.mu = self.sd = None

    @torch.no_grad()
    def raw(self, x):                                           # (N, 4, H, S) -> (N, 2 NR, S)
        N, Cc, Hh, S = x.shape
        xp = Fn.pad(x, (2, 2))
        u = torch.stack([xp[..., k:k + S] for k in range(5)], 2).reshape(N, Cc * 5, Hh, S)
        u = u.permute(0, 3, 2, 1).reshape(N * S, Hh, Cc * 5)
        h = torch.zeros(N * S, self.NR, device=DEV)
        acc = torch.zeros_like(h)
        for t in range(Hh):
            h = 0.5 * h + 0.5 * torch.tanh(h @ self.Wr + u[:, t] @ self.Win + self.bias)
            acc += h
        f = torch.cat([h, acc / Hh], 1)
        return f.view(N, S, -1).permute(0, 2, 1)


EXTRACTORS = {"hydra": Hydra2D, "esn": ESN}


def nfeat(ext):
    return NFEAT if ext == "hydra" else 2 * ESN.NR


def feats(hy, xw, bs=64):
    return torch.cat([hy(xw[a:a + bs]) for a in range(0, len(xw), bs)]) if len(xw) else \
        torch.zeros(0, NFEAT + NEXT, xw.shape[-1], device=DEV)


# ------------------------------------------------------------------------------------------------ cell selection
def near(q_any, r=8):
    """(N, S) bool -> cells within r links of a True cell."""
    t = torch.from_numpy(q_any.astype(np.float32)).to(DEV)[:, None]
    return (Fn.max_pool1d(t, 2 * r + 1, 1, r)[:, 0] > 0).cpu().numpy()


def select_cells(hist_q, fut_q, rng, keep_far=0.05):
    m = near(hist_q.any(1) | fut_q.any(1))
    return m | (rng.random(m.shape) < keep_far)


# ------------------------------------------------------------------------------------------------ stage: fit
def stage_fit(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = LC.train_panels()
    hy = EXTRACTORS[args.extractor](args.seed, raw=args.raw)
    NF = nfeat(args.extractor)
    sample = []
    for p in parts:                                             # normalisation from 32 windows of every panel
        j = rng.choice(len(p["d"]), 32, replace=False)
        x = torch.from_numpy(LC.short_x(p, p["d"][j], p["t0"][j]).astype(np.float32)).to(DEV)
        sample.append(hy.raw(x[:, :4]).permute(1, 0, 2).reshape(NF, -1))
    sample = torch.cat(sample, 1)
    hy.mu, hy.sd = sample.mean(1)[None, :, None], sample.std(1)[None, :, None].clamp(min=1e-4)
    del sample
    rows = {"og": ([], [], []), "on": ([], [], [])}             # phi cells, labels (F or 1), masks
    for p in parts:
        n_on = len(p["ctx"].eo[0])
        is_on = np.arange(len(p["d"])) < n_on
        keep = np.flatnonzero(~p["es"])
        for kind, sel in (("on", keep[is_on[keep]]), ("og", keep[~is_on[keep]])):
            for a in range(0, len(sel), 256):
                j = sel[a:a + 256]
                d, t0 = p["d"][j], p["t0"][j]
                xw, y, m = LC.short_x(p, d, t0), None, None
                fi = t0[:, None] + np.arange(1, F + 1)
                y, m = p["q"][d[:, None], fi], p["el"][d[:, None], fi]
                hq = p["q"][d[:, None], t0[:, None] + np.arange(-H, 0)]
                if kind == "og":
                    cm = select_cells(hq, y, rng) & m.any(1)
                else:
                    cm = m[:, -1].copy()
                    y, m = y[:, -1:], m[:, -1:]
                phi = feats(hy, xw)
                ni, si = (torch.from_numpy(a).to(DEV) for a in np.nonzero(cm))
                rows[kind][0].append(phi[ni, :, si].half())
                nn_, ss_ = ni.cpu().numpy(), si.cpu().numpy()
                rows[kind][1].append(torch.from_numpy(y[nn_, :, ss_].astype(np.float32)))
                rows[kind][2].append(torch.from_numpy(m[nn_, :, ss_].astype(np.float32)))
        print(f"  [{p['panel']}] cells og {sum(len(r) for r in rows['og'][1]):,}  on {sum(len(r) for r in rows['on'][1]):,}",
              flush=True)
    heads = {}
    for kind, nout in (("og", F), ("on", 1)):
        X = torch.cat(rows[kind][0]); Y = torch.cat(rows[kind][1]).to(DEV); M = torch.cat(rows[kind][2]).to(DEV)
        lin = torch.nn.Linear(X.shape[1], nout).to(DEV)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=0)
        n = len(X)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * ((n + 8191) // 8192))
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=DEV)
            tot = 0.0
            for a in range(0, n, 8192):
                j = perm[a:a + 8192]
                z = lin(X[j].float())
                l = (Fn.binary_cross_entropy_with_logits(z, Y[j], reduction="none") * M[j]).sum() / M[j].sum().clamp(min=1)
                l = l + args.l2 * lin.weight.pow(2).sum()
                opt.zero_grad(); l.backward(); opt.step(); sched.step()
                tot += l.item() * len(j)
            print(f"  head {kind} epoch {ep + 1}: loss {tot / n:.4f}", flush=True)
        heads[kind] = {k: v.detach().cpu() for k, v in lin.state_dict().items()}
        del X, Y, M
    torch.save({"seed": args.seed, "raw": args.raw, "extractor": args.extractor, "mu": hy.mu.cpu(), "sd": hy.sd.cpu(), "heads": heads}, MODELS / f"hydra_{args.tag}.pt")
    print(f"saved hydra_{args.tag}.pt")


# ------------------------------------------------------------------------------------------------ stage: lab
def load_hydra(tag):
    d = torch.load(MODELS / f"hydra_{tag}.pt", map_location=DEV)
    hy = EXTRACTORS[d.get("extractor", "hydra")](d["seed"], raw=d.get("raw", False))
    hy.ext = d.get("extractor", "hydra")
    hy.mu, hy.sd = d["mu"].to(DEV), d["sd"].to(DEV)
    heads = {k: (v["weight"].to(DEV), v["bias"].to(DEV)) for k, v in d["heads"].items()}
    return hy, heads


def logits(phi, head):                                          # phi (N, D, S) -> (N, nout, S)
    w, b = head
    return torch.einsum("nds,od->nos", phi, w) + b[None, :, None]


class RunHead:
    """Accumulated exact Newton statistics of the logistic loss, per output (horizon step)."""
    def __init__(self, D, nout):
        self.A = torch.zeros(nout, D, D, device=DEV, dtype=torch.float64)
        self.b = torch.zeros(nout, D, device=DEV, dtype=torch.float64)
        self.n = 0

    def add(self, phi_c, z_c, y_c, m_c):                        # cells: (n, D), (n, nout), (n, nout), (n, nout)
        p = torch.sigmoid(z_c)
        w, r = p * (1 - p) * m_c, (y_c - p) * m_c
        P = phi_c.float()
        for o in range(self.A.shape[0]):                       # fp32 products, fp64 accumulation
            self.A[o] += ((P * w[:, o:o + 1]).T @ P).double()
            self.b[o] += (P.T @ r[:, o]).double()
        self.n += len(phi_c)

    def solve(self, lam):
        if self.n == 0:
            return None
        I = torch.eye(self.A.shape[1], device=DEV, dtype=torch.float64)
        return torch.stack([torch.linalg.solve(self.A[o] + lam * I, self.b[o]) for o in range(self.A.shape[0])]).float()


def pseudo_windows(ctx, stat, z, pool, d, t_lim, kind, step=2):
    """Released pseudo windows of day d with the horizon ending before t_lim: (x, y, m, hist_q)."""
    ov, oq, ok, q = pool
    t0 = np.arange(H + 1, min(t_lim, T) - F, step)
    if not len(t0):
        return None
    hq = np.stack([q[d, t - H:t] for t in t0]); fq = np.stack([q[d, t + 1:t + 1 + F] for t in t0])
    if kind == "og":
        keep = (hq.sum((1, 2)) >= 2) & fq.any((1, 2))
    else:
        keep = ~hq.any((1, 2)) & ~fq[:, :F - 1].any((1, 2)) & fq[:, F - 1].any(1)
    t0 = t0[keep]
    if not len(t0):
        return None
    dd = np.full(len(t0), d)
    hi, fi = t0[:, None] + np.arange(-H, 0), t0[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, ov[dd[:, None], hi], oq[dd[:, None], hi], ok[dd[:, None], hi], t0, z["weekday"][dd] >= 5)
    return x, q[dd[:, None], fi], ok[dd[:, None], fi], q[dd[:, None], hi]


def add_day(rh, hy, heads, ctx, stat, z, pool, d, t_lim, rng):
    for kind in ("og", "on"):
        pw = pseudo_windows(ctx, stat, z, pool, d, t_lim, kind)
        if pw is None:
            continue
        x, y, m, hq = pw
        if kind == "og":
            cm = select_cells(hq, y, rng) & m.any(1)
        else:
            cm = m[:, -1].copy()
            y, m = y[:, -1:], m[:, -1:]
        for a in range(0, len(x), 64):
            phi = feats(hy, x[a:a + 64])
            z_ = logits(phi, heads[kind])
            ni, si = np.nonzero(cm[a:a + 64])
            if not len(ni):
                continue
            tn = lambda v: torch.from_numpy(v[a:a + 64][ni, :, si].astype(np.float32)).to(DEV)  # noqa: E731
            ti, ts = torch.from_numpy(ni).to(DEV), torch.from_numpy(si).to(DEV)
            rh[kind].add(phi[ti, :, ts], z_[ti, :, ts], tn(y), tn(m))


def stage_lab(args):
    hy, heads = load_hydra(args.tag)
    D = nfeat(hy.ext) + NEXT + (RAW if hy.use_raw else 0)
    t_start = time.time()
    res = {}
    modes = ["none"] + [f"run{lam:g}" for lam in LAMS]
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            ov, oq = z["obs_speed"], z["obs_flow"]
            ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
            pool = (ov, oq, ok, ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf))
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in days}
            rng = np.random.default_rng(7 + i)
            rh = {"og": RunHead(D, F), "on": RunHead(D, 1)}
            done = 0                                          # days fully accumulated: 0 .. done-1
            P = {m: (np.zeros((len(on["d"]), ctx.S), np.float32), np.zeros(og["fut"].shape, np.float32)) for m in modes}
            for d in days:
                d = int(d)
                while done < d:                               # whole earlier days
                    add_day(rh, hy, heads, ctx, stat, z, pool, done, T, rng)
                    done += 1
                tmp = {k: RunHead(D, v.A.shape[0]) for k, v in rh.items()}
                for k in rh:
                    tmp[k].A, tmp[k].b, tmp[k].n = rh[k].A.clone(), rh[k].b.clone(), rh[k].n
                add_day(tmp, hy, heads, ctx, stat, z, pool, d, firsts[d], rng)   # today, before the first origin
                ws = {(k, lam): tmp[k].solve(lam) for k in tmp for lam in LAMS}
                for Wn, kind in ((on, "on"), (og, "og")):
                    idx = np.flatnonzero(Wn["d"] == d)
                    if not len(idx):
                        continue
                    phi = feats(hy, C.make_x(ctx, stat, Wn, idx, z))
                    z0 = logits(phi, heads[kind])
                    for m in modes:
                        zz = z0
                        if m != "none":
                            w = ws[(kind, float(m[3:]))]
                            if w is not None:
                                zz = z0 + torch.einsum("nds,od->nos", phi, w)
                        pr = torch.sigmoid(zz).cpu().numpy()
                        if kind == "on":
                            P[m][0][idx] = pr[:, 0]
                        else:
                            P[m][1][idx] = pr
            for m in modes:
                p_on, p_og = P[m]
                res.setdefault((s, m), {})[panel] = (C.onset_iou(p_on, on).mean(), C.ongoing_iou(p_og, og, 0.5).mean())
                np.savez(CACHE / f"fam_hydra_{args.tag}_{m}_{s}_{panel}.npz", og=p_og.astype(np.float16), on=p_on)
                np.savez(CACHE / f"onset_adapt_hydra_{args.tag}_{m}_{s}_{panel}.npz", prob=p_on)
            msg = "  ".join(f"{m} {res[(s, m)][panel][1]:.4f}" for m in modes)
            print(f"  {panel} {s}: ongoing {msg} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: hydra {args.tag} (single member, own IoU) ===")
        for m in modes:
            d = res[(s, m)]
            print(f"  {m:<8} onset {C.family_mean({k: v[0] for k, v in d.items()}):.4f}  "
                  f"ongoing {C.family_mean({k: v[1] for k, v in d.items()}):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("fit", "lab"))
    ap.add_argument("--tag", default="h0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--extractor", default="hydra", choices=("hydra", "esn"))
    args = ap.parse_args()
    stage_fit(args) if args.stage == "fit" else stage_lab(args)


if __name__ == "__main__":
    main()
