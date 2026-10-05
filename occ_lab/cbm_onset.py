"""Onset as a Concept Bottleneck Model (CBM), trained on the TRAINING months only.

Why: the onset loss is picking the wrong bottleneck among the candidates (note 129), and link identities carry the
training run's own activation pattern (note 169: I5_N head 114 queued on 198 of 273 training days, 0 in both scored
months). A CBM makes the onset decision pass through run-invariant physical quantities predicted per link:

    concept net   corridor history (12 slots of v/vf, q/q95_train, ok, on/off-ramp flows) + static road channels
                  (lanes, cap, vf, ramps, train q95 per lane, capacity step), NO link identity ->
                  dilated 1-D convolutions along the corridor -> per link, at T+15 / T+30:
                      c0 queued at T+30 (logit)       c1 queued at T+15 (logit)
                      c2 v(T+30)/vf                   c3 q(T+30)/q95         c4 v(T+15)/vf
                  trained on EVERY window of the training months (link-level mask: the link and +-5 neighbours
                  queue-free over the history, the onset precondition), ~ 100x the onset windows' labels
    onset head    g(concepts only, +-3 links) -> onset logit per link; trained on the training months' onset
                  windows (corridor queue-free over history and T+5..T+25, queue at T+30), concepts frozen
                  (sequential CBM) - the decision can only use the concepts
    direct        (baseline) the same backbone trained end-to-end on the onset windows only

    CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe occ_lab/cbm_onset.py --mode cbm --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts", LAB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import ES_START  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import fd_params, load_ramps  # noqa: E402

RESID = os.environ.get("CBM_RESID", "") == "1"
# static channels kept (0 lanes, 1 cap/lane, 2 vf, 3 on-ramp, 4 off-ramp, 5 train q95/lane, 6 cap step, 7 has q95)
ST_KEEP = [int(c) for c in os.environ.get("CBM_ST", "0,1,2,3,4,5,6,7").split(",") if c != ""]
DIL = tuple(int(c) for c in os.environ.get("CBM_DIL", "1,2,4,8,16,1,2,4").split(","))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = LAB / "models"
CACHE = LAB / "cache"
LO, HI = 48, 168                       # onset slots T+30 in 04:00 .. 14:00 (as the official / lab selector)
NBR = 5                                # link-level calm mask: +-5 links queue-free over the history
NC = 5
CH = 5                                 # dynamic channels: v/vf, q/q95, ok, on-ramp, off-ramp


# ------------------------------------------------------------------------------------------------ data
def panel_static(panel):
    zt = load(panel, "train")
    vf = np.asarray(zt["free_speed"], np.float32)
    prm = fd_params(panel, zt["link_ids"])
    lanes, cap = np.asarray(prm["lanes"], np.float32), np.asarray(prm["cap"], np.float32)
    v, q = zt["obs_speed"], zt["obs_flow"]
    ok = zt["elig"] & np.isfinite(v) & np.isfinite(q)
    free = ok & (np.nan_to_num(v, nan=0) >= 0.7 * vf)
    with np.errstate(all="ignore"):
        q95 = np.nanpercentile(np.where(free, q, np.nan).reshape(-1, len(vf)), 95, axis=0).astype(np.float32)
    has = np.isfinite(q95)
    q95 = np.where(has, q95, 0.9 * cap).astype(np.float32)
    S = len(vf)
    left = np.array([q95[max(0, j - 4):max(1, j - 1)].max() if j > 1 else q95[j] for j in range(S)])
    right = np.array([q95[j + 2:j + 5].max() if j + 2 < S else q95[j] for j in range(S)])
    step = q95 / np.fmin(left, right)
    ramps = load_ramps(panel, "train", zt)
    st = np.stack([lanes / 5, cap / lanes / 2500, vf / 120, ramps["on_att"].astype(np.float32),
                   ramps["off_att"].astype(np.float32), q95 / lanes / 2000, np.clip(step, 0, 3), has.astype(np.float32)])
    return dict(vf=vf, q95=q95, st=st[ST_KEEP].astype(np.float32), zt=zt, ramps_t=ramps)


def dyn(z, ramps, vf, q95):
    """(D, T, CH, S) float16 dynamic channels, (D, T, S) queued, (D, T, S) ok."""
    v, q = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    vv = np.where(ok, np.nan_to_num(v) / vf, 0).clip(0, 1.5)
    qq = np.where(ok, np.nan_to_num(q) / q95, 0).clip(0, 3)
    on = np.where(ramps["on_v"], np.log1p(np.nan_to_num(ramps["on_q"]).clip(0)) / 6, 0)
    off = np.where(ramps["off_v"], np.log1p(np.nan_to_num(ramps["off_q"]).clip(0)) / 6, 0)
    X = np.stack([vv, qq, ok.astype(np.float32), on, off], 2).astype(np.float16)
    queued = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    return X, queued, ok


class Panel:
    def __init__(self, i, panel):
        s = panel_static(panel)
        self.panel, self.vf, self.q95 = panel, s["vf"], s["q95"]
        zt = s["zt"]
        X, qd, ok = dyn(zt, s["ramps_t"], self.vf, self.q95)
        self.D, self.T, self.S = qd.shape
        self.part = np.where(zt["dates"] < ES_START, 0, 1)
        self.X = torch.from_numpy(X).to(DEV)                        # (D, T, CH, S)
        self.qd = torch.from_numpy(qd).to(DEV)
        self.ok = torch.from_numpy(ok).to(DEV)
        self.st = torch.from_numpy(s["st"]).to(DEV)                # (8, S)
        self.wkend = torch.from_numpy((zt["weekday"] >= 5).astype(np.float32)).to(DEV)
        # link-level calm over the history: no queue within +-NBR links in slots t0-H .. t0-1
        qn = qd.copy()
        for k in range(1, NBR + 1):
            qn[:, :, k:] |= qd[:, :, :-k]
            qn[:, :, :-k] |= qd[:, :, k:]
        cs = np.concatenate([np.zeros((self.D, 1, self.S), np.int32), np.cumsum(qn, 1, dtype=np.int32)], 1)
        self.calm_cs = torch.from_numpy(cs).to(DEV)                 # calm(t0) = cs[t0] - cs[t0-H] == 0
        # onset windows: corridor queue-free over t0-H .. t0+F-1, queue at t0+F, t0+F in [LO, HI)
        anyq = qd.any(2)
        on_d, on_t = [], []
        for d in range(self.D):
            for t0 in range(max(H, LO - F), HI - F):
                if anyq[d, t0 + F] and not anyq[d, t0 - H:t0 + F].any():
                    on_d.append(d); on_t.append(t0)
        self.on_d, self.on_t = np.array(on_d), np.array(on_t)


def inputs(P, d, t0):
    """d, t0 (B,) long tensors -> x (B, H*CH + 8 + 3, S)."""
    idx = t0[:, None] + torch.arange(-H, 0, device=DEV)[None]      # (B, H)
    x = P.X[d[:, None], idx].float()                                # (B, H, CH, S)
    B = len(d)
    x = x.reshape(B, H * CH, P.S)
    tod = t0.float() / 288 * 2 * np.pi
    g = torch.stack([torch.sin(tod), torch.cos(tod), P.wkend[d]], 1)[:, :, None].expand(-1, -1, P.S)
    return torch.cat([x, P.st[None].expand(B, -1, -1), g], 1)


def targets(P, d, t0):
    t30, t15 = t0 + F, t0 + F // 2
    calm = (P.calm_cs[d, t0] - P.calm_cs[d, t0 - H]) == 0          # (B, S)
    y = dict(q30=P.qd[d, t30].float(), q15=P.qd[d, t15].float(),
             v30=P.X[d, t30, 0].float(), q30f=P.X[d, t30, 1].float(), v15=P.X[d, t15, 0].float(),
             m30=(P.ok[d, t30] & calm).float(), m15=(P.ok[d, t15] & calm).float())
    return y


def scored_inputs(P, split, on, st_np, shift=0, arrays=None):
    """The lab / official windows' input from their released history (same channels); shift moves the origin back."""
    if arrays is None:
        z = load(P.panel, split)
        ramps = load_ramps(P.panel, split, z)
        arrays = dyn(z, ramps, P.vf, P.q95) + (z,)
    X, qd, ok, z = arrays
    d, t0 = on["d"], on["t0"] - shift
    idx = t0[:, None] + np.arange(-H, 0)[None]
    x = X[d[:, None], idx].astype(np.float32).reshape(len(d), H * CH, P.S)
    tod = t0 / 288 * 2 * np.pi
    g = np.stack([np.sin(tod), np.cos(tod), (z["weekday"][d] >= 5).astype(np.float32)], 1)[:, :, None]
    g = np.broadcast_to(g, (len(d), 3, P.S))
    return np.concatenate([x, np.broadcast_to(st_np[None], (len(d),) + st_np.shape), g], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ model
class Backbone(nn.Module):
    def __init__(self, cin, d=96, dil=None):
        super().__init__()
        dil = dil or DIL
        self.inp = nn.Conv1d(cin, d, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 5, padding=2 * k, dilation=k), nn.GELU(),
                                                   nn.Conv1d(d, d, 1)) for k in dil])
        self.norms = nn.ModuleList([nn.GroupNorm(1, d) for _ in dil])

    def forward(self, x):
        h = self.inp(x)
        for b, n in zip(self.blocks, self.norms):
            h = h + b(n(h))
        return h


class ConceptNet(nn.Module):
    def __init__(self, cin, d=96, dil=None, nfree=0):
        super().__init__()
        self.bb = Backbone(cin, d, dil)
        self.head = nn.Conv1d(d, NC + nfree, 1)        # note 171 OceanCBM: + unsupervised 'free' concept channels

    def forward(self, x):
        return self.head(self.bb(x))                                # (B, NC, S) raw


def concept_feats(c):
    """raw concept outputs -> the values the onset head sees (probabilities / ratios)."""
    return torch.cat([torch.sigmoid(c[:, :2]), c[:, 2:NC], torch.tanh(c[:, NC:])], 1)


RS = F + 1                             # residual origin: t0 - F - 1 predicts slot t0 - 1 (released, slot T is not)


def resid_obs(X, d, t0):
    """(B, 3, S) observed v/vf, q/q95, ok at slot t0-1 (torch; X (D, T, CH, S))."""
    return X[d, t0 - 1, :3].float()


def residuals(c_prev, obs):
    """today's concept surprise per link: observed - predicted at slot t0-1 (masked where not observed)."""
    ok = obs[:, 2:3]
    return torch.cat([(obs[:, 0:1] - c_prev[:, 2:3]) * ok, (obs[:, 1:2] - c_prev[:, 3:4]) * ok, ok], 1)


class OnsetHead(nn.Module):
    """g: concepts of the link and +-3 neighbours -> onset logit. Tiny on purpose (the decision is in the concepts)."""
    def __init__(self, nc=NC, w=16, resid=None):
        super().__init__()
        nc = nc + (3 if (RESID if resid is None else resid) else 0)
        self.net = nn.Sequential(nn.Conv1d(nc, w, 7, padding=3), nn.GELU(), nn.Conv1d(w, w, 1), nn.GELU(),
                                 nn.Conv1d(w, 1, 1))

    def forward(self, cf):
        return self.net(cf)[:, 0]


class Direct(nn.Module):
    def __init__(self, cin, d=96, dil=None):
        super().__init__()
        self.bb = Backbone(cin, d, dil)
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, x):
        return self.head(self.bb(x))[:, 0]


def onset_loss(logit, y, m):
    bce = (Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
    lsm = torch.log_softmax(logit.masked_fill(m == 0, -1e4), 1)
    w = y * m
    lw = -(w * lsm).sum(1) / w.sum(1).clamp(min=1)
    return bce + 0.5 * lw[w.sum(1) > 0].mean()


def concept_loss(c, y):
    m30, m15 = y["m30"], y["m15"]
    def bce(z, t, m):
        return (Fn.binary_cross_entropy_with_logits(z, t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    def hub(z, t, m):
        return (Fn.smooth_l1_loss(z, t, reduction="none", beta=0.1) * m).sum() / m.sum().clamp(min=1)
    return (bce(c[:, 0], y["q30"], m30) + bce(c[:, 1], y["q15"], m15) + hub(c[:, 2], y["v30"], m30)
            + hub(c[:, 3], y["q30f"], m30) + hub(c[:, 4], y["v15"], m15))


# ------------------------------------------------------------------------------------------------ training
def sample_all(P, bs, rng, part):
    days = np.nonzero(P.part == part)[0]
    d = rng.choice(days, bs)
    t0 = rng.integers(H, P.T - F, bs)
    return torch.from_numpy(d).to(DEV), torch.from_numpy(t0).to(DEV)


def sample_onset(P, bs, rng, part):
    j = np.nonzero(P.part[P.on_d] == part)[0]
    j = rng.choice(j, bs)
    return torch.from_numpy(P.on_d[j]).to(DEV), torch.from_numpy(P.on_t[j]).to(DEV)


def onset_eval(Ps, fn, part, cn=None, g=None):
    """top-1 hit (arg-max link within +-2 of a true T+30 queue cell) on the training months' held-out onset windows."""
    hit = tot = 0
    with torch.no_grad():
        for P in Ps:
            j = np.nonzero(P.part[P.on_d] == part)[0]
            for a in range(0, len(j), 128):
                jj = j[a:a + 128]
                d = torch.from_numpy(P.on_d[jj]).to(DEV); t0 = torch.from_numpy(P.on_t[jj]).to(DEV)
                if cn is not None:
                    cf = torch.cat([concept_feats(cn(inputs(P, d, t0))),
                                    residuals(cn(inputs(P, d, t0 - RS)), resid_obs(P.X, d, t0))], 1)
                    lg = g(cf)
                else:
                    lg = fn(inputs(P, d, t0))
                m = P.ok[d, t0 + F]
                lg = lg.masked_fill(~m, -1e4)
                am = lg.argmax(1)
                y = P.qd[d, t0 + F]
                yn = y.clone()
                for k in (1, 2):
                    yn[:, k:] |= y[:, :-k]; yn[:, :-k] |= y[:, k:]
                hit += int(yn[torch.arange(len(am)), am].sum()); tot += len(am)
    return hit / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cbm", choices=("cbm", "direct", "joint"))
    ap.add_argument("--nfree", type=int, default=1, help="joint: free concept channels (OceanCBM)")
    ap.add_argument("--w-on", type=float, default=0.5, help="joint: weight of the onset loss (mixed supervision)")
    ap.add_argument("--free-l2", type=float, default=1e-3, help="joint: L2 on the free concept (limits leakage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--g-steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cn-from", default="", help="reuse the concept net of this checkpoint; train g only")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    Ps = [Panel(i, p) for i, p in enumerate(T2_PANELS)]
    print(f"data {time.time() - t_start:.0f}s; onset windows " + " ".join(f"{P.panel}:{len(P.on_d)}" for P in Ps),
          flush=True)
    cin = H * CH + len(ST_KEEP) + 3
    name = f"cbm_{a.mode}{a.tag}_s{a.seed}"
    if a.mode == "joint":
        # OceanCBM-style mixed supervision: concepts (all windows) + onset (onset windows) trained together through
        # the bottleneck [prescribed concepts + free concept]; the decision g sees only the bottleneck
        cn = ConceptNet(cin, nfree=a.nfree).to(DEV)
        g = OnsetHead(nc=NC + a.nfree, resid=False).to(DEV)
        params = list(cn.parameters()) + list(g.parameters())
        opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
        best, best_sd = -1, None
        for step in range(a.steps):
            P = Ps[rng.integers(len(Ps))]
            d, t0 = sample_all(P, a.bs, rng, 0)
            c = cn(inputs(P, d, t0))
            loss_c = concept_loss(c, targets(P, d, t0))
            d, t0 = sample_onset(P, a.bs, rng, 0)
            co = cn(inputs(P, d, t0))
            y = P.qd[d, t0 + F].float(); m = P.ok[d, t0 + F].float()
            loss_o = onset_loss(g(concept_feats(co)), y, m)
            reg = (co[:, NC:] ** 2).mean() if a.nfree else 0.0
            loss = loss_c + a.w_on * loss_o + a.free_l2 * reg
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            if step % 2000 == 0 or step == a.steps - 1:
                cn.eval(); g.eval()
                acc = onset_eval(Ps, lambda x: g(concept_feats(cn(x))), 1)
                cn.train(); g.train()
                print(f"  joint step {step} loss_c {loss_c.item():.4f} loss_o {loss_o.item():.4f} top1(es) {acc:.3f}"
                      f"  {time.time() - t_start:.0f}s", flush=True)
                if acc > best and step >= a.steps // 2:
                    best = acc
                    best_sd = ({k: v.clone() for k, v in cn.state_dict().items()},
                               {k: v.clone() for k, v in g.state_dict().items()})
        torch.save(dict(cn=best_sd[0], g=best_sd[1], cin=cin, resid=False, st=ST_KEEP, dil=DIL, nfree=a.nfree),
                   MODELS / f"{name}.pt")
    elif a.mode == "cbm":
        cn = ConceptNet(cin).to(DEV)
        if a.cn_from:
            cn.load_state_dict(torch.load(MODELS / f"{a.cn_from}.pt", map_location=DEV)["cn"])
            a.steps = 0
        opt = torch.optim.AdamW(cn.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=max(a.steps, 1), pct_start=0.05)
        for step in range(a.steps):
            P = Ps[rng.integers(len(Ps))]
            d, t0 = sample_all(P, a.bs, rng, 0)
            loss = concept_loss(cn(inputs(P, d, t0)), targets(P, d, t0))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(cn.parameters(), 1.0); opt.step(); sch.step()
            if step % 2000 == 0 or step == a.steps - 1:
                cn.eval()
                with torch.no_grad():
                    ev = []
                    for P in Ps:
                        r = np.random.default_rng(1)
                        d, t0 = sample_all(P, 256, r, 1)
                        ev.append(float(concept_loss(cn(inputs(P, d, t0)), targets(P, d, t0))))
                    c0 = onset_eval(Ps, lambda x: cn(x)[:, 0], 1)
                cn.train()
                print(f"  concept step {step} loss {loss.item():.4f} es {np.mean(ev):.4f} c0-top1(es onset) {c0:.3f}"
                      f"  {time.time() - t_start:.0f}s", flush=True)
        cn.eval()
        for p_ in cn.parameters():
            p_.requires_grad_(False)
        g = OnsetHead().to(DEV)
        opt = torch.optim.AdamW(g.parameters(), 3e-3, weight_decay=1e-4)
        best, best_sd = -1, None
        for step in range(a.g_steps):
            P = Ps[rng.integers(len(Ps))]
            d, t0 = sample_onset(P, a.bs, rng, 0)
            with torch.no_grad():
                cf = concept_feats(cn(inputs(P, d, t0)))
                if RESID:
                    cf = torch.cat([cf, residuals(cn(inputs(P, d, t0 - RS)), resid_obs(P.X, d, t0))], 1)
            y = P.qd[d, t0 + F].float(); m = P.ok[d, t0 + F].float()
            loss = onset_loss(g(cf), y, m)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0 or step == a.g_steps - 1:
                g.eval()
                acc = onset_eval(Ps, lambda x: g(concept_feats(cn(x))), 1, cn if RESID else None, g)
                g.train()
                print(f"  g step {step} loss {loss.item():.4f} top1(es) {acc:.3f}", flush=True)
                if acc > best:
                    best, best_sd = acc, {k: v.clone() for k, v in g.state_dict().items()}
        g.load_state_dict(best_sd)
        torch.save(dict(cn=cn.state_dict(), g=g.state_dict(), cin=cin, resid=RESID, st=ST_KEEP, dil=DIL), MODELS / f"{name}.pt")
    else:
        net = Direct(cin).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
        best, best_sd = -1, None
        for step in range(a.steps):
            P = Ps[rng.integers(len(Ps))]
            d, t0 = sample_onset(P, a.bs, rng, 0)
            y = P.qd[d, t0 + F].float(); m = P.ok[d, t0 + F].float()
            loss = onset_loss(net(inputs(P, d, t0)), y, m)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            if step % 1000 == 0 or step == a.steps - 1:
                net.eval()
                acc = onset_eval(Ps, net, 1)
                net.train()
                print(f"  direct step {step} loss {loss.item():.4f} top1(es) {acc:.3f}  {time.time() - t_start:.0f}s",
                      flush=True)
                if acc > best:
                    best, best_sd = acc, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_sd)
        torch.save(dict(net=net.state_dict(), cin=cin, st=ST_KEEP, dil=DIL), MODELS / f"{name}.pt")
    print(f"saved {name} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
