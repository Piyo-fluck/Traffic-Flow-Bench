"""Task 1 method M3: fully convolutional inpainting of one day as a (time x station) image.

Input channels (masked layer only, as in validation/private): speed and
flow-per-lane anomalies where visible (0 elsewhere), visibility, occupancy,
the weekday x tod profile, lanes, free speed, time of day. Output: speed and
flow-per-lane anomalies. Loss is on target cells with known truth, plus cells
hidden on the fly from the visible layer (their truth is the observation).

Options (environment variables, all off by default = the v1 model):
  M3_V2=1        add link length and released ramp flows (+validity, attachment) as inputs
  M3_DNLOSS=λ    add λ · (Σ|ΔN error| / Σ|ΔN true|), the Task 3 conservation proxy, to the loss (needs M3_V2)
  M3_EPOCHS, M3_TAG, M3_SEED, M3_CKPT (evaluate a saved checkpoint only)

Days: Jun–Nov 2030 train, Dec 2030 selects the checkpoint, Jan–Feb 2031 test.

    py analysis/t1_m3_cnn.py D12_I5_N D7_I210_W
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import PANELS, T, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.metrics import FLOW_NORM as FLOW_NORM_M, FLOW_W as FLOW_W_M, SPEED_NORM as SPEED_NORM_M  # noqa: E402
from tfb.metrics import SPEED_W as SPEED_W_M, fmt, state_score  # noqa: E402

net_ref: list = [None]
from tfb.physics import fd_params, fmt_phys, load_ramps, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

ES_START = "2030-12-01"
SCALE_V, SCALE_Q = 10.0, 100.0
LOSS_W = tuple(float(v) for v in os.environ.get("M3_LOSSW", "0.75,0.25").split(","))
# speed, flow. The default came from d(S_state)/d(MSE) at RMSE ~3 km/h and ~35 veh/h/lane. S_state is LINEAR in
# RMSE, so the value of a unit of squared error is w_g / (2 RMSE_g): as the model improved on speed (3.0 -> 1.31)
# the metric-optimal split moved to about 0.87 / 0.13 (analysis/t1_metric_alignment.py, note 101).
# M3_GROUPW=1 goes further and weights每 cell by w_g / (n_g RMSE_g) per (panel, regime) group and per channel,
# which is the exact gradient of the official macro score; the group RMSEs are refreshed each epoch from the ES
# days, so the weights lag by one epoch and stay stable.
GROUP_W = os.environ.get("M3_GROUPW", "0") == "1"
CROP_S, BATCH, STEPS_PER_EPOCH, LR = 64, 16, 400, 2e-3
EPOCHS = int(os.environ.get("M3_EPOCHS", 25))
SEED = int(os.environ.get("M3_SEED", 0))
# M3_NDAYS=n: train on only the LAST n fit days per panel, for a learning curve. The optimisation budget is
# unchanged, so the curve shows what data alone buys. 0 = all of them (production).
N_DAYS = int(os.environ.get("M3_NDAYS", 0))
V2 = os.environ.get("M3_V2", "0") == "1"
DN_LAMBDA = float(os.environ.get("M3_DNLOSS", 0))
WIDTH = int(os.environ.get("M3_WIDTH", 64))
# M3_RF: the dilation schedule, at a FIXED block count and width, so the parameter count never changes. The
# production reach is 215 time slots (18 h) and 69 links, against a measured informative window of 13 slots
# (note 01 p99) and no spatial signal at all (note 107), and the day-scale pattern is already supplied by the
# profile channels. "t1" and "all1" ask whether the long reach is dead weight, "wide" asks the opposite.
RF = os.environ.get("M3_RF", "prod")
# M3_KHEAD=λ: third output = per-lane density deviation, trained with λ · length-weighted L1 on density.
# S_LWR's residual is |ΔN| = |Δk|·L at target cells (the organizer rhs equals the truth's own ΔN), so density,
# not speed/flow MSE, is what Task 3 scores. At prediction (v, q) can be projected onto this density.
K_LAMBDA = float(os.environ.get("M3_KHEAD", 0))
# M3_AUXH="l1,l2,l3": auxiliary heads trained from the full truth and THROWN AWAY at inference - the temporal
# difference of flow, its spatial difference along the corridor, and occupancy. The point is not to predict them
# but to make the shared trunk represent level, temporal derivative and spatial derivative separately (note 104).
AUXH = tuple(float(v) for v in os.environ.get("M3_AUXH", "").split(",") if v.strip())
# M3_LWR=lambda: add the exact S_LWR residual as a loss. With e = k_hat - k_true zeroed outside the loss cells,
# the metric's numerator is sum |e[t+1] - e[t]| L over transitions, which is what this computes - not a proxy on
# the density level (M3_KHEAD) and not a conservation identity (M3_DNLOSS). Needs two extra target channels
# (the true density and the link length), which do not touch the inputs, so A's network is unchanged.
LWR_LAMBDA = float(os.environ.get("M3_LWR", 0))
# M3_SCALE=b0: predict the flow deviation RELATIVE to s(base) = max(base_q per lane, b0) instead of the fixed
# SCALE_Q. The loss stays in the original flow space - the difference is multiplied back by s - so the objective
# is numerically identical; only the gradient geometry changes, giving high-flow cells a larger share of the
# network's capacity (note 100: that is where the residual is). M3_SCALECH=1 also feeds the scale explicitly.
SCALE_B0 = float(os.environ.get("M3_SCALE", 0))
SCALE_CH = os.environ.get("M3_SCALECH", "0") == "1"
SCALE_K = 10.0  # veh/km/lane
N_OUT = (3 if K_LAMBDA > 0 else 2) + len(AUXH)
LEN_CH = 10  # first v2 channel: link length in km
DEV = "cuda"
ROOT = Path(__file__).resolve().parent.parent


def make_inputs(z: dict, base_v: np.ndarray, base_q: np.ndarray, extra: list | None = None) -> np.ndarray:
    """(D, C, T, S) float16 network input from the masked layer and the profile (+ optional channels)."""
    lanes = z["lanes"]
    ov, oq, oo = z["obs_speed"], z["obs_flow"], z["obs_occ"]
    vis = np.isfinite(ov) & np.isfinite(oq)
    D, _, S = ov.shape
    ang = 2 * np.pi * np.arange(T) / T
    bc_s = lambda v: np.broadcast_to(np.asarray(v, np.float32)[None, None, :], (D, T, S))  # noqa: E731
    bc_t = lambda v: np.broadcast_to(np.asarray(v, np.float32)[None, :, None], (D, T, S))  # noqa: E731
    return np.stack([
        np.where(vis, (ov - base_v) / SCALE_V, 0),
        np.where(vis, (oq - base_q) / lanes / SCALE_Q, 0),
        vis,
        np.where(vis & np.isfinite(oo), oo * 10, 0),
        base_v / 100, base_q / lanes / 1000,
        bc_s(lanes / 5), bc_s(z["free_speed"] / 100),
        bc_t(np.sin(ang)), bc_t(np.cos(ang)),
    ] + list(extra or []), axis=1).astype(np.float16)


def scale_channels(z: dict, base_q: np.ndarray) -> list[np.ndarray]:
    """Explicit scale information: the profile's level and the visible flow level, both in log space."""
    lanes = z["lanes"]
    oq = z["obs_flow"]
    vis = np.isfinite(z["obs_speed"]) & np.isfinite(oq)
    return [np.log1p(np.maximum(base_q / lanes, 0)) / 8.0,
            np.where(vis, np.log1p(np.maximum(oq / lanes, 0)) / 8.0, 0)]


def v2_channels(panel: str, split: str, z: dict) -> list[np.ndarray]:
    """Link length and the released ramp flows, per lane, where valid."""
    r = load_ramps(panel, split, z)
    prm = fd_params(panel, z["link_ids"])
    lanes = z["lanes"]
    D, _, S = z["obs_speed"].shape
    bc_s = lambda v: np.broadcast_to(np.asarray(v, np.float32)[None, None, :], (D, T, S))  # noqa: E731
    return [bc_s(prm["length"]),
            np.where(r["on_v"], r["on_q"] / lanes / 1000, 0), r["on_v"],
            np.where(r["off_v"], r["off_q"] / lanes / 1000, 0), r["off_v"],
            bc_s(r["on_att"]), bc_s(r["off_att"])]


def panel_tensors(panel: str) -> dict:
    z = load(panel, "train")
    fit, test = split_days(z)
    es = z["dates"] >= ES_START
    ts, tq, lanes = z["true_speed"], z["true_flow"], z["lanes"]
    valid = truth_valid(z)
    base_v = profile(ts, valid, z["weekday"], fit)
    base_q = profile(tq, valid, z["weekday"], fit)
    extra = list(v2_channels(panel, "train", z)) if V2 else []
    if SCALE_CH:
        extra += scale_channels(z, base_q)
    X = make_inputs(z, base_v, base_q, extra or None)
    q_scale = np.maximum(base_q / lanes, SCALE_B0) if SCALE_B0 > 0 else None
    chans = [np.nan_to_num((ts - base_v) / SCALE_V),
             np.nan_to_num((tq - base_q) / lanes / (q_scale if q_scale is not None else SCALE_Q))]
    if K_LAMBDA > 0:  # density target (per lane, as a deviation from the profile) and its S_LWR weight lanes * L
        length = fd_params(panel, z["link_ids"])["length"]
        k_true = tq / lanes / np.maximum(ts, 1.0)
        k_base = base_q / lanes / np.maximum(base_v, 1.0)
        w = lanes * length
        chans += [np.nan_to_num((k_true - k_base) / SCALE_K), np.broadcast_to((w / w.mean())[None, None, :], ts.shape)]
    if LWR_LAMBDA > 0:          # target-only channels: the true density and the link length, for the S_LWR loss
        length = fd_params(panel, z["link_ids"])["length"]
        chans += [np.nan_to_num(tq / np.maximum(ts, 1.0) / 50.0).astype(np.float32),
                  np.broadcast_to((length / length.mean())[None, None, :], ts.shape).astype(np.float32)]
    if AUXH:
        assert K_LAMBDA == 0, "M3_AUXH and M3_KHEAD both use the extra output channels"
        ql = np.nan_to_num(tq / lanes, nan=0.0)
        dt = np.zeros_like(ql); dt[:, 1:-1] = ql[:, 2:] - ql[:, :-2]
        dx = np.zeros_like(ql); dx[:, :, 1:-1] = ql[:, :, 2:] - ql[:, :, :-2]
        # each auxiliary target is scaled to about unit standard deviation, measured on the train truth, so that
        # lambda really is the weight (the first attempt used /SCALE_Q and x10, which left them 5x the main
        # targets and therefore dominating the loss - note 104)
        aux = [dt / 500.0, dx / 500.0, np.nan_to_num(z["true_occ"], nan=0.0) * 7.5]
        chans += [a.astype(np.float32) for a in aux[:len(AUXH)]]
    Y = np.stack(chans, axis=1).astype(np.float16)
    M = z["target"] & np.isfinite(ts) & np.isfinite(tq)
    prm, ok = fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z)
    for k in ("obs_speed", "obs_flow", "obs_occ", "pct", "true_occ"):
        z.pop(k, None)
    return dict(panel=panel, z=z, X=X, Y=Y, M=M, E=z["elig"], base_v=base_v, base_q=base_q, prm=prm, ok=ok,
                q_scale=q_scale, regime=np.asarray(z["regime"], str),
                tr_days=(np.flatnonzero(fit & ~es)[-N_DAYS:] if N_DAYS else np.flatnonzero(fit & ~es)),
                es_days=np.flatnonzero(fit & es), te_days=np.flatnonzero(test))


class Block(nn.Module):
    def __init__(self, c: int, dt: int, ds: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=(dt, ds), dilation=(dt, ds)),
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


PROD_DIL = [(1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (32, 1), (1, 1), (2, 2), (4, 4), (8, 1), (16, 2), (1, 1)]
DILS = {
    "prod": PROD_DIL,
    "t1": [(1, ds) for _, ds in PROD_DIL],                       # time reach 215 -> 49 slots, space unchanged
    "all1": [(1, 1) for _ in PROD_DIL],                          # both at the 12-block minimum, 49 x 49
    "wide": [(1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (32, 4),   # time 215 -> 403 slots, space 69 -> 77 links
             (64, 2), (2, 1), (4, 2), (8, 4), (16, 2), (32, 1)],
}


class DilNet(nn.Module):
    DIL = DILS[RF]

    def __init__(self, cin: int, c: int = 64):
        super().__init__()
        self.inp = nn.Conv2d(cin, c, 1)
        self.blocks = nn.Sequential(*[Block(c, dt, ds) for dt, ds in self.DIL])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, N_OUT, 1))  # 2, or 3 with the density head

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


# M3_ARCH=unet: a multi-scale encoder-decoder (note 125; the supervised upper-bound model class of Gan et al.,
# Physica A 2026, and the NAFNet / MaIR restoration backbones). Three levels, halving time and links and doubling
# the width at each, residual blocks at every level, skip connections. With c = M3_WIDTH // 2 the parameter count
# is close to the width-192 dilated net, so the comparison is about multi-scale structure, not size.
ARCH = os.environ.get("M3_ARCH", "dil")
UNET_BLOCKS = int(os.environ.get("M3_UNET_BLOCKS", 2))


class UNet(nn.Module):
    def __init__(self, cin: int, c: int = 64):
        super().__init__()
        c = max(8, (c // 2) // 8 * 8)
        ch = [c, 2 * c, 4 * c]
        nb = UNET_BLOCKS
        self.inp = nn.Conv2d(cin, ch[0], 1)
        self.enc = nn.ModuleList([nn.Sequential(*[Block(w, 1, 1) for _ in range(nb)]) for w in ch])
        self.down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 2, stride=2) for i in range(2)])
        self.mid = nn.Sequential(*[Block(ch[2], 1, 1) for _ in range(nb)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(2)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * ch[i], ch[i], 1) for i in range(2)])
        self.dec = nn.ModuleList([nn.Sequential(*[Block(ch[i], 1, 1) for _ in range(nb)]) for i in range(2)])
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.GELU(), nn.Conv2d(ch[0], N_OUT, 1))

    def forward(self, x):
        T, S = x.shape[-2:]
        pt, ps = (-T) % 4, (-S) % 4                       # both axes must halve twice
        if pt or ps:
            x = nn.functional.pad(x, (0, ps, 0, pt))
        h0 = self.enc[0](self.inp(x))
        h1 = self.enc[1](self.down[0](h0))
        h2 = self.mid(self.enc[2](self.down[1](h1)))
        d1 = self.dec[1](self.fuse[1](torch.cat([self.up[1](h2), h1], 1)))
        d0 = self.dec[0](self.fuse[0](torch.cat([self.up[0](d1), h0], 1)))
        return self.out(d0)[..., :T, :S]


Net = UNet if ARCH == "unet" else DilNet


def sample_batch(parts, rng, wtab=None):
    xs, ys, ms, es, wv, wq = [], [], [], [], [], []
    for _ in range(BATCH):
        p = parts[rng.integers(len(parts))]
        d = p["tr_days"][rng.integers(len(p["tr_days"]))]
        if wtab is not None:
            a, b = wtab.get((p["panel"], p["regime"][d]), (1.0, 1.0))
            wv.append(a); wq.append(b)
        S = p["X"].shape[-1]
        s0 = rng.integers(0, S - CROP_S + 1) if S > CROP_S else 0
        sl = slice(s0, s0 + CROP_S)
        x, y, m, e = p["X"][d, :, :, sl], p["Y"][d, :, :, sl], p["M"][d, :, sl], p["E"][d, :, sl]
        pad = CROP_S - x.shape[-1]
        if pad > 0:
            x, y = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, ((0, 0), (0, 0), (0, pad)))
            m, e = np.pad(m, ((0, 0), (0, pad))), np.pad(e, ((0, 0), (0, pad)))
        xs.append(x); ys.append(y); ms.append(m); es.append(e)
    to = lambda a, dt: torch.from_numpy(np.stack(a)).to(DEV, dt)  # noqa: E731
    w = None if wtab is None else (to(wv, torch.float32), to(wq, torch.float32))
    return to(xs, torch.float32), to(ys, torch.float32), to(ms, torch.bool), to(es, torch.bool), w


def hide_extra(x, m, e, gen):
    """Hide a random share of visible eligible cells; they join the loss with the observation as truth."""
    B = x.shape[0]
    rate = torch.rand(B, 1, 1, device=DEV, generator=gen) * 0.3 * (torch.rand(B, 1, 1, device=DEV, generator=gen) < 0.5)
    hide = (x[:, 2] > 0.5) & e & (torch.rand(x[:, 2].shape, device=DEV, generator=gen) < rate)
    keep = (~hide).unsqueeze(1).float()
    x = x.clone()
    x[:, 0:4] = x[:, 0:4] * keep
    return x, m | hide


def flow_scale(x):
    """s(base) / SCALE_Q, from the base-flow channel, so the flow loss stays in the original units."""
    if SCALE_B0 <= 0:
        return 1.0
    return torch.clamp(x[:, 5] * 1000.0, min=SCALE_B0) / SCALE_Q


def loss_fn(out, y, m, x=None, w=None):
    mf = m.float()
    n = mf.sum().clamp(min=1)
    dv = out[:, 0] - y[:, 0]
    dq = out[:, 1] - y[:, 1]
    if SCALE_B0 > 0 and x is not None:
        dq = dq * flow_scale(x)          # back to (q - q_hat) / lanes / SCALE_Q, exactly the additive objective
    if w is not None:                    # the official score's own per-cell weights, by (panel, regime)
        wv, wq = w[0][:, None, None], w[1][:, None, None]
        return (((dv ** 2) * wv + (dq ** 2) * wq) * mf).sum() / n
    lv = ((dv ** 2) * mf).sum() / n
    lq = ((dq ** 2) * mf).sum() / n
    loss = LOSS_W[0] * lv + LOSS_W[1] * lq
    if LWR_LAMBDA > 0 and x is not None:
        # the metric's own numerator: |e[t+1] - e[t]| L, with e = k_hat - k_true only at the loss cells
        v_hat = (x[:, 4] * 100 + out[:, 0] * SCALE_V).clamp(min=1.0)
        q_lane = x[:, 5] * 1000 + out[:, 1] * SCALE_Q
        lanes = x[:, 6] * 5
        e = (q_lane * lanes / v_hat - y[:, 2] * 50.0) * mf
        w = y[:, 3][:, 1:]
        d = ((e[:, 1:] - e[:, :-1]).abs() * w).sum()
        loss = loss + LWR_LAMBDA * d / (mf[:, 1:] + mf[:, :-1]).clamp(max=1.0).sum().clamp(min=1)
    for j, lam in enumerate(AUXH):       # auxiliary heads: trained here, discarded at inference
        loss = loss + lam * (((out[:, 2 + j] - y[:, 2 + j]) ** 2) * mf).sum() / n
    return loss


def dn_loss(out, x, y, m, e):
    """Σ|ΔN error| / Σ|ΔN true| over time transitions between cells with known truth (the Task 3 proxy).

    The state takes the truth at released cells, so only predicted cells (m) carry an error.
    """
    base_v, base_ql, lanes, L = x[:, 4] * 100, x[:, 5] * 1000, x[:, 6] * 5, x[:, LEN_CH]
    k_hat = (base_ql + out[:, 1] * SCALE_Q) * lanes / (base_v + out[:, 0] * SCALE_V).clamp(min=1)
    k_true = (base_ql + y[:, 1] * SCALE_Q) * lanes / (base_v + y[:, 0] * SCALE_V).clamp(min=1)
    known = m | ((x[:, 2] > 0.5) & e)
    err = torch.where(m, (k_hat - k_true) * L, torch.zeros_like(k_true))
    tr = (known[:, 1:] & known[:, :-1]).float()
    num = ((err[:, 1:] - err[:, :-1]).abs() * tr).sum()
    den = (((k_true[:, 1:] - k_true[:, :-1]) * L[:, 1:]).abs() * tr).sum().clamp(min=1e-6)
    return num / den


def k_loss(out, y, m):
    """Length x lanes weighted L1 on the per-lane density deviation at target cells (S_LWR's edge residual)."""
    mf = m.float()
    return ((out[:, 2] - y[:, 2]).abs() * y[:, 3] * mf).sum() / mf.sum().clamp(min=1)


@torch.no_grad()
def predict_days(net, p, days):
    pv, pq = p["base_v"].copy(), p["base_q"].copy()
    lanes = p["z"]["lanes"]
    for d in days:
        x = torch.from_numpy(p["X"][d:d + 1].astype(np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        out = out.float()[0].cpu().numpy()
        pv[d] = p["base_v"][d] + out[0] * SCALE_V
        sq = p["q_scale"][d] if p.get("q_scale") is not None else SCALE_Q
        pq[d] = p["base_q"][d] + out[1] * sq * lanes
    return pv, pq


def group_weights(parts, key="es_days"):
    """w_g / (n_g RMSE_g) per (panel, regime) and channel, normalised so the mean cell weight is LOSS_W."""
    tab, stats = {}, {}
    for p in parts:
        pv, pq = predict_days(net_ref[0], p, p[key])
        z, lanes = p["z"], p["z"]["lanes"]
        ts, tq = z["true_speed"], z["true_flow"]
        for d in p[key]:
            g = (p["panel"], p["regime"][d])
            m = p["M"][d] & np.isfinite(ts[d]) & np.isfinite(tq[d])
            if not m.any():
                continue
            a = stats.setdefault(g, [0.0, 0.0, 0])
            a[0] += float(((pv[d] - ts[d])[m] ** 2).sum())
            a[1] += float((((pq[d] - tq[d]) / np.broadcast_to(lanes, m.shape))[m] ** 2).sum())
            a[2] += int(m.sum())
    n_tr = {}
    for p in parts:
        for d in p["tr_days"]:
            g = (p["panel"], p["regime"][d])
            n_tr[g] = n_tr.get(g, 0) + int(p["M"][d].sum())
    av, aq = SPEED_W_M / SPEED_NORM_M, FLOW_W_M / FLOW_NORM_M
    raw = {}
    for g, a in stats.items():
        rv, rq = np.sqrt(a[0] / a[2]), np.sqrt(a[1] / a[2])
        n = max(n_tr.get(g, a[2]), 1)
        raw[g] = (av / (2 * max(rv, 1e-6) * n) * SCALE_V ** 2, aq / (2 * max(rq, 1e-6) * n) * SCALE_Q ** 2)
    tot = sum(n_tr.get(g, 1) * (raw[g][0] + raw[g][1]) for g in raw)
    n_all = sum(n_tr.get(g, 1) for g in raw)
    for g, (u, v) in raw.items():
        tab[g] = (u * n_all / tot, v * n_all / tot)
    return tab


def evaluate(net, parts, key):
    net.eval()
    scores, rows = {}, {}
    for p in parts:
        days = p[key]
        pv, pq = predict_days(net, p, days)
        cells = np.zeros(len(p["z"]["dates"]), bool)
        cells[days] = True
        r = state_score(p["z"], pv, pq, cells=cells[:, None, None])
        scores[p["panel"]], rows[p["panel"]] = r["mean"]["S"], r
    net.train()
    return family_mean(scores), rows


def report_test(net, parts, tag: str) -> None:
    """S_state and the Task 3 proxy on the test days; saves the test predictions."""
    preds_dir = ROOT / "work" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    net.eval()
    st, ph = {}, {}
    for p in parts:
        days = p["te_days"]
        pv, pq = predict_days(net, p, days)
        mask = np.zeros(len(p["z"]["dates"]), bool)
        mask[days] = True
        r = state_score(p["z"], pv, pq, cells=mask[:, None, None])
        f = physics_scores(p["z"], pv, pq, p["prm"], p["ok"], days=mask)
        st[p["panel"]], ph[p["panel"]] = r["mean"]["S"], f["mean"]["S_phys"]
        print(f"[{p['panel']}] {tag}  {fmt(r)}\n    {fmt_phys(f)}")
        np.savez(preds_dir / f"t1_{tag}_{p['panel']}_test.npz", days=days, pv=pv[days], pq=pq[days])
    s, f = family_mean(st), family_mean(ph)
    print(f"test family mean [{tag}]: S_state {s:.4f} | S_phys~ {f:.4f} | 0.35*S_state + 0.15*S_phys~ = {0.35 * s + 0.15 * f:.4f}")


def main() -> None:
    panels = sys.argv[1:] or PANELS
    if DN_LAMBDA > 0 and not V2:
        raise SystemExit("M3_DNLOSS needs M3_V2=1 (the loss reads link length from the v2 channels)")
    tag = os.environ.get("M3_TAG", "m3")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    t0 = time.time()
    parts = [panel_tensors(p) for p in panels]
    print(f"loaded {len(parts)} panels in {time.time() - t0:.0f}s; channels {parts[0]['X'].shape[1]}; "
          f"v2={V2} dn_lambda={DN_LAMBDA} seed={SEED}", flush=True)
    net = Net(parts[0]["X"].shape[1], WIDTH).to(DEV)
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M (width {WIDTH})", flush=True)
    if os.environ.get("M3_CKPT"):
        # Evaluate a saved checkpoint only: no training.
        net.load_state_dict(torch.load(os.environ["M3_CKPT"], map_location=DEV))
        report_test(net, parts, tag)
        return
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start=0.1)
    best, best_state = -1.0, None
    wtab = None
    net_ref[0] = net
    ckpt = ROOT / "work" / "models"
    ckpt.mkdir(parents=True, exist_ok=True)
    for ep in range(EPOCHS):
        t1, tot, tot_dn = time.time(), 0.0, 0.0
        for _ in range(STEPS_PER_EPOCH):
            x, y, m, e, w = sample_batch(parts, rng, wtab)
            x, m = hide_extra(x, m, e, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            outf = out.float()
            loss = loss_fn(outf, y, m, x, w)
            if DN_LAMBDA > 0:
                dn = dn_loss(outf, x, y, m, e)
                loss = loss + DN_LAMBDA * dn
                tot_dn += dn.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        es_score, _ = evaluate(net, parts, "es_days")
        if GROUP_W:
            wtab = group_weights(parts)
        flag = ""
        if es_score > best:
            best, best_state, flag = es_score, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
        extra = f" dN {tot_dn / STEPS_PER_EPOCH:.3f}" if DN_LAMBDA > 0 else ""
        print(f"epoch {ep + 1:>2}: loss {tot / STEPS_PER_EPOCH:.4f}{extra}  ES S_state {es_score:.4f}  ({time.time() - t1:.0f}s){flag}", flush=True)
    net.load_state_dict(best_state)
    name = "_".join(panels) if len(panels) <= 2 else "all"
    torch.save(best_state, ckpt / f"t1_{tag}_{name}.pt")
    print(f"best ES {best:.4f}")
    report_test(net, parts, tag)


if __name__ == "__main__":
    main()
