"""Task 2: a second model family, a spatio-temporal CNN, and its blend with the LightGBM models.

Idea #4 of note 30 (diverse-model ensembles). Task 1's CNN was far better than the tabular models, so the Task 2
history is also treated as an image: 12 history slots x links.
  input channels   v/vf, q/cap, observed flag, queued flag (v <= 0.6 vf), each masked where unseen; static per
                   link (lanes/5, cap per lane/2500, vf/120, on-ramp / off-ramp attached); time of day (sin, cos)
                   and a weekend flag
  output           6 horizon steps x links: queue logits
  network          1x1 input conv, residual blocks with 3x3 convs dilated along links (1..16), then a (12 x 1)
                   conv that collapses the history, then 1x1 to 6 outputs. Fully convolutional over links, so one
                   model serves every panel.
  training         train windows (every earliest-onset window + up to 3,000 ongoing candidates per panel; labels
                   = the queue flags of the truth). BCE masked by label eligibility, 64-link crops, AdamW, and early
                   stopping on 2031-01/02 windows. No recurrence priors (run-shift robust by construction).
Evaluation reuses the blank-aware official-like windows and the cached LightGBM probabilities
(analysis/t2_decision_variants.py, work/cache/t2probs_*). The windows are rebuilt with the same functions, in the
same order. Compared: LightGBM alone (K's rules), CNN alone, and their mean. Onset uses the expected-IoU ratio rule,
ongoing uses tau 0.65.

    py analysis/t2_cnn.py            (env T2CNN_STEPS, T2CNN_WIDTH, T2CNN_DEVICE)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import table_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, T, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402

STEPS = int(os.environ.get("T2CNN_STEPS", 6000))
# T2CNN_PRIORS=1: add a link-position channel and the recurrence priors (P_q at T+5..T+30, P_on at T+30) as extra
# channels, broadcast over the history. Priors are zeroed for a random half of the training windows (the prior
# dropout that made LightGBM's onset robust to the run shift, note 20). At scoring: train-table priors, as in K.
PRIORS = os.environ.get("T2CNN_PRIORS", "0") == "1"
N_PRI = 7  # 6 P_q channels + 1 P_on channel (the position channel precedes them)
# T2CNN_LOVASZ=w: add w x the Lovasz hinge (a tractable surrogate of the per-window IoU; Berman et al. 2018) to BCE.
LOVASZ = float(os.environ.get("T2CNN_LOVASZ", 0))
WIDTH = int(os.environ.get("T2CNN_WIDTH", 64))
DEV = os.environ.get("T2CNN_DEVICE", "cuda")
CROP, BATCH, MAX_OG = 64, 32, 3000
DAY_SETS = {"d1-14": 14, "all": 31}
SEED = int(os.environ.get("T2CNN_SEED", 0))
EVAL_ONLY = os.environ.get("T2CNN_EVAL_ONLY", "")  # path to saved weights: skip training, only score (and cache probs)
# Winners' tricks (note 42), all off by default:
# T2CNN_AUX=w: multi-task head regressing the future v/vf at T+5..T+30 (masked Huber, weight w)
AUX = float(os.environ.get("T2CNN_AUX", 0))
# T2CNN_EMBED=k: learned link-identity embedding of size k (STID / T4c node embeddings). It is zeroed for a random
# T2CNN_EMB_DROP share of the training windows, the same robustness trick as the prior dropout (note 20).
EMBED = int(os.environ.get("T2CNN_EMBED", 0))
EMB_DROP = float(os.environ.get("T2CNN_EMB_DROP", 0.5))
# T2CNN_EMA=decay: exponential moving average of the weights, used for early stopping, saving and scoring
EMA = float(os.environ.get("T2CNN_EMA", 0))
# T2CNN_GLOBAL=1: add the 12 panel-wide state features of analysis/t2_gstate.py (note 43) as channels broadcast over
# links and history. The crops hide the rest of the panel from the CNN; this gives it the global view back.
GLOBAL = os.environ.get("T2CNN_GLOBAL", "0") == "1"
# T2CNN_ARCH=tfm: a third model family (note 42, item 9). A GRU encodes each link's 12-slot history into a token, then
# a transformer attends across ALL links of the panel with a learned relative-position bias (up- vs downstream), so it
# is trained on whole panels instead of 64-link crops. Same inputs, labels, losses and caches as the CNN.
ARCH = os.environ.get("T2CNN_ARCH", "cnn")
# Onset-focused training of the transformer (tfm only; note 46). Only ~290 of ~3,300 training windows per panel are onset
# windows, and their loss is mostly spent on T+5..T+25 cells that are empty by definition.
#   T2CNN_ON_SHARE=s   draw a window from the onset windows with probability s (default: uniform over all windows)
#   T2CNN_LISTWISE=w   add w x a listwise loss at T+30 on onset windows: softmax over the window's eligible links against
#                      the normalised true set ("which link breaks down first", the competing-risk view of note 44)
#   T2CNN_INIT=path    start from saved weights (fine-tuning); T2CNN_ONSET_ONLY=1 trains on onset windows only
#   T2CNN_LR=lr        peak learning rate (default 1e-3)
ON_SHARE = float(os.environ.get("T2CNN_ON_SHARE", 0))
LISTWISE = float(os.environ.get("T2CNN_LISTWISE", 0))
INIT = os.environ.get("T2CNN_INIT", "")
T1INIT = os.environ.get("T2CNN_T1INIT", "")
FUTURE = os.environ.get("T2CNN_FUTURE", "0") == "1"      # teacher input: the horizon flow as extra channels
TEACHER = os.environ.get("T2CNN_TEACHER", "")            # a checkpoint trained with FUTURE=1
DISTILL = float(os.environ.get("T2CNN_DISTILL", 0))      # weight of the soft-target loss
CONTRAST = int(os.environ.get("T2CNN_CONTRAST", 0))      # steps of history/future contrastive pretraining
REC = float(os.environ.get("T2CNN_REC", 0))              # idea 9/10: self-supervised reconstruction head
PL = float(os.environ.get("T2CNN_PL", 0))                # idea 10: Task-4 style path-load head  # Task 1 checkpoint: copy its residual trunk into T2Net.blocks
ONSET_ONLY = os.environ.get("T2CNN_ONSET_ONLY", "0") == "1"
LR = float(os.environ.get("T2CNN_LR", 1e-3))
# T2CNN_SSL=n: note 119's mechanism on Task 2. Before the supervised stage, n steps of masked reconstruction on
# history windows drawn from EVERY fit-day slot of the train truth (not only the labelled windows), with the hidden
# share drawn per window from U(T2CNN_SSL_LO, T2CNN_SSL_HI). It trains the same rec head the scored-month adaptation
# uses, so the backbone has solved the adaptation task before it is ever asked to adapt. Needs T2CNN_REC > 0.
SSL = int(os.environ.get("T2CNN_SSL", 0))
SSL_LO = float(os.environ.get("T2CNN_SSL_LO", 0.10))
SSL_HI = float(os.environ.get("T2CNN_SSL_HI", 0.70))
ONSET_TAG = (f"_on{ON_SHARE:g}" if ON_SHARE else "") + (f"_lw{LISTWISE:g}" if LISTWISE else "") + \
            ("_ft" if INIT else "") + ("_t1i" if T1INIT else "") + ("_fut" if FUTURE else "") + ("_dis" if DISTILL else "") + ("_con" if CONTRAST else "") + ("_rec" if REC else "") + ("_pl" if PL else "") + ("_ononly" if ONSET_ONLY else "") + (f"_ssl{SSL // 1000}k" if SSL else "")
TAG = "t2_cnn" + ("_pri" if PRIORS else "") + ("_lov" if LOVASZ > 0 else "") + ("_aux" if AUX > 0 else "") + \
      (f"_emb{EMBED}" if EMBED else "") + ("_ema" if EMA > 0 else "") + ("_glob" if GLOBAL else "") + \
      ("_tfm" if ARCH == "tfm" else "") + (f"_w{WIDTH}" if WIDTH != 64 else "") + ONSET_TAG + (f"_s{SEED}" if SEED else "")
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- inputs
def static_channels(ctx, ramps) -> np.ndarray:
    lanes = np.asarray(ctx.lanes, np.float32)
    return np.stack([lanes / 5, np.asarray(ctx.cap, np.float32) / lanes / 2500, np.asarray(ctx.vf, np.float32) / 120,
                     ramps["on_att"].astype(np.float32), ramps["off_att"].astype(np.float32)])   # (5, S)


def prior_extra(S: int, n: int, t0: np.ndarray, pri_fn) -> np.ndarray:
    """(n, 1 + N_PRI, S): link position, P_q at T+5..T+30, P_on at T+30. pri_fn(name, t) -> (n, S)."""
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], (n, 1, S))
    pq = np.stack([pri_fn("q", t0 + h) for h in range(1, F + 1)], 1)
    pon = pri_fn("on", t0 + F)[:, None]
    return np.concatenate([pos, pq, pon], 1).astype(np.float32)


def score_extra(ctx, z, d, t0):
    """Scoring-time position + train-table priors (as K uses), or None without T2CNN_PRIORS."""
    if not PRIORS:
        return None
    key = np.stack([d, (z["weekday"][d] >= 5).astype(int)], 1)
    pri = table_prior(ctx)
    return prior_extra(ctx.S, len(d), t0, lambda name, t: pri(name, key, t))


def global_extra(ctx, v, qf, ok) -> np.ndarray:
    """(N, 12, S): the panel-wide state of analysis/t2_gstate.py (note 43), with its saved per-panel PCA."""
    from t2_gstate import gstate
    if not hasattr(ctx, "gpca"):
        pz = np.load(ROOT / "work" / "models_gstate" / f"pca_{ctx.panel}.npz")
        ctx.gpca = (pz["mu"], pz["vt"])
    g = gstate(ctx, ctx.gpca, v, qf, ok)
    g[:, 6] /= 10.0  # the number of queue heads, brought to the scale of the other features
    return np.ascontiguousarray(np.broadcast_to(g[:, :, None], g.shape + (v.shape[-1],)), np.float32)


def make_x(ctx, stat, v, qf, ok, t0, weekend, extra=None) -> np.ndarray:
    """(N, H, S) history -> (N, C, H, S) float16. extra (N, K, S) is broadcast over the history and appended."""
    N, _, S = v.shape
    vr = np.where(ok, v / ctx.vf, 0.0)
    qc = np.where(ok, qf / ctx.cap, 0.0)
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    ang = 2 * np.pi * t0 / T
    time_ch = np.stack([np.sin(ang), np.cos(ang), weekend.astype(float)], 1)[:, :, None, None]
    parts = [np.stack([vr, qc, ok, qd], 1),
             np.broadcast_to(stat[None, :, None, :], (N, stat.shape[0], H, S)),
             np.broadcast_to(time_ch, (N, 3, H, S))]
    if extra is not None:
        parts.append(np.broadcast_to(extra[:, :, None, :], (N, extra.shape[1], H, S)))
    return np.concatenate(parts, 1).astype(np.float16)


# ---------------------------------------------------------------- network
class Block(nn.Module):
    def __init__(self, c, ds):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=(1, ds), dilation=(1, ds)),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class T2Net(nn.Module):
    """Defaults reproduce the production net. n_links/emb add a link-identity embedding (the last id is the zero
    'unknown link' used by embedding dropout and padding); aux adds F v/vf regression outputs after the F logits."""
    def __init__(self, cin, c=64, n_links=0, emb=0, aux=False):
        super().__init__()
        self.emb = nn.Embedding(n_links + 1, emb, padding_idx=n_links) if emb else None
        self.inp = nn.Conv2d(cin + emb, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 1, 2, 4)])
        self.collapse = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, (H, 1)))
        self.head = nn.Sequential(nn.GELU(), nn.Conv2d(c, F * (2 if aux else 1), 1))
        self.rec = nn.Conv2d(c, 2, 1) if REC > 0 else None          # idea 9/10: reconstruct the history state
        self.pl = nn.Conv2d(c, 1, 1) if PL > 0 else None            # idea 10: Task-4 path load per link

    def feats(self, x, lid=None):                           # -> pooled features (B, c, S), before the head
        if self.emb is not None:
            e = self.emb(lid).permute(0, 2, 1)[:, :, None, :].expand(-1, -1, x.shape[2], -1)
            x = torch.cat([x, e.to(x.dtype)], 1)
        return self.collapse(self.blocks(self.inp(x)))[:, :, 0]

    def forward(self, x, lid=None):                         # (B, C, H, S) -> (B, F or 2F, S)
        return self.head(self.feats(x, lid)[:, :, None])[:, :, 0]


class RelMHA(nn.Module):
    """Multi-head self-attention over links with a learned bias per head and signed link distance (clipped at K)."""
    def __init__(self, d, heads=4, K=32):
        super().__init__()
        self.h, self.K = heads, K
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.table = nn.Parameter(torch.zeros(heads, 2 * K + 1))

    def forward(self, x):                                   # (B, S, d)
        B, S, d = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)   # each (B, h, S, dh)
        idx = torch.arange(S, device=x.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-self.K, self.K) + self.K
        att = (q @ k.transpose(-1, -2)) / (d // self.h) ** 0.5 + self.table[:, rel][None].to(q.dtype)
        y = torch.softmax(att.float(), -1).to(v.dtype) @ v
        return self.o(y.transpose(1, 2).reshape(B, S, d))


class TfmBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RelMHA(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.mlp(self.n2(x))


class T2Tfm(nn.Module):
    """GRU over each link's history -> transformer across the panel's links -> F queue logits per link.
    Same call signature as T2Net (the identity embedding and the aux head are not supported here)."""
    def __init__(self, cin, c=64, n_links=0, emb=0, aux=False, layers=4):
        super().__init__()
        assert not emb and not aux, "T2Tfm: T2CNN_EMBED / T2CNN_AUX are CNN-only"
        self.gru = nn.GRU(cin, c, batch_first=True)
        self.blocks = nn.Sequential(*[TfmBlock(c) for _ in range(layers)])
        self.norm = nn.LayerNorm(c)
        self.head = nn.Linear(c, F)

    def forward(self, x, lid=None):                         # (B, C, H, S) -> (B, F, S)
        B, C, Hh, S = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(B * S, Hh, C).float()
        with torch.autocast("cuda", enabled=False):
            _, hN = self.gru(seq)
        tok = hN[-1].view(B, S, -1)
        return self.head(self.norm(self.blocks(tok))).permute(0, 2, 1)


# ---------------------------------------------------------------- data
def path_load(ctx, qf, ok, t0, wd):
    """(N, S) projected path load over capacity: fit path flows to the last three history slots of observed link
    flow, then map back through the incidence. One factorisation per panel (note 60)."""
    import pandas as pd
    from scipy.linalg import cho_factor, cho_solve
    from tfb.data import RELEASE
    if not hasattr(ctx, "_pl"):
        net = RELEASE / "corridors" / ctx.panel / "network"
        inc = pd.read_csv(net / "path_link_incidence.csv", dtype={"link_id": str, "path_id": str})
        paths = pd.read_csv(net / "path_set.csv", dtype={"path_id": str})
        pid = {p: i for i, p in enumerate(paths.path_id)}
        lid = {str(l): i for i, l in enumerate(ctx.z["link_ids"])}
        m = inc.link_id.map(lid)
        keep_rows = m.notna()
        A_full = np.zeros((len(lid), len(pid)), np.float64)
        A_full[m[keep_rows].to_numpy(int), inc.path_id[keep_rows].map(pid).to_numpy(int)] = 1.0
        keep = A_full.any(1)
        A = A_full[keep]
        ctx._pl = (A_full, keep, cho_factor(A @ A.T + 0.05 * np.eye(A.shape[0])))
    A_full, keep, cf = ctx._pl
    A = A_full[keep]
    with np.errstate(all="ignore"):
        q3 = np.nan_to_num(np.nanmean(np.where(ok[:, -3:], qf[:, -3:], np.nan), 1), nan=0.0)
    x = np.maximum(A.T @ cho_solve(cf, q3[:, keep].T), 0.0)
    return ((A_full @ x).T / np.maximum(ctx.cap[None, :], 1.0)).astype(np.float32)


def train_windows(ctx, stat):
    z = ctx.z
    d_on, t_on = ctx.eo
    c = candidates(z)
    d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    if len(d_og) > MAX_OG:
        sel = rng.choice(len(d_og), MAX_OG, replace=False)
        d_og, t_og = d_og[sel], t_og[sel]
    if ONSET_ONLY:
        d_og, t_og = d_og[:0], t_og[:0]
    d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
    is_on = np.r_[np.ones(len(d_on), bool), np.zeros(len(d_og), bool)]
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    v, qf, ok = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi], ctx.el[d[:, None], hi]
    extra = None
    if PRIORS:  # leave-one-day-out train priors for train windows (fit days only, as ctx.prior does)
        extra = prior_extra(ctx.S, len(d), t0, lambda name, t: ctx.prior(ctx.xs_q if name == "q" else ctx.xs_on, d, t))
    if GLOBAL:
        assert not PRIORS, "T2CNN_GLOBAL and T2CNN_PRIORS both use the extra channels"
        extra = global_extra(ctx, v, qf, ok)
    if FUTURE:  # idea 4, teacher side: the horizon FLOW, not the speed the label is cut from
        assert extra is None, "T2CNN_FUTURE uses the extra channels"
        extra = np.nan_to_num(np.where(ctx.el[d[:, None], fi],
                                      z["true_flow"][d[:, None], fi] / ctx.cap, 0.0), nan=0.0).astype(np.float32)
    x = make_x(ctx, stat, v, qf, ok, t0, z["weekday"][d] >= 5, extra)
    y = ctx.q[d[:, None], fi]                                            # (N, F, S)
    m = ctx.el[d[:, None], fi]
    es = z["dates"][d] >= FIT_END
    out = dict(x=x, y=y.astype(np.float16), m=m, es=es, lid=np.arange(ctx.S) + ctx.loff, is_on=is_on)
    if PL:  # idea 10: the demand the path set routes through each link, projected to T+30
        out["pl"] = path_load(ctx, qf, ok, t0, z["weekday"][d]).astype(np.float16)
    if TEACHER:  # the same horizon flow the teacher was trained with, kept small until the teacher is run
        out["fut"] = np.nan_to_num(np.where(ctx.el[d[:, None], fi],
                                           z["true_flow"][d[:, None], fi] / ctx.cap, 0.0), nan=0.0).astype(np.float16)
    if CONTRAST:  # idea 5: the window seen from its own future, as the positive pair. The collapse layer needs H
        # slots, so each of the F horizon slots is repeated H // F times.
        rep = H // F
        fv = np.repeat(z["true_speed"][d[:, None], fi], rep, axis=1)
        fq = np.repeat(z["true_flow"][d[:, None], fi], rep, axis=1)
        fo = np.repeat(ctx.el[d[:, None], fi], rep, axis=1)
        out["xf"] = make_x(ctx, stat, fv, fq, fo, t0, z["weekday"][d] >= 5)
    if AUX > 0:  # future v/vf targets for the multi-task head
        va = z["true_speed"][d[:, None], fi] / ctx.vf
        out["ma"] = m & np.isfinite(va)
        out["ya"] = np.nan_to_num(va, nan=0.0).astype(np.float16)
    return out


def batch(parts, idx_pool, bs, pad_id=0):
    """Random 64-link crops. Returns x, y, m and a dict of extras (lid for the embedding; ya, ma for the aux head)."""
    if ARCH == "tfm":  # whole panels: every window of a batch comes from one panel, all its links, no crop
        p = parts[rng.integers(len(parts))]
        pool = idx_pool[id(p)]
        j = pool[rng.integers(len(pool), size=bs)]
        if ON_SHARE > 0:  # oversample the onset windows of this panel
            pool_on = pool[p["is_on"][pool]]
            if len(pool_on):
                use = rng.random(bs) < ON_SHARE
                j[use] = pool_on[rng.integers(len(pool_on), size=int(use.sum()))]
        j = np.sort(j)
        tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(DEV, dt)  # noqa: E731
        x = tt(p["x"][j], torch.float32)
        if PRIORS:  # prior dropout per window, as in the crop path (note 20)
            x[torch.from_numpy(rng.random(bs) < 0.5).to(DEV), -N_PRI:] = 0
        exb = {"lid": None, "is_on": torch.from_numpy(p["is_on"][j]).to(DEV)}
        if "yt" in p:
            exb["yt"] = tt(p["yt"][j], torch.float32)
        if "pl" in p:
            exb["pl"] = tt(p["pl"][j], torch.float32)
        return x, tt(p["y"][j], torch.float32), tt(p["m"][j], torch.float32), exb
    xs, ys, ms, ls, yas, mas = [], [], [], [], [], []
    yts, pls = [], []
    for _ in range(bs):
        p = parts[rng.integers(len(parts))]
        pool = idx_pool[id(p)]
        j = pool[rng.integers(len(pool))]
        S = p["x"].shape[-1]
        s0 = rng.integers(0, S - CROP + 1) if S > CROP else 0
        sl = slice(s0, s0 + CROP)
        x, y, m, lid = p["x"][j, :, :, sl], p["y"][j, :, sl], p["m"][j, :, sl], p["lid"][sl]
        ya, ma = (p["ya"][j, :, sl], p["ma"][j, :, sl]) if AUX > 0 else (None, None)
        yt_ = p["yt"][j, :, sl] if "yt" in p else None
        pl_ = p["pl"][j, sl] if "pl" in p else None
        pad = CROP - x.shape[-1]
        if pad > 0:
            x, y, m = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, ((0, 0), (0, pad))), np.pad(m, ((0, 0), (0, pad)))
            lid = np.pad(lid, (0, pad), constant_values=pad_id)
            if AUX > 0:
                ya, ma = np.pad(ya, ((0, 0), (0, pad))), np.pad(ma, ((0, 0), (0, pad)))
            if yt_ is not None:
                yt_ = np.pad(yt_, ((0, 0), (0, pad)))
        if PRIORS and rng.random() < 0.5:  # prior dropout: this window sees no priors
            x = x.copy()
            x[-N_PRI:] = 0
        if EMBED and rng.random() < EMB_DROP:  # embedding dropout: this window sees no link identity
            lid = np.full_like(lid, pad_id)
        xs.append(x); ys.append(y); ms.append(m); ls.append(lid)
        yts.append(yt_)
        if pl_ is not None:
            pls.append(np.pad(pl_, (0, pad)) if pad > 0 else pl_)
        if AUX > 0:
            yas.append(ya); mas.append(ma)
    t = lambda a, dt: torch.from_numpy(np.stack(a)).to(DEV, dt)  # noqa: E731
    ex = {"lid": t(ls, torch.long)}
    if AUX > 0:
        ex["ya"], ex["ma"] = t(yas, torch.float32), t(mas, torch.float32)
    if yts and yts[0] is not None:
        ex["yt"] = t(yts, torch.float32)
    if pls:
        ex["pl"] = t(pls, torch.float32)
    return t(xs, torch.float32), t(ys, torch.float32), t(ms, torch.float32), ex


def loss_fn(logit, y, m):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)


def lovasz_hinge(logit, y, m):
    """Mean over windows of the Lovasz hinge on each window's labelled cells (the IoU is scored per window)."""
    losses = []
    for lg, lb, mk in zip(logit.reshape(len(logit), -1), y.reshape(len(y), -1), m.reshape(len(m), -1)):
        keep = mk > 0.5
        lg, lb = lg[keep], lb[keep]
        if lb.numel() == 0 or lb.sum() == 0:
            continue
        errors = 1.0 - lg * (2.0 * lb - 1.0)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt = lb[perm]
        inter = gt.sum() - gt.cumsum(0)
        union = gt.sum() + (1.0 - gt).cumsum(0)
        jac = 1.0 - inter / union
        grad = torch.cat([jac[:1], jac[1:] - jac[:-1]])
        losses.append(torch.dot(nn.functional.relu(errors_sorted), grad))
    return torch.stack(losses).mean() if losses else logit.sum() * 0.0


@torch.no_grad()
def predict(net, x, lid=None):
    """Queue probabilities (N, F, S). lid (S,): the panel's global link ids, needed only by an embedding net."""
    net.eval()
    out = []
    for j in range(0, len(x), 64):
        xb = torch.from_numpy(x[j:j + 64].astype(np.float32)).to(DEV)
        lb = None if lid is None else torch.from_numpy(np.broadcast_to(lid, (len(xb), len(lid))).copy()).to(DEV, torch.long)
        out.append(torch.sigmoid(net(xb, lb)[:, :F]).float().cpu().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0, F, x.shape[-1]), np.float32)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    parts, ctxs, stats = [], [], {}
    n_links = 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = n_links  # global link ids for the identity embedding
        n_links += ctx.S
        ramps = load_ramps(panel, "train", ctx.z)
        stats[panel] = static_channels(ctx, ramps)
        parts.append(train_windows(ctx, stats[panel]))
        if SSL:  # whole fit days for the reconstruction stage
            fd = np.flatnonzero(ctx.z["dates"] < FIT_END)
            parts[-1]["ssl"] = dict(v=ctx.z["true_speed"][fd].astype(np.float32), q=ctx.z["true_flow"][fd].astype(np.float32),
                                    ok=ctx.el[fd].copy(), wk=ctx.z["weekday"][fd] >= 5, ctx=ctx, stat=stats[panel])
        ctx.z = {k: ctx.z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] windows {len(parts[-1]['x']):,} ({time.time() - t_start:.0f}s)", flush=True)
    tr_pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    mk = lambda: (T2Tfm if ARCH == "tfm" else T2Net)(parts[0]["x"].shape[1], WIDTH, n_links, EMBED, AUX > 0).to(DEV)  # noqa: E731
    net = mk()
    ema = None
    if EMA > 0:
        ema = mk()
        ema.load_state_dict(net.state_dict())
    ev = ema if ema is not None else net  # the weights that are early-stopped, saved and scored
    assert ARCH == "tfm" or not (ON_SHARE or LISTWISE), "T2CNN_ON_SHARE / T2CNN_LISTWISE need T2CNN_ARCH=tfm"
    if INIT:  # fine-tuning from saved weights
        net.load_state_dict(torch.load(INIT, map_location=DEV))
    if T1INIT:  # transfer the Task 1 trunk: identical Block shapes, only the dilation values differ
        t1 = torch.load(T1INIT, map_location=DEV)
        own, n = net.state_dict(), 0
        for k, v in t1.items():
            if k.startswith("blocks.") and k in own and own[k].shape == v.shape:
                own[k] = v
                n += 1
        net.load_state_dict(own)
        print(f"T1 transfer from {T1INIT}: {n} tensors copied into the trunk", flush=True)
        if ema is not None:
            ema.load_state_dict(net.state_dict())
    if TEACHER:  # idea 4: run the privileged teacher once over every training window and keep its probabilities
        tnet = (T2Tfm if ARCH == "tfm" else T2Net)(parts[0]["x"].shape[1] + F, WIDTH, n_links,
                                                   EMBED, AUX > 0).to(DEV)
        tnet.load_state_dict(torch.load(TEACHER, map_location=DEV))
        tnet.eval()
        for p_ in parts:
            n_w = len(p_["x"])
            yt = np.empty(p_["y"].shape, np.float16)
            for a in range(0, n_w, 128):
                b_ = slice(a, min(a + 128, n_w))
                xb = torch.from_numpy(np.ascontiguousarray(p_["x"][b_])).to(DEV, torch.float32)
                fb = torch.from_numpy(np.ascontiguousarray(p_["fut"][b_])).to(DEV, torch.float32)
                fb = fb[:, :, None, :].expand(-1, -1, xb.shape[2], -1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = tnet(torch.cat([xb, fb], 1))[:, :F].float()
                yt[b_] = torch.sigmoid(o).cpu().numpy().astype(np.float16)
            p_["yt"] = yt
            del p_["fut"]
        del tnet
        print(f"teacher {TEACHER}: soft targets computed for every training window", flush=True)
    if CONTRAST:  # idea 5: pretrain the trunk against its own future, then throw the future encoder away
        fut_net = mk()
        copt = torch.optim.AdamW(list(net.parameters()) + list(fut_net.parameters()), lr=LR, weight_decay=1e-4)
        for cstep in range(1, CONTRAST + 1):
            p_ = parts[rng.integers(len(parts))]
            pool = tr_pool[id(p_)]
            j = np.sort(pool[rng.integers(len(pool), size=min(BATCH, len(pool)))])
            S_ = p_["x"].shape[-1]
            s0 = rng.integers(0, max(S_ - CROP, 0) + 1) if S_ > CROP else 0
            sl = slice(s0, s0 + CROP)
            xb = torch.from_numpy(np.ascontiguousarray(p_["x"][j][:, :, :, sl])).to(DEV, torch.float32)
            xf = torch.from_numpy(np.ascontiguousarray(p_["xf"][j][:, :, :, sl])).to(DEV, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zh = net.feats(xb).float().mean(-1)
                zf = fut_net.feats(xf).float().mean(-1)
            zh = torch.nn.functional.normalize(zh, dim=1)
            zf = torch.nn.functional.normalize(zf, dim=1)
            lg = zh @ zf.t() / 0.07
            lab = torch.arange(len(zh), device=DEV)
            closs = 0.5 * (torch.nn.functional.cross_entropy(lg, lab) + torch.nn.functional.cross_entropy(lg.t(), lab))
            copt.zero_grad(set_to_none=True)
            closs.backward()
            copt.step()
            if cstep % 500 == 0:
                print(f"contrast {cstep}/{CONTRAST}: loss {closs.item():.4f}", flush=True)
        del fut_net, copt
        for p_ in parts:
            p_.pop("xf", None)
        if ema is not None:
            ema.load_state_dict(net.state_dict())
        print("contrastive pretraining done; the history trunk is kept", flush=True)
    if SSL:
        assert REC > 0 and net.rec is not None and ARCH == "cnn", "T2CNN_SSL needs T2CNN_REC > 0 and the CNN"
        sopt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
        ssched = torch.optim.lr_scheduler.OneCycleLR(sopt, max_lr=LR, total_steps=SSL, pct_start=0.05)
        run = 0.0
        for sstep in range(1, SSL + 1):
            xs = []
            for _ in range(BATCH):
                sp = parts[rng.integers(len(parts))]["ssl"]
                nd, nt, S_ = sp["v"].shape
                d_ = int(rng.integers(nd)); t_ = int(rng.integers(H, nt + 1))
                s0 = int(rng.integers(0, S_ - CROP + 1)) if S_ > CROP else 0
                sl = slice(s0, s0 + CROP)
                c_ = sp["ctx"]
                v_, q_, o_ = sp["v"][d_, t_ - H:t_], sp["q"][d_, t_ - H:t_], sp["ok"][d_, t_ - H:t_]
                o_ = o_ & np.isfinite(v_) & np.isfinite(q_)
                x_ = make_x(c_, sp["stat"], v_[None], q_[None], o_[None], np.array([t_ % T]), np.array([sp["wk"][d_]]))[0]
                x_ = x_[:, :, sl]
                if x_.shape[-1] < CROP:
                    x_ = np.pad(x_, ((0, 0), (0, 0), (0, CROP - x_.shape[-1])))
                xs.append(x_)
            xb = torch.from_numpy(np.stack(xs)).to(DEV, torch.float32)
            rate = torch.empty(len(xb), 1, 1, device=DEV).uniform_(SSL_LO, SSL_HI)
            hid = torch.rand_like(xb[:, 2]) < rate
            xin = xb.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = net.rec(net.blocks(net.inp(xin))).float()
            wgt = (xb[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
            sloss = ((rec - xb[:, 0:2]) ** 2 * wgt).sum() / wgt.sum().clamp(min=1)
            sopt.zero_grad(set_to_none=True)
            sloss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            sopt.step(); ssched.step()
            run += sloss.item()
            if sstep % 1000 == 0:
                print(f"ssl {sstep}/{SSL}: rec loss {run / 1000:.5f} ({time.time() - t_start:.0f}s)", flush=True)
                run = 0.0
        del sopt, ssched
        for p_ in parts:
            p_.pop("ssl", None)
        if ema is not None:
            ema.load_state_dict(net.state_dict())
        print("ssl pretraining done", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    best, best_state = 1e9, None
    if EVAL_ONLY:
        ev.load_state_dict(torch.load(EVAL_ONLY, map_location=DEV))
    for step in range(1, (0 if EVAL_ONLY else STEPS) + 1):
        x, y, m, ex = batch(parts, tr_pool, BATCH, n_links)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, ex["lid"] if EMBED else None)
        logit = out[:, :F].float()
        loss = loss_fn(logit, y, m)
        if LOVASZ > 0:
            loss = loss + LOVASZ * lovasz_hinge(logit, y, m)
        if LISTWISE > 0 and "is_on" in ex and bool(ex["is_on"].any()):
            # onset windows: which eligible link breaks down first -> softmax over links at T+30 vs the true set
            on = ex["is_on"]
            lo, yo, mo = logit[on, F - 1], y[on, F - 1], m[on, F - 1]
            tgt = yo * mo
            has = tgt.sum(1) > 0
            if bool(has.any()):
                lsm = torch.log_softmax(lo[has].masked_fill(mo[has] < 0.5, -1e4), 1)
                loss = loss + LISTWISE * (-(tgt[has] / tgt[has].sum(1, keepdim=True)) * lsm).sum(1).mean()
        if DISTILL > 0 and "yt" in ex:  # idea 4: the privileged teacher as a soft target
            loss = loss + DISTILL * (nn.functional.binary_cross_entropy_with_logits(
                logit, ex["yt"], reduction="none") * m).sum() / m.sum().clamp(min=1)
        if REC > 0 and net.rec is not None:  # idea 9/10: reconstruct v/vf and q/cap of the visible history cells
            hid = torch.rand_like(x[:, 2]) < 0.2
            xin = x.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hh = net.blocks(net.inp(xin))
                rec = net.rec(hh).float()
            tgt = x[:, 0:2].float()
            wgt = (x[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
            loss = loss + REC * ((rec - tgt) ** 2 * wgt).sum() / wgt.sum().clamp(min=1)
        if PL > 0 and net.pl is not None and "pl" in ex:  # idea 10: the Task 4 auxiliary
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hp = net.pl(net.collapse(net.blocks(net.inp(x)))).float()[:, 0, 0]
            loss = loss + PL * nn.functional.huber_loss(hp, ex["pl"], delta=0.5)
        if AUX > 0:  # multi-task: future v/vf regression
            hub = nn.functional.huber_loss(out[:, F:].float(), ex["ya"], reduction="none", delta=0.2)
            loss = loss + AUX * (hub * ex["ma"]).sum() / ex["ma"].sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(EMA).add_(pn.detach(), alpha=1 - EMA)
        if step % 500 == 0:
            with torch.no_grad():
                tot, n = 0.0, 0
                for p in parts:
                    es = np.flatnonzero(p["es"])[:400]
                    if len(es):
                        pr = predict(ev, p["x"][es], p["lid"] if EMBED else None)
                        yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                        eps = 1e-6
                        bce = -(yv * np.log(pr + eps) + (1 - yv) * np.log(1 - pr + eps))
                        tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in ev.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  ES bce {es_loss:.4f}{flag} ({time.time() - t_start:.0f}s)", flush=True)
    net = ev
    (ROOT / "work" / "models").mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        net.load_state_dict(best_state)
        torch.save(best_state, ROOT / "work" / "models" / f"{TAG}.pt")
    if FUTURE:  # the teacher sees the horizon, so the official-window scoring path does not apply to it
        print(f"FUTURE run: teacher checkpoint saved as {TAG}.pt; skipping the scoring pass", flush=True)
        return
    del parts
    res: dict = {}
    for ctx in ctxs:
        lid_of = (np.arange(ctx.S) + ctx.loff) if EMBED else None
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            z = load(ctx.panel, split)
            blank = official_blank(ctx.panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            og, _ = ongoing_official(ctx, z, blank)
            assert len(on["d"]) == len(c["on_d"]) and len(og["d"]) == len(c["og_d"]), "window sets differ from the cache"
            outs, pc_on, pc_og = {}, np.zeros((0, ctx.S), np.float32), np.zeros((0, F, ctx.S), np.float32)
            if len(on["d"]):
                xo = make_x(ctx, stats[ctx.panel], on["v"], on["qf"], on["ok"], on["t0"], z["weekday"][on["d"]] >= 5,
                            global_extra(ctx, on["v"], on["qf"], on["ok"]) if GLOBAL else score_extra(ctx, z, on["d"], on["t0"]))
                pc = pc_on = predict(net, xo, lid_of)[:, F - 1]
                for name, p in (("lgbm", c["on_p"]), ("cnn", pc), ("mean", (c["on_p"] + pc) / 2)):
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    outs[("onset", name)] = (iou(pr, on["fut"], on["efut"]), on["d"])
            if len(og["d"]):
                xg = make_x(ctx, stats[ctx.panel], og["v"], og["qf"], og["ok"], og["t0"], z["weekday"][og["d"]] >= 5,
                            global_extra(ctx, og["v"], og["qf"], og["ok"]) if GLOBAL else score_extra(ctx, z, og["d"], og["t0"]))
                pc = pc_og = predict(net, xg, lid_of)
                for name, p in (("lgbm", c["og_p"]), ("cnn", pc), ("mean", (c["og_p"] + pc) / 2)):
                    outs[("ongoing", name)] = (iou(p >= 0.65, og["fut"], og["efut"]), og["d"])
            np.savez_compressed(CACHE / f"t2cnn_{TAG}_{split}_{ctx.panel}.npz", on_p=pc_on, og_p=pc_og,
                                on_d=on["d"], og_d=og["d"])  # CNN probs on the same windows as t2probs_*
            for (kind, name), (s, dd) in outs.items():
                for ds, dmax in DAY_SETS.items():
                    msk = dd < dmax
                    res.setdefault((kind, split, ds), {}).setdefault(name, {})[ctx.panel] = s[msk].mean() if msk.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (kind, split, ds), r in res.items():
        base = r["lgbm"]
        line = []
        for name, d in r.items():
            diffs = [d[p] - base[p] for p in d if np.isfinite(d[p]) and np.isfinite(base[p])]
            line.append(f"{name} {fmean(d):.3f} ({fmean(d) - fmean(base):+.3f}, {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        print(f"== {kind} {split} {ds}: " + "  ".join(line))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
