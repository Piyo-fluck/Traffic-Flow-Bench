"""D2STGNN (Shao et al., VLDB 2022) for Task 2, re-implemented from the paper.

Why this family: in two independent benchmarks (LibCity 2023; BasicTS+, TKDE 2024) it is the strongest traffic
forecaster on speed and flow, and its inductive bias is new to our stack - directional diffusion over the corridor
graph, a decoupling of the propagated (diffusion) and local (inherent) signals, and a dynamic adjacency recomputed
from the current history (spatial relations that change with the state, as they change between runs).

    layer l      Lambda = sigmoid(FC([t-emb, w-emb, E^u, E^d]))                    estimation gate (eq. 3)
                 X_dif = Lambda * X^l
                 diffusion  H_t = sum_k sum_{g in f,b,apt} M_g^k (sum_j relu(X_dif[t-j] W_j)) W_{k,g}   (eqs. 4-8)
                            M_g^k = P_g^k with the diagonal removed; P_f, P_b from the corridor adjacency,
                            modulated by the dynamic attention (eqs. 13-14); P_apt = softmax(relu(E^d E^u^T))
                 X_inh = X^l - relu(H_dif W_b)                                      backcast residual (eq. 1)
                 inherent   GRU over time + positional encoding + multi-head self-attention (eqs. 10-12)
                 X^{l+1} = X_inh - relu(H_inh W_b')                                 (eq. 2)
                 forecast   last hidden state of each block -> F hidden states (linear; the paper is
                            autoregressive)
    output       sum over layers and blocks -> 2-layer FC -> F queue logits per link (eq. 15)

Deviations from the paper, all forced by Task 2: queue logits with BCE + 0.5 Lovasz instead of MAE regression;
the time embeddings are a projection of the input's time channels (sin, cos, weekend); node embeddings index the
global link id (STID's numbering); the adjacency is a Gaussian kernel of milepost distance (sigma 3 km, cut 0.1).

    CUDA_VISIBLE_DEVICES=0 python analysis/t2_d2stgnn.py --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "analysis", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

DEV = "cuda"
SIGMA_KM, CUT = 3.0, 0.1


def corridor_adjacency(milepost) -> np.ndarray:
    """(S, S) thresholded Gaussian kernel of the milepost distance (DCRNN's construction), no self loops."""
    m = np.asarray(milepost, np.float64)
    m = np.where(np.isfinite(m), m, np.nanmedian(m))
    dist = np.abs(m[:, None] - m[None, :])
    # mileposts are in miles in the release; convert to km for the kernel width
    a = np.exp(-(dist * 1.609344 / SIGMA_KM) ** 2)
    a[a < CUT] = 0.0
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float32)


class DiffusionBlock(nn.Module):
    def __init__(self, d, kt=3, ks=2):
        super().__init__()
        self.kt, self.ks = kt, ks
        self.wt = nn.ModuleList([nn.Linear(d, d) for _ in range(kt)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3 * ks)])
        self.fore = nn.Linear(d, F * d)
        self.back = nn.Linear(d, d)

    def forward(self, x, mats):                                  # x (B, T, N, d); mats: list of 3 (B, N, N)
        B, T, N, d = x.shape
        z = 0
        for j in range(self.kt):                                 # sum_j relu(X[t-j] W_j), aligned at t
            s = torch.relu(self.wt[j](x))
            z = z + torch.nn.functional.pad(s, (0, 0, 0, 0, j, 0))[:, :T]
        h = 0
        for g, P in enumerate(mats):
            Pk = P
            for k in range(self.ks):
                eye = torch.eye(N, device=x.device, dtype=P.dtype)
                M = Pk * (1 - eye)
                h = h + self.wk[g * self.ks + k](torch.einsum("bnm,btmd->btnd", M, z))
                Pk = torch.bmm(Pk, P)
        back = torch.relu(self.back(h))
        fore = self.fore(h[:, -1]).view(B, N, F, d).permute(0, 2, 1, 3)        # (B, F, N, d)
        return fore, back


class InherentBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.fore = nn.Linear(d, F * d)
        self.back = nn.Linear(d, d)
        pe = torch.zeros(H, d)
        pos = torch.arange(H).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                                         # (B, T, N, d)
        B, T, N, d = x.shape
        s = x.permute(0, 2, 1, 3).reshape(B * N, T, d)
        with torch.autocast("cuda", enabled=False):
            g, _ = self.gru(s.float())
        g = g.to(s.dtype) + self.pe[:T].to(s.dtype)
        a, _ = self.att(g, g, g, need_weights=False)
        h = a.view(B, N, T, d).permute(0, 2, 1, 3)
        back = torch.relu(self.back(h))
        fore = self.fore(h[:, -1]).view(B, N, F, d).permute(0, 2, 1, 3)
        return fore, back


class D2STGNN(nn.Module):
    def __init__(self, cin, n_links, d=32, layers=4, kt=3, ks=2):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(cin, d)
        self.eu = nn.Embedding(n_links + 1, d, padding_idx=n_links)
        self.ed = nn.Embedding(n_links + 1, d, padding_idx=n_links)
        self.temb = nn.Linear(3, 2 * d)                          # (sin, cos, weekend) -> day and week embeddings
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, 1))
                                   for _ in range(layers)])
        self.dif = nn.ModuleList([DiffusionBlock(d, kt, ks) for _ in range(layers)])
        self.inh = nn.ModuleList([InherentBlock(d) for _ in range(layers)])
        self.dyn_fc = nn.Sequential(nn.Linear(H * d, d), nn.ReLU(), nn.Linear(d, d))
        self.wq, self.wk = nn.Linear(4 * d, d, bias=False), nn.Linear(4 * d, d, bias=False)
        self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, lid, A):
        """x (B, C, H, S) - the CNN input; lid (S,) global link ids; A (S, S) corridor adjacency -> (B, F, S)."""
        B, C, T, N = x.shape
        xt = x.permute(0, 2, 3, 1)                                # (B, T, N, C)
        h = self.inp(xt)
        tt = self.temb(xt[:, -1, 0, 9:12])                        # time channels of the last slot
        tD, tW = tt[:, :self.d], tt[:, self.d:]
        eu, ed = self.eu(lid)[None].expand(B, -1, -1), self.ed(lid)[None].expand(B, -1, -1)
        ctx = torch.cat([tD[:, None].expand(-1, N, -1), tW[:, None].expand(-1, N, -1), eu, ed], -1)   # (B, N, 4d)
        # static transitions and the self-adaptive one (eq. 7)
        Pf = A / A.sum(1, keepdim=True).clamp(min=1e-6)
        Pb = A.t() / A.t().sum(1, keepdim=True).clamp(min=1e-6)
        Papt = torch.softmax(torch.relu(self.ed(lid) @ self.eu(lid).t()), -1)
        # dynamic transitions (eqs. 13-14): the history as a feature, then attention masked by the static support
        dyn = self.dyn_fc(h.permute(0, 2, 1, 3).reshape(B, N, T * self.d))
        dfu = torch.cat([dyn, tD[:, None].expand(-1, N, -1), tW[:, None].expand(-1, N, -1), eu], -1)
        dfd = torch.cat([dyn, tD[:, None].expand(-1, N, -1), tW[:, None].expand(-1, N, -1), ed], -1)
        att_u = torch.softmax(self.wq(dfu) @ self.wk(dfu).transpose(1, 2) / math.sqrt(self.d), -1)
        att_d = torch.softmax(self.wq(dfd) @ self.wk(dfd).transpose(1, 2) / math.sqrt(self.d), -1)
        mats = [Pf[None] * att_u, Pb[None] * att_d, Papt[None].expand(B, -1, -1)]
        out = 0
        for l in range(len(self.dif)):
            lam = torch.sigmoid(self.gate[l](ctx))[:, None]      # (B, 1, N, 1)
            f1, b1 = self.dif[l](lam * h, mats)
            x_inh = h - b1
            f2, b2 = self.inh[l](x_inh)
            h = x_inh - b2
            out = out + f1 + f2
        return self.out(out)[..., 0]                              # (B, F, N)


# ------------------------------------------------------------------------------------------ training
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    parts, n_links = [], 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = n_links
        n_links += ctx.S
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        p = cnn.train_windows(ctx, stat)
        p["A"] = torch.from_numpy(corridor_adjacency(ctx.z["milepost"])).to(DEV)
        p["lid_t"] = torch.from_numpy(p["lid"]).to(DEV, torch.long)
        parts.append(p)
        print(f"[{panel}] windows {len(p['x']):,}, adjacency degree {float((p['A'] > 0).sum(1).float().mean()):.1f}",
              flush=True)
    net = D2STGNN(parts[0]["x"].shape[1], n_links, args.d, args.layers).to(DEV)
    print(f"params {sum(q.numel() for q in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    tr = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    best, best_state, run = 1e9, None, 0.0
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(DEV, dt)  # noqa: E731
    for step in range(1, args.steps + 1):
        p = parts[rng.integers(len(parts))]
        j = np.sort(tr[id(p)][rng.integers(len(tr[id(p)]), size=args.bs)])
        x, y, m = tt(p["x"][j], torch.float32), tt(p["y"][j], torch.float32), tt(p["m"][j], torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x, p["lid_t"], p["A"]).float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
        if step % 500 == 0:
            net.eval()
            tot, n = 0.0, 0.0
            with torch.no_grad():
                for q in parts:
                    es = np.flatnonzero(q["es"])[:200]
                    for a in range(0, len(es), 32):
                        k = es[a:a + 32]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pr = torch.sigmoid(net(tt(q["x"][k], torch.float32), q["lid_t"], q["A"]).float())
                        yy, mm = tt(q["y"][k], torch.float32), tt(q["m"][k], torch.float32)
                        b = -(yy * torch.log(pr + 1e-6) + (1 - yy) * torch.log(1 - pr + 1e-6))
                        tot += float((b * mm).sum()); n += float(mm.sum())
            net.train()
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, flag = es_loss, " *"
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"step {step}: train loss {run / 500:.4f}  ES bce {es_loss:.4f}{flag} ({time.time() - t_start:.0f}s)",
                  flush=True)
            run = 0.0
    out = ROOT / "work" / "models" / f"t2_d2stgnn_s{args.seed}.pt"
    torch.save({"state": best_state, "cin": parts[0]["x"].shape[1], "n_links": n_links, "d": args.d,
                "layers": args.layers}, out)
    print(f"saved {out} (best ES bce {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
