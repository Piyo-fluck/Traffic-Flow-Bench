"""Diagnostic / model 4: K onset hypotheses that are NOT averaged (multiple-choice learning, relaxed WTA).

Model   the rec_s20 CNN trunk (inp -> blocks -> collapse) + K onset heads (1x1 over links, T+30 logit each)
        + a selector (K logits from link-pooled trunk features). Heads start from the trained T+30 head (+ noise).
        The rec head is kept and still trained (--rec-w), so idea 9b adaptation stays meaningful at evaluation.
Loss    per window and head: L_k = masked BCE + lov * Lovasz hinge at T+30
        winner-takes-most: (1 - eps) on argmin_k L_k, eps / (K - 1) on the others
        + div * mean pairwise cosine similarity of the K probability maps (anti-collapse)
        + sel * CE(selector, winner)
Data    train-month onset windows (ctx.eo) of the 8 panels, 64-link crops; early stop on the 2031-01/02 windows by
        the ORACLE loss mean_w min_k L_k.
Eval    (--eval) replays 9b day by day on the official-like onset windows (trunk + rec adapt, 30 x AdamW 1e-4) and
        reports, next to candidate T:  best-of-K oracle IoU,  best-of-K top-1 hit of the truth,  per-head IoU,
        selector-chosen head, pi-mixture, and the T-blend with the chosen head (0.7 H + 0.3 head).
        The first thing to read is the oracle on the windows T misses (EXPERIMENTS_MM.md section 4).

    python mm/train_mh.py --name mh5 --K 5            # train (about 10 min)
    python mm/train_mh.py --name mh1 --K 1            # the single-head control, same recipe
    python mm/train_mh.py --name mh5 --K 5 --eval     # evaluate a trained run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from diag_123 import hits, nms_peaks  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402

F, H = C.F, C.H
OUT = LAB / "out"
CROP = 64


class MH(nn.Module):
    def __init__(self, net, K, noise=0.05):
        super().__init__()
        self.net = net
        c = C.cnn.WIDTH
        w = net.head[1].weight.detach()[F - 1, :, :, 0]              # (c, 1)
        b = net.head[1].bias.detach()[F - 1]
        self.heads = nn.Conv1d(c, K, 1)
        with torch.no_grad():
            self.heads.weight.copy_(w[None].repeat(K, 1, 1) + noise * w.std() * torch.randn(K, c, 1, device=w.device))
            self.heads.bias.fill_(float(b))
        self.sel = nn.Linear(2 * c, K)
        self.K = K

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.net.collapse(self.net.blocks(self.net.inp(x)))[:, :, 0]      # (B, c, S)
            g = nn.functional.gelu(h)
            logit = self.heads(g).float()                                        # (B, K, S)
            pooled = torch.cat([h.mean(-1), h.amax(-1)], 1).float()
        return logit, self.sel(pooled)


def lovasz_rows(logit, y, m):
    """(B,) Lovasz hinge per row, 0 where the row has no positive."""
    out = []
    for lg, lb, mk in zip(logit, y, m):
        keep = mk > 0.5
        lg, lb = lg[keep], lb[keep]
        if lb.numel() == 0 or lb.sum() == 0:
            out.append(lg.sum() * 0)
            continue
        err = 1.0 - lg * (2.0 * lb - 1.0)
        es, perm = torch.sort(err, descending=True)
        gt = lb[perm]
        inter = gt.sum() - gt.cumsum(0)
        union = gt.sum() + (1.0 - gt).cumsum(0)
        jac = 1.0 - inter / union
        grad = torch.cat([jac[:1], jac[1:] - jac[:-1]])
        out.append(torch.dot(nn.functional.relu(es), grad))
    return torch.stack(out)


def head_losses(logit, y, m, lov):
    """logit (B,K,S), y/m (B,S) -> (B,K)."""
    B, K, S = logit.shape
    ye, me = y[:, None].expand(B, K, S), m[:, None].expand(B, K, S)
    bce = (nn.functional.binary_cross_entropy_with_logits(logit, ye, reduction="none") * me).sum(-1) \
        / me.sum(-1).clamp(min=1)
    if lov > 0:
        bce = bce + lov * lovasz_rows(logit.reshape(B * K, S), ye.reshape(B * K, S), me.reshape(B * K, S)).view(B, K)
    return bce


def load_parts():
    parts = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, _ = C.windows(i, panel, need_train=True)
        d, t0 = ctx.eo
        hi = t0[:, None] + np.arange(-H, 0)
        z = ctx.z
        x = C.cnn.make_x(ctx, stat, z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi],
                         ctx.el[d[:, None], hi], t0, z["weekday"][d] >= 5)
        parts.append(dict(x=x, y=ctx.q[d, t0 + F].astype(np.float32), m=ctx.el[d, t0 + F].astype(np.float32),
                          es=z["dates"][d] >= FIT_END))
        print(f"[{panel}] {len(d)} onset windows", flush=True)
    return parts


def batch(parts, bs, rng, es=False):
    xs, ys, ms = [], [], []
    for _ in range(bs):
        p = parts[rng.integers(len(parts))]
        pool = np.flatnonzero(p["es"] if es else ~p["es"])
        j = pool[rng.integers(len(pool))]
        S = p["x"].shape[-1]
        s0 = rng.integers(0, S - CROP + 1) if S > CROP else 0
        sl = slice(s0, s0 + CROP)
        x, y, m = p["x"][j, :, :, sl], p["y"][j, sl], p["m"][j, sl]
        pad = CROP - x.shape[-1]
        if pad > 0:
            x, y, m = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, (0, pad)), np.pad(m, (0, pad))
        xs.append(x); ys.append(y); ms.append(m)
    return C.to_dev(np.stack(xs)), C.to_dev(np.stack(ys)), C.to_dev(np.stack(ms))


def train(args, out):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = load_parts()
    net, _ = C.build_net(args.ckpt, "full")
    for p in net.parameters():
        p.requires_grad_(not args.freeze_trunk)
    model = MH(net, args.K).to(C.DEV)
    new = list(model.heads.parameters()) + list(model.sel.parameters())
    groups = [{"params": new, "lr": args.lr_head}]
    if not args.freeze_trunk:
        groups.append({"params": list(net.parameters()), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    best, t0 = 1e9, time.time()
    for step in range(1, args.steps + 1):
        x, y, m = batch(parts, args.bs, rng)
        logit, sl = model(x)
        L = head_losses(logit, y, m, args.lov)                              # (B, K)
        win = L.detach().argmin(1)
        K = args.K
        wts = torch.full_like(L, args.eps / max(K - 1, 1)) if K > 1 else torch.zeros_like(L)
        wts.scatter_(1, win[:, None], 1.0 - (args.eps if K > 1 else 0.0))
        loss = (wts * L).sum(1).mean()
        if K > 1 and args.div > 0:
            pm = torch.sigmoid(logit) * m[:, None]
            pn = nn.functional.normalize(pm, dim=-1)
            cos = pn @ pn.transpose(1, 2)                                   # (B, K, K)
            loss = loss + args.div * (cos.sum((1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(1)).mean() / (K * (K - 1))
        if K > 1 and args.sel > 0:
            loss = loss + args.sel * nn.functional.cross_entropy(sl, win)
        if args.rec_w > 0 and not args.freeze_trunk:
            loss = loss + args.rec_w * C.ssl_loss(net, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                er = np.random.default_rng(0)
                vals, usage = [], np.zeros(K)
                for _ in range(20):
                    xe, ye, me = batch(parts, 64, er, es=True)
                    le = head_losses(model(xe)[0], ye, me, args.lov)
                    vals.append(le.min(1).values.mean().item())
                    usage += np.bincount(le.argmin(1).cpu().numpy(), minlength=K)
            model.train()
            v = float(np.mean(vals))
            flag = ""
            if v < best:
                best, flag = v, " *"
                torch.save(model.state_dict(), out / "mh.pt")
            print(f"step {step}: loss {loss.item():.4f}  ES oracle {v:.4f}{flag}  head usage "
                  f"{np.round(usage / usage.sum(), 2).tolist()} ({time.time() - t0:.0f}s)", flush=True)


def evaluate(args, out):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    rows, fam = [], {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for split in C.SPLITS:
            w, z, ref = W[split]["on"], W[split]["z"], W[split]["ref"]
            if not len(w["d"]):
                continue
            net, _ = C.build_net(args.ckpt, "full")
            model = MH(net, args.K).to(C.DEV)
            model.load_state_dict(torch.load(out / "mh.pt", map_location=C.DEV))
            model.eval()
            params = list(net.inp.parameters()) + list(net.blocks.parameters()) + list(net.rec.parameters())
            N, S = w["cells"].shape
            PK = np.zeros((N, args.K, S), np.float32)
            PI = np.zeros((N, args.K), np.float32)
            for day in sorted(set(w["d"].tolist())):
                m = np.flatnonzero(w["d"] == day)
                x = C.to_dev(C.make_x(ctx, stat, w, m, z))
                with torch.no_grad():
                    lg, sl = model(x)
                PK[m] = torch.sigmoid(lg).cpu().numpy()
                PI[m] = torch.softmax(sl, 1).cpu().numpy()
                C.adapt(net, params, (lambda day=day: C.past_batch(ctx, stat, z, np.arange(day), 16, rng)), 30, 1e-4)
            np.savez_compressed(out / f"probs_{split}_{panel}.npz", pk=PK, pi=PI, on_d=w["d"])
            pT = C.W_ON * ref["on_H"] + (1 - C.W_ON) * ref["on_ad9b"]
            iT = C.onset_iou(pT, w)
            ik = np.stack([C.onset_iou(PK[:, k], w) for k in range(args.K)], 1)
            ksel = PI.argmax(1)
            psel = PK[np.arange(N), ksel]
            pmix = (PI[:, :, None] * PK).sum(1)
            variants = {"T": iT, "oracle best-of-K": ik.max(1), "oracle best-of-K + T": np.maximum(ik.max(1), iT),
                        "selector head": C.onset_iou(psel, w), "pi-mixture": C.onset_iou(pmix, w),
                        "T-blend w/ selector head": C.onset_iou(C.W_ON * ref["on_H"] + (1 - C.W_ON) * psel, w),
                        "T-blend w/ pi-mixture": C.onset_iou(C.W_ON * ref["on_H"] + (1 - C.W_ON) * pmix, w)}
            for k in range(args.K):
                variants[f"head {k}"] = ik[:, k]
            for name, v in variants.items():
                fam.setdefault((split, name), {})[panel] = v
            for j in range(N):
                tr, el = w["cells"][j], w["efut"][j, F - 1]
                tops = [nms_peaks(PK[j, k], el, 1, 3) for k in range(args.K)]
                rows.append(dict(panel=panel, split=split, iouT=iT[j], oracle=ik[j].max(), best_head=int(ik[j].argmax()),
                                 sel_head=int(ksel[j]),
                                 T_top1=int(any(hits(c, tr, 3) for c in nms_peaks(pT[j], el, 1, 3))),
                                 K_top1_any=int(any(len(t) and hits(t[0], tr, 3) for t in tops)),
                                 n_distinct_tops=len({int(t[0]) // 4 for t in tops if len(t)})))
            print(f"[{panel} {split}] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_per_window.csv", index=False)
    lines = []
    for split in C.SPLITS:
        lines.append(f"\n=== {split} | onset | family (delta vs T) ===")
        base = C.family_mean({p: v.mean() for p, v in fam[(split, "T")].items()})
        for (sp, name), d in fam.items():
            if sp != split:
                continue
            f_ = C.family_mean({p: v.mean() for p, v in d.items()})
            lines.append(f"  {name:<28} {f_:.4f} ({f_ - base:+.4f})")
        s = df[df.split == split]
        for sname, q in (("all", s), ("missed (IoU_T = 0)", s[s.iouT == 0]), ("weak (IoU_T < 0.5)", s[s.iouT < 0.5])):
            if len(q):
                lines.append(f"  [{sname}, n={len(q)}] top-1 hit: T {q.T_top1.mean():.2f} | any of K {q.K_top1_any.mean():.2f}"
                             f"  oracle IoU {q.oracle.mean():.3f} vs T {q.iouT.mean():.3f}"
                             f"  distinct top peaks per window {q.n_distinct_tops.mean():.2f}"
                             f"  selector picks the best head {np.mean(q.sel_head == q.best_head):.2f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "eval.txt").write_text(txt + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ckpt", default=str(C.MODELS / "t2_cnn_lov_rec_s20.pt"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="trunk")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--lov", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.05, help="relaxed WTA share of the losing heads")
    ap.add_argument("--div", type=float, default=0.1)
    ap.add_argument("--sel", type=float, default=0.1)
    ap.add_argument("--rec-w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.eval:
        evaluate(args, out)
    else:
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))
        train(args, out)


if __name__ == "__main__":
    main()
