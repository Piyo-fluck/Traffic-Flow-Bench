"""Chronos-2 (Apache 2.0) as a pretrained backbone trained DIRECTLY on the Task 2 queue labels.

Round 1 (note 124) used Chronos-2's own quantile-forecast fine-tuning for 3k steps of 4 series-groups each - about
1% of what the production CNN sees - and read queues off the forecast quantiles. Here the pretrained encoder is
the initialisation of a full-scale supervised Task 2 model:

    input     one window = one group of 2 x L series: every link's v/vf and q/cap (released layer, NaN where
              unseen) from midnight of day d - n_prev (n_prev ~ U{0..max_days}, same split) up to slot T-1 of day d;
              64-link crops in training, whole corridors at scoring
    encoder   Chronos-2 `model.encode` (patch embedding, time attention, group attention across the 2L series)
    head      per link: [output-patch token, REG token] of its v/vf series -> MLP -> 6 queue logits (T+5..T+30)
    loss      BCE + 0.5 Lovasz on eligible horizon cells (the production CNN's loss), labels from train truth
    windows   the CNN's training windows: earliest onsets + up to 3000 ongoing candidates per panel; fit days train,
              days from FIT_END early-stop

Logs every --log-every steps to the log file given by the runner; the best early-stopping state is saved to
work/models/chronos_queue_<tag>.pt, then the official-like windows of both scored months are scored (alone and
in the production blends) and cached to fm_lab/cache/chq_<tag>_<split>_<panel>.npz.

    python fm_lab/chronos_queue.py --tag a --max-days 7 --steps 30000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import common as C  # noqa: E402
from t2_eda02 import candidates  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402

F, H = C.F, C.H
BLOCK = True
DEV = C.DEV
CROP = 64
T_DAY = 288


# ------------------------------------------------------------------ block-diagonal group attention
# Chronos-2's GroupSelfAttention attends over EVERY series of the forward and masks other groups out, so memory is
# quadratic in the series of a forward (2 windows x 128 series already took 40 GiB). Our groups are contiguous and
# of equal size, so attention is computed inside each group instead: the same result, linear in the batch.
_GROUP_N = None          # series per group for the current forward, or None for the library's own path


def enable_block_group_attention():
    from chronos.chronos2 import layers as L, model as M
    from einops import rearrange
    orig_mask = M.Chronos2Encoder._construct_and_invert_group_time_mask
    orig_fwd = L.GroupSelfAttention.forward

    def mask(group_ids, attention_mask, floating_type):
        if _GROUP_N is None:
            return orig_mask(group_ids, attention_mask, floating_type)
        B, T = attention_mask.shape
        g = B // _GROUP_N
        am = attention_mask.view(g, _GROUP_N, T).permute(0, 2, 1).reshape(g * T, 1, 1, _GROUP_N)
        return (1.0 - am.to(floating_type)) * torch.finfo(floating_type).min      # keys of the same group

    def fwd(self, hidden_states, attention_mask, output_attentions=False):
        if _GROUP_N is None:
            return orig_fwd(self, hidden_states, attention_mask, output_attentions)
        B, T, D = hidden_states.shape
        g = B // _GROUP_N
        h = rearrange(hidden_states, "(g n) t d -> (g t) n d", g=g, n=_GROUP_N)
        out = self.self_attention(self.layer_norm(h), mask=attention_mask, output_attentions=output_attentions)
        h = h + self.dropout(out[0])
        h = rearrange(h, "(g t) n d -> (g n) t d", g=g, t=T)
        return L.AttentionOutput(hidden_states=h, attn_weights=out.attn_weights)

    M.Chronos2Encoder._construct_and_invert_group_time_mask = staticmethod(mask)
    L.GroupSelfAttention.forward = fwd


def enable_checkpointing(enc):
    """activation checkpointing on each of the 12 encoder blocks (recomputed in backward): memory per window drops
    from ~20 activations per block to one, which is what lets the batch grow."""
    from torch.utils.checkpoint import checkpoint
    for blk in enc.encoder.block:
        f = blk.forward

        def wrapped(hidden_states, *a, _f=f, **k):
            if torch.is_grad_enabled() and hidden_states.requires_grad:
                return checkpoint(lambda h: _f(h, *a, **k), hidden_states, use_reentrant=False)
            return _f(hidden_states, *a, **k)
        blk.forward = wrapped


class QueueHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, F))

    def forward(self, tok):
        return self.mlp(tok)


class ChronosQueue(nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.enc = pipe.model
        d = self.enc.config.d_model
        self.head = QueueHead(d)

    def forward(self, ctx, gid, n_links):
        """ctx (B_series, L) with the v/vf series of each group first, then its q/cap series; gid (B_series,);
        n_links list of links per group -> logits (sum n_links, F) in group order."""
        global _GROUP_N
        _GROUP_N = 2 * n_links[0] if BLOCK and len(set(n_links)) == 1 else None
        out, *_ = self.enc.encode(context=ctx, group_ids=gid, num_output_patches=1)
        # _GROUP_N stays set: checkpointed blocks are recomputed in backward and must see the same grouping
        h = out[0]                                              # (B_series, n_patches + 1 + 1, d)
        tok = torch.cat([h[:, -1], h[:, -2]], -1)               # output-patch token, REG token
        idx, o = [], 0
        for n in n_links:
            idx.append(torch.arange(o, o + n, device=h.device))  # the v/vf series of this group
            o += 2 * n
        return self.head(tok[torch.cat(idx)].float())


def series(z, vf, cap):
    """(days, T, S) v/vf and q/cap on the released layer, NaN where unseen or ineligible."""
    v, q = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    return (np.where(ok, v / vf, np.nan).astype(np.float32), np.where(ok, q / cap, np.nan).astype(np.float32))


def context(V, Q, d, t0, n_prev, sl=slice(None)):
    """(2L, len) context of day d up to t0-1 plus n_prev earlier days, links sl; v/vf rows first."""
    lo = max(0, d - n_prev)
    v = np.concatenate([V[k, :, sl] for k in range(lo, d)] + [V[d, :t0, sl]], 0).T
    q = np.concatenate([Q[k, :, sl] for k in range(lo, d)] + [Q[d, :t0, sl]], 0).T
    return np.concatenate([v, q], 0)


def pack(ctxs, max_len):
    """left-pad a list of (rows, len) arrays with NaN to one (sum rows, max_len) tensor + group ids."""
    L = min(max(c.shape[1] for c in ctxs), max_len)
    rows, gid = [], []
    for g, c in enumerate(ctxs):
        c = c[:, -L:]
        if c.shape[1] < L:
            c = np.pad(c, ((0, 0), (L - c.shape[1], 0)), constant_values=np.nan)
        rows.append(c)
        gid.append(np.full(len(c), g))
    return (torch.from_numpy(np.concatenate(rows)).to(DEV), torch.from_numpy(np.concatenate(gid)).to(DEV))


def lovasz(logit, y, m):
    return C.cnn.lovasz_hinge(logit, y, m) if C.cnn.LOVASZ > 0 else logit.sum() * 0


def load_panels():
    P = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        z = ctx.z
        vf = np.asarray(ctx.vf, np.float32)
        cap = np.asarray(ctx.cap, np.float32)
        V, Q = series(z, vf, cap)
        d_on, t_on = ctx.eo
        c = candidates(z)
        d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        rng = np.random.default_rng(i)
        if len(d_og) > 3000:
            s = rng.choice(len(d_og), 3000, replace=False)
            d_og, t_og = d_og[s], t_og[s]
        d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
        fit = np.asarray(z["dates"])[d].astype(str) < FIT_END
        P.append(dict(panel=panel, V=V, Q=Q, q=ctx.q, el=ctx.el, d=d, t0=t0, fit=fit, S=ctx.S))
        print(f"[{panel}] windows {len(d)} (fit {fit.sum()})", flush=True)
    return P


def batch(P, rng, bs, max_days, max_len, split="fit"):
    ctxs, ys, ms, nl = [], [], [], []
    for _ in range(bs):
        p = P[rng.integers(len(P))]
        pool = np.flatnonzero(p["fit"] if split == "fit" else ~p["fit"])
        j = pool[rng.integers(len(pool))]
        d, t0 = int(p["d"][j]), int(p["t0"][j])
        s0 = int(rng.integers(0, p["S"] - CROP + 1)) if p["S"] > CROP else 0
        sl = slice(s0, s0 + CROP)
        n_prev = int(rng.integers(0, max_days + 1))
        ctxs.append(context(p["V"], p["Q"], d, t0, n_prev, sl))
        fi = t0 + np.arange(1, F + 1)
        ys.append(p["q"][d, fi][:, sl].T)
        ms.append(p["el"][d, fi][:, sl].T)
        nl.append(ctxs[-1].shape[0] // 2)
    x, g = pack(ctxs, max_len)
    y = torch.from_numpy(np.concatenate(ys).astype(np.float32)).to(DEV)
    m = torch.from_numpy(np.concatenate(ms).astype(np.float32)).to(DEV)
    return x, g, nl, y, m


def window_loss(logit, y, m, nl):
    """(BCE over all cells, Lovasz per window) - cells of one window = its links x F."""
    bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    lov = bce * 0
    if C.cnn.LOVASZ > 0:
        o, lg, yy, mm = 0, [], [], []
        for n in nl:
            lg.append(logit[o:o + n].T); yy.append(y[o:o + n].T); mm.append(m[o:o + n].T)
            o += n
        lov = C.cnn.lovasz_hinge(torch.stack(lg), torch.stack(yy), torch.stack(mm))
    return bce, lov


@torch.no_grad()
def predict_windows(net, V, Q, d, t0, n_prev, max_len, S, bs=1):
    net.eval()
    out = np.zeros((len(d), F, S), np.float32)
    for j in range(0, len(d), bs):
        ctxs = [context(V, Q, int(d[k]), int(t0[k]), n_prev) for k in range(j, min(j + bs, len(d)))]
        x, g = pack(ctxs, max_len)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x, g, [S] * len(ctxs))
        out[j:j + len(ctxs)] = torch.sigmoid(lg.float()).view(len(ctxs), S, F).permute(0, 2, 1).cpu().numpy()
    net.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=2, help="windows per forward (each = 2 x 64 series); group attention "
                    "is quadratic in the series of a forward, so keep it small and accumulate")
    ap.add_argument("--accum", type=int, default=4, help="forwards accumulated per optimiser step")
    ap.add_argument("--max-days", type=int, default=7)
    ap.add_argument("--score-days", type=int, default=None, help="earlier days at scoring (default max-days)")
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--es-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # data-parallel over the GPUs when launched by torchrun: each rank draws its own windows, gradients averaged
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)),         int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    main_rank = rank == 0
    log = print if main_rank else (lambda *a, **k: None)
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + 1000 * rank)
    from chronos import Chronos2Pipeline
    enable_block_group_attention()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=f"cuda:{local}", dtype=torch.float32)
    max_len = int(pipe.model_context_length)
    net = ChronosQueue(pipe).to(DEV)
    enable_checkpointing(net.enc)
    net.train()
    model = net
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(net, device_ids=[local], find_unused_parameters=True)
    log(f"world {world}; encoder {sum(p.numel() for p in net.enc.parameters()) / 1e6:.1f}M params, d_model "
          f"{net.enc.config.d_model}, context cap {max_len}; args {vars(args)}", flush=True)
    P = load_panels()
    log(f"effective windows per step {args.bs * args.accum * world}", flush=True)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": args.lr_enc},
                             {"params": net.head.parameters(), "lr": args.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_enc, args.lr_head], total_steps=args.steps,
                                                pct_start=0.03)
    es_rng = np.random.default_rng(123)
    es_batches = [batch(P, es_rng, args.bs, args.max_days, max_len, "es") for _ in range(80)]
    best, best_state, run, run_l, t_start = 1e9, None, 0.0, 0.0, time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for a in range(args.accum):
            x, g, nl, y, m = batch(P, rng, args.bs, args.max_days, max_len)
            sync = (not ddp) or a == args.accum - 1
            cm = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with cm:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(x, g, nl)
                bce, lov = window_loss(lg.float(), y, m, nl)
                loss = (bce + C.cnn.LOVASZ * lov) / args.accum
                loss.backward()
            run += bce.item() / args.accum
            run_l += lov.item() / args.accum
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0:
            log(f"step {step}: train bce {run / args.log_every:.5f}  lovasz {run_l / args.log_every:.4f} "
                f"({time.time() - t_start:.0f}s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
            run = run_l = 0.0
        if (step % args.es_every == 0 or step == args.steps) and main_rank:
            net.eval()
            tot, n = 0.0, 0.0
            with torch.no_grad():
                for x, g, nl, y, m in es_batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = net(x, g, nl).float()
                    b = nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none")
                    tot += float((b * m).sum()); n += float(m.sum())
            net.train()
            es = tot / max(n, 1)
            flag = ""
            if es < best:
                best, flag = es, " *"
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
                torch.save(best_state, PROJ / "work" / "models" / f"chronos_queue_{args.tag}.pt")
            print(f"== ES step {step}: bce {es:.5f}{flag} ({time.time() - t_start:.0f}s)", flush=True)
        if ddp and step % args.es_every == 0:
            dist.barrier()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if not main_rank:
        return
    net.load_state_dict(best_state)
    torch.save(best_state, PROJ / "work" / "models" / f"chronos_queue_{args.tag}.pt")
    print(f"saved chronos_queue_{args.tag}.pt (best ES bce {best:.5f})", flush=True)

    # ---------------- scoring on the official-like windows of both scored months
    n_prev = args.max_days if args.score_days is None else args.score_days
    (LAB / "cache").mkdir(exist_ok=True)
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx_, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx_.vf, np.float32), np.asarray(ctx_.cap, np.float32)
        for s in C.SPLITS:
            w = W[s]
            on, og, z = w["on"], w["og"], w["z"]
            V, Q = series(z, vf, cap)
            p_on = predict_windows(net, V, Q, on["d"], on["t0"], n_prev, max_len, ctx_.S)[:, F - 1] \
                if len(on["d"]) else np.zeros((0, ctx_.S), np.float32)
            p_og = predict_windows(net, V, Q, og["d"], og["t0"], n_prev, max_len, ctx_.S) \
                if len(og["d"]) else np.zeros((0, F, ctx_.S), np.float32)
            np.savez(LAB / "cache" / f"chq_{args.tag}_{s}_{panel}.npz", on=p_on, og=p_og.astype(np.float16))
            ref = w["ref"]
            old = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
            H_, ad = ref["on_H"], ref["on_ad9b"]
            r = res.setdefault(s, {})
            if len(on["d"]):
                for nm, p in (("onset chronos-queue", p_on), ("onset H", H_), ("onset 0.7H+0.3ad9b", 0.7 * H_ + 0.3 * ad),
                              ("onset 0.6H+0.25ad9b+0.15cq", 0.6 * H_ + 0.25 * ad + 0.15 * p_on)):
                    r.setdefault(nm, {})[panel] = C.onset_iou(p, on).mean()
            if len(og["d"]):
                for nm, p in (("ongoing chronos-queue", p_og), ("ongoing old(12)", old),
                              ("ongoing 0.8old+0.2cq", 0.8 * old + 0.2 * p_og),
                              ("ongoing 0.6old+0.4cq", 0.6 * old + 0.4 * p_og)):
                    r.setdefault(nm, {})[panel] = C.ongoing_iou(p, og, 0.5).mean()
            print(f"  scored {panel} {s} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s} (family mean; ongoing tau 0.50, onset ratio rule) ===")
        for nm, d in res.get(s, {}).items():
            print(f"  {nm:<30} {C.family_mean(d):.4f}")


if __name__ == "__main__":
    main()
