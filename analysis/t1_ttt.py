"""Task 1 test-time training (TTT) on the scored month: self-supervised fine-tuning of the production CNNs.

T1 is offline, so the whole scored month's released masked layer may be used. The CNNs were trained on the train
run, while validation and private come from other simulation runs (note 17). The month-deviation base (note 18)
already gained +0.007 S_state from that shift alone. TTT adapts the network weights themselves:
    self-supervised target: hide a random 10-50% of the month's visible eligible cells and reconstruct them (the
    observation is the label there), starting from each production checkpoint with a small learning rate.
Evaluation is the in-month pseudo-eval of t1_pseudo_phys: 20% of visible cells hidden with the same seed and panel
order, scored by S_state~ and S_LWR~. **Those eval-hidden cells are removed from the TTT input too**, so the
network never sees the values it is scored on.
Compared: ens3 + month base (no TTT) vs the same after TTT.

    py analysis/t1_ttt.py validation
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

HIDE = 0.2
CKPTS = [("work/sub_m3v1/t1_m3_seed0.pt", 64), ("work/sub_m3v1_s1/t1_m3_seed1.pt", 64), ("work/sub_m3w128/t1_m3_seed0.pt", 128)]
# TTT_CKPTS="path:width,path:width" overrides the production trio, e.g. to adapt a single wider checkpoint.
# The architecture also follows M3_WIDTH / M3_RF, so those must match the checkpoint being loaded.
if os.environ.get("TTT_CKPTS"):
    CKPTS = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in os.environ["TTT_CKPTS"].split(",")]
TTT_STEPS = int(os.environ.get("TTT_STEPS", 400))
TTT_LR = float(os.environ.get("TTT_LR", 2e-4))
# Three knobs, all defaulting to the production behaviour (note 102):
#   TTT_RATE=regime  hide the day's OFFICIAL regime rate (0.20 / 0.30 / 0.50, MASK_SPEC) instead of U(0.1, 0.5),
#                    so the self-supervised task tracks the difficulty of the day it adapts on
#   TTT_COSINE=1     decay the learning rate to zero over the run instead of holding it at TTT_LR
#   TTT_SEEDS=k      adapt every checkpoint k times with different hide/batch seeds; they join the average
TTT_RATE = os.environ.get("TTT_RATE", "uniform")
TTT_COSINE = os.environ.get("TTT_COSINE", "0") == "1"
TTT_SEEDS = int(os.environ.get("TTT_SEEDS", 1))
REGIME_RATE = {"R1": 0.20, "R2": 0.30, "R3": 0.50}
# Mask geometry (note 125). The official Task 1 targets come in temporal runs (about 52% single cells, 38% runs of
# 2-3, 9% runs of 4-11, none longer), while both the TTT hide and this evaluation hide are i.i.d. points.
#   TTT_MASK=target   hide the cells of a REAL target mask (a random other day / crop of the split's own template),
#                     so the self-supervised task has the official gap geometry (block masking, Gan et al. 2026)
#   EVAL_MASK=target  evaluate on cells hidden by another day's real target mask instead of 20% i.i.d. points
TTT_MASK = os.environ.get("TTT_MASK", "point")
EVAL_MASK = os.environ.get("EVAL_MASK", "point")
TEMPLATES: list = []          # per panel (D, T, S) official target masks, filled by main()


def make_sched(opt, steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if TTT_COSINE else None


def sample_with_rates(parts, rng):
    """m3.sample_batch, plus the official mask rate of the day each sample came from (None unless TTT_RATE)."""
    if TTT_RATE != "regime":
        b = m3.sample_batch(parts, rng)
        return b[0], b[1], b[2], b[3], None
    state = rng.bit_generator.state
    rates = []
    for _ in range(m3.BATCH):                      # mirror m3.sample_batch's draw order exactly
        p = parts[rng.integers(len(parts))]
        d = p["tr_days"][rng.integers(len(p["tr_days"]))]
        S = p["X"].shape[-1]
        if S > m3.CROP_S:
            rng.integers(0, S - m3.CROP_S + 1)
        rates.append(REGIME_RATE.get(str(p["regime"][d]), 0.3))
    rng.bit_generator.state = state
    b = m3.sample_batch(parts, rng)
    return b[0], b[1], b[2], b[3], np.array(rates, np.float32)


_trng = np.random.default_rng(77)


def template_crops(B, T, W):
    """B crops (T, W) of real target masks drawn from any panel / day / link offset."""
    out = np.zeros((B, T, W), bool)
    for b in range(B):
        tm = TEMPLATES[_trng.integers(len(TEMPLATES))]
        d = _trng.integers(len(tm))
        S = tm.shape[2]
        s0 = _trng.integers(0, S - W + 1) if S > W else 0
        c = tm[d, :, s0:s0 + W]
        out[b, :, :c.shape[1]] = c
    return torch.from_numpy(out).to(m3.DEV)


def ttt_hide(x, e, gen, rates=None):
    """Hide a share of each sample's visible eligible cells; they become the loss cells.

    `rates` (B,) replaces the default U(0.1, 0.5) with the official rate of the day the sample came from.
    """
    B = x.shape[0]
    if TTT_MASK == "target":
        hide = (x[:, 2] > 0.5) & e & template_crops(B, x.shape[2], x.shape[3])
        x = x.clone()
        x[:, 0:4] = x[:, 0:4] * (~hide).unsqueeze(1).float()
        return x, hide
    if rates is not None:
        rate = torch.as_tensor(rates, device=m3.DEV, dtype=torch.float32).view(B, 1, 1)
    else:
        rate = 0.1 + 0.4 * torch.rand(B, 1, 1, device=m3.DEV, generator=gen)
    hide = (x[:, 2] > 0.5) & e & (torch.rand(x[:, 2].shape, device=m3.DEV, generator=gen) < rate)
    x = x.clone()
    x[:, 0:4] = x[:, 0:4] * (~hide).unsqueeze(1).float()
    return x, hide


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    rng = np.random.default_rng(0)  # the same hide masks as t1_pseudo_phys (same seed, same panel order)
    parts, evals = [], []
    t_start = time.time()
    for panel in PANELS:
        with np.load(CACHE / f"{panel}_t1tables.npz") as f:
            tv, tq = f["tv"], f["tq"]
        z = load(panel, split)
        vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        TEMPLATES.append(np.asarray(z["target"], bool))
        if EVAL_MASK == "target":   # another day's official target mask, applied to today's visible cells
            nd = len(z["dates"])
            shift = rng.integers(1, nd, size=nd)
            hide = vis & np.asarray(z["target"], bool)[(np.arange(nd) + shift) % nd]
        else:
            hide = vis & (rng.random(vis.shape) < HIDE)
        zm = dict(z)
        for k in ("obs_speed", "obs_flow", "obs_occ"):
            zm[k] = np.where(hide, np.nan, z[k]).astype(np.float32)
        bv0, bq0 = tv[z["weekday"]], tq[z["weekday"]]
        dv, dq = month_deviation(zm, bv0, bq0, np.ones(len(z["dates"]), bool), 2.0, kappa_q=10.0)
        bv, bq = bv0 + dv, bq0 + dq
        X = m3.make_inputs(zm, bv, bq)
        lanes = z["lanes"]
        vism = vis & ~hide
        Y = np.stack([np.where(vism, (zm["obs_speed"] - bv) / m3.SCALE_V, 0),
                      np.where(vism, (zm["obs_flow"] - bq) / lanes / m3.SCALE_Q, 0)], axis=1).astype(np.float16)
        p = dict(panel=panel, z=zm, X=X, Y=Y, M=np.zeros(vis.shape, bool), E=vism, base_v=bv, base_q=bq,
                 regime=np.asarray(z["regime"], str), tr_days=np.arange(len(z["dates"])))
        parts.append(p)
        evals.append(dict(hide=hide, vis=vis, ov=z["obs_speed"], oq=z["obs_flow"], lanes=lanes,
                          L=fd_params(panel, z["link_ids"])["length"], ok=ramp_ok(panel, split, z)))
        print(f"[{panel}] ready ({time.time() - t_start:.0f}s)", flush=True)

    def score(nets):
        """nets: a list (the same nets for every panel) or {panel: list} (per-panel TTT, note 173)."""
        st, lw = {}, {}
        for p, ev in zip(parts, evals):
            use = nets[p["panel"]] if isinstance(nets, dict) else nets
            preds = [m3.predict_days(net, p, range(len(p["z"]["dates"]))) for net in use]
            pv, pq = np.mean([a[0] for a in preds], 0), np.mean([a[1] for a in preds], 0)
            h = ev["hide"]
            rv = float(np.sqrt(np.mean((pv - ev["ov"])[h] ** 2)))
            rq = float(np.sqrt(np.mean(((pq - ev["oq"]) / ev["lanes"])[h] ** 2)))
            st[p["panel"]] = SPEED_W * max(0, 1 - rv / SPEED_NORM) + FLOW_W * max(0, 1 - rq / FLOW_NORM)
            k_obs = ev["oq"] / np.maximum(ev["ov"], 1.0)
            k_st = np.where(h, pq / np.maximum(pv, 1.0), k_obs)
            vis, ok, L = ev["vis"], ev["ok"], ev["L"]
            trans = vis[:, :-1] & vis[:, 1:] & ok[:, :-1] & (h[:, :-1] | h[:, 1:])
            known = vis[:, :-1] & vis[:, 1:] & ok[:, :-1]
            dn_obs = (k_obs[:, 1:] - k_obs[:, :-1]) * L
            dn_st = (k_st[:, 1:] - k_st[:, :-1]) * L
            lw[p["panel"]] = 1 - np.abs(dn_st - dn_obs)[trans].sum() / (np.abs(dn_obs)[known].sum() + 1e-9)
        return st, lw

    nets = []
    for path, width in CKPTS:
        net = m3.Net(10, width).to(m3.DEV)
        net.load_state_dict(torch.load(path, map_location=m3.DEV))
        net.eval()
        nets.append(net)
    st0, lw0 = score(nets)
    print(f"before TTT: S_state~ {family_mean(st0):.4f}  S_LWR~ {family_mean(lw0):.4f} ({time.time() - t_start:.0f}s)", flush=True)

    per_panel = os.environ.get("TTT_PER_PANEL", "") == "1"     # note 173: one adaptation per (member, panel)
    groups = [(p_["panel"], [p_]) for p_ in parts] if per_panel else [(None, parts)]
    tuned = {} if per_panel else []
    for s in range(TTT_SEEDS):
      gen = torch.Generator(device=m3.DEV).manual_seed(s)
      brng = np.random.default_rng(1 + 100 * s)
      for gname, gparts in groups:
        for path, width in CKPTS:
            net = m3.Net(10, width).to(m3.DEV)
            net.load_state_dict(torch.load(path, map_location=m3.DEV))
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=TTT_LR, weight_decay=1e-4)
            sched = make_sched(opt, TTT_STEPS)
            tot = 0.0
            for step in range(TTT_STEPS):
                x, y, _, e, rates = sample_with_rates(gparts, brng)
                x, m = ttt_hide(x, e, gen, rates)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(x)
                loss = m3.loss_fn(out.float(), y, m)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if sched is not None:
                    sched.step()
                tot += loss.item()
            net.eval()
            if per_panel:
                tuned.setdefault(gname, []).append(net)
            else:
                tuned.append(net)
            print(f"TTT seed {s} {Path(path).name} {gname or 'pooled'}: {TTT_STEPS} steps, "
                  f"mean loss {tot / TTT_STEPS:.4f} ({time.time() - t_start:.0f}s)", flush=True)
    st1, lw1 = score(tuned)
    print(f"\n== {split}: in-month pseudo T1/T3 (20% hidden), ens3 + month base ==")
    for pn in st0:
        print(f"  {pn:<11} S_state~ {st0[pn]:.4f} -> {st1[pn]:.4f}   S_LWR~ {lw0[pn]:.4f} -> {lw1[pn]:.4f}")
    s0, s1, l0, l1 = family_mean(st0), family_mean(st1), family_mean(lw0), family_mean(lw1)
    print(f"  family mean: S_state~ {s0:.4f} -> {s1:.4f} ({s1 - s0:+.4f})  S_LWR~ {l0:.4f} -> {l1:.4f} ({l1 - l0:+.4f})  "
          f"=> total ~ {0.35 * (s1 - s0) + 0.10 * (l1 - l0):+.4f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
