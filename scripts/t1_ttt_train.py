"""Production Task 1 test-time training: fine-tune the 3 production CNNs on one scored month (note 32).

Same recipe as analysis/t1_ttt.py (validated there: +0.0019 S_state~, +0.0033 S_LWR~ on validation): 400 steps,
lr 2e-4, 10-50% of the month's visible eligible cells hidden per sample, fixed seeds. Two differences: no cells
are held out for evaluation (every visible cell is used), and the inputs are built exactly like
scripts/t1_predict_ckpt.py with --month-kappa 2 --month-kappa-q 10 (train table + month deviation on the full
masked layer). T1 is offline, so the whole scored month may be used.

Writes <out-dir>/ttt_<split>_<i>.pt for the checkpoints in CKPTS order; use them with t1_predict_ckpt.py --splits <split>.

    py scripts/t1_ttt_train.py --split validation --out-dir work/sub_m3ttt
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from t1_ttt import CKPTS, TTT_LR, TTT_STEPS, ttt_hide  # noqa: E402
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402

# TTT_ES=1 (note 173): per-panel TTT with early stopping. Per (member, panel) a share ES_FRAC of the visible cells,
# in the official target geometry (another day's target mask), is held out of the TTT input; every ES_EVERY steps
# up to ES_MAX the held-out cells are predicted and scored (0.35 S_state + 0.10 S_LWR proxy, as analysis/t1_ttt.py);
# the best step, 0 included (= no TTT for that panel), is the checkpoint saved.
ES_MAX = int(os.environ.get("ES_MAX", 600))
ES_EVERY = int(os.environ.get("ES_EVERY", 50))
ES_FRAC = float(os.environ.get("ES_FRAC", 0.5))


def es_part(panel, split, z, tv, tq, seed):
    """the panel's TTT part with held-out cells removed, and what is needed to score them"""
    rng = np.random.default_rng(seed)
    vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
    nd = len(z["dates"])
    shift = rng.integers(1, nd, size=nd)
    hide = vis & np.asarray(z["target"], bool)[(np.arange(nd) + shift) % nd] & (rng.random(vis.shape) < ES_FRAC)
    zm = dict(z)
    for k in ("obs_speed", "obs_flow", "obs_occ"):
        zm[k] = np.where(hide, np.nan, z[k]).astype(np.float32)
    bv, bq = tv[z["weekday"]], tq[z["weekday"]]
    dv, dq = month_deviation(zm, bv, bq, np.ones(nd, bool), 2.0, kappa_q=10.0)
    bv, bq = bv + dv, bq + dq
    vism = vis & ~hide
    Y = np.stack([np.where(vism, (zm["obs_speed"] - bv) / m3.SCALE_V, 0),
                  np.where(vism, (zm["obs_flow"] - bq) / z["lanes"] / m3.SCALE_Q, 0)], axis=1).astype(np.float16)
    part = dict(panel=panel, X=m3.make_inputs(zm, bv, bq), Y=Y, M=np.zeros(vis.shape, bool), E=vism,
                tr_days=np.arange(nd), base_v=bv, base_q=bq, z=zm)
    ev = dict(hide=hide, vis=vis, ov=z["obs_speed"], oq=z["obs_flow"], lanes=z["lanes"],
              L=fd_params(panel, z["link_ids"])["length"], ok=ramp_ok(panel, split, z))
    return part, ev


def es_score(net, part, ev):
    net.eval()
    pv, pq = m3.predict_days(net, part, range(len(part["tr_days"])))
    net.train()
    h = ev["hide"]
    rv = float(np.sqrt(np.mean((pv - ev["ov"])[h] ** 2)))
    rq = float(np.sqrt(np.mean(((pq - ev["oq"]) / ev["lanes"])[h] ** 2)))
    st = SPEED_W * max(0, 1 - rv / SPEED_NORM) + FLOW_W * max(0, 1 - rq / FLOW_NORM)
    k_obs = ev["oq"] / np.maximum(ev["ov"], 1.0)
    k_st = np.where(h, pq / np.maximum(pv, 1.0), k_obs)
    vis, ok, L = ev["vis"], ev["ok"], ev["L"]
    trans = vis[:, :-1] & vis[:, 1:] & ok[:, :-1] & (h[:, :-1] | h[:, 1:])
    known = vis[:, :-1] & vis[:, 1:] & ok[:, :-1]
    dn_obs = (k_obs[:, 1:] - k_obs[:, :-1]) * L
    dn_st = (k_st[:, 1:] - k_st[:, :-1]) * L
    lw = 1 - np.abs(dn_st - dn_obs)[trans].sum() / (np.abs(dn_obs)[known].sum() + 1e-9)
    return 0.35 * st + 0.10 * lw, st, lw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    parts = []
    for panel in PANELS:
        with np.load(CACHE / f"{panel}_t1tables.npz") as f:
            tv, tq = f["tv"], f["tq"]
        z = load(panel, args.split)
        bv, bq = tv[z["weekday"]], tq[z["weekday"]]
        dv, dq = month_deviation(z, bv, bq, np.ones(len(z["dates"]), bool), 2.0, kappa_q=10.0)
        bv, bq = bv + dv, bq + dq
        vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        Y = np.stack([np.where(vis, (z["obs_speed"] - bv) / m3.SCALE_V, 0),
                      np.where(vis, (z["obs_flow"] - bq) / z["lanes"] / m3.SCALE_Q, 0)], axis=1).astype(np.float16)
        parts.append(dict(panel=panel, X=m3.make_inputs(z, bv, bq), Y=Y, M=np.zeros(vis.shape, bool), E=vis,
                          tr_days=np.arange(len(z["dates"]))))
        print(f"[{args.split} {panel}] ready ({time.time() - t_start:.0f}s)", flush=True)
    gen = torch.Generator(device=m3.DEV).manual_seed(0)
    brng = np.random.default_rng(1)
    if os.environ.get("TTT_ES", "") == "1":
        only = [x for x in os.environ.get("TTT_ONLY", "").split(",") if x]
        for j, panel in enumerate(PANELS):
            with np.load(CACHE / f"{panel}_t1tables.npz") as f:
                tv, tq = f["tv"], f["tq"]
            part, ev = es_part(panel, args.split, load(panel, args.split), tv, tq, seed=1000 + j)
            for i, (path, width) in enumerate(CKPTS):
                if only and str(i) not in only:
                    continue
                net = m3.Net(10, width).to(m3.DEV)
                net.load_state_dict(torch.load(ROOT / path, map_location=m3.DEV))
                net.train()
                opt = torch.optim.AdamW(net.parameters(), lr=TTT_LR, weight_decay=1e-4)
                j0, s0, l0 = es_score(net, part, ev)
                best = (j0, 0, {k: v.detach().clone() for k, v in net.state_dict().items()})
                curve = [f"0:{j0:.5f}"]
                for step in range(1, ES_MAX + 1):
                    x, y, _, e, _w = m3.sample_batch([part], brng)
                    x, m = ttt_hide(x, e, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = net(x)
                    loss = m3.loss_fn(out.float(), y, m)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    if step % ES_EVERY == 0:
                        jj, _, _ = es_score(net, part, ev)
                        curve.append(f"{step}:{jj:.5f}")
                        if jj > best[0]:
                            best = (jj, step, {k: v.detach().clone() for k, v in net.state_dict().items()})
                dst = args.out_dir / f"ttt_{args.split}_{i}_{panel}.pt"
                torch.save(best[2], dst)
                print(f"[{args.split} {panel} m{i}] best step {best[1]} J {best[0]:.5f} (step 0 {j0:.5f}: S_state~ "
                      f"{s0:.4f} S_LWR~ {l0:.4f}) curve {' '.join(curve)} ({time.time() - t_start:.0f}s)", flush=True)
        return
    # TTT_PER_PANEL=1: one adaptation per (checkpoint, panel) on that panel's month only - the run shift is
    # corridor-specific (T2 note 134: per-panel adaptation beats pooled even with 4x the steps)
    per_panel = os.environ.get("TTT_PER_PANEL", "") == "1"
    groups = [(p_["panel"], [p_]) for p_ in parts] if per_panel else [(None, parts)]
    only = [x for x in os.environ.get("TTT_ONLY", "").split(",") if x]          # optional subset of checkpoints
    for i, (path, width) in enumerate(CKPTS):
      if only and str(i) not in only:
        continue
      for gname, gparts in groups:
        net = m3.Net(10, width).to(m3.DEV)
        net.load_state_dict(torch.load(ROOT / path, map_location=m3.DEV))
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=TTT_LR, weight_decay=1e-4)
        tot = 0.0
        for _ in range(TTT_STEPS):
            x, y, _, e, _w = m3.sample_batch(gparts, brng)   # m3.sample_batch gained a weight slot (note 101)
            x, m = ttt_hide(x, e, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            loss = m3.loss_fn(out.float(), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        dst = args.out_dir / (f"ttt_{args.split}_{i}_{gname}.pt" if gname else f"ttt_{args.split}_{i}.pt")
        torch.save(net.state_dict(), dst)
        print(f"[{args.split}] {path} -> {dst.name}: {TTT_STEPS} steps, mean loss {tot / TTT_STEPS:.4f} "
              f"({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
