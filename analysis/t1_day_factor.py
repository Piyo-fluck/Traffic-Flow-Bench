"""Does a day-specific demand calibration remain in the Task 1 CNN residual? (the user's proposal, note 98)

Note 96 put 90% of the Task 1 error on free-flow cells and showed neither the mask geometry nor corridor
conservation can reach it. This asks a different question: after the CNN, is there a bias that belongs to THE DAY
- "today runs 3% heavy, and 6% heavy in the evening" - that a fixed forward pass cannot express?

Task 1 is offline, so the day's own observed cells are all usable. But the CNN copies observed cells, so its
residual there is meaningless. Hence probes:

    1. hide a share of the day's visible eligible cells (the day's own regime rate, as MASK_SPEC draws it)
    2. predict them with the production CNN
    3. residual e = observed - predicted on the probes only - the targets are never touched
    4. estimate a correction from e:  scalar per day / per day x hour / per day x hour x panel
    5. apply it to the real targets and score

Stop criterion agreed with the user: if the scalar daily correction does not improve free-flow flow RMSE by at
least 0.3 veh/h/lane, the hypothesis is dropped.

    py analysis/t1_day_factor.py [ckpt] [panels...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import PANELS, T, load  # noqa: E402

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "models" / "t1_m3_all.pt"
PROBE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05   # small, so the input stays close to the real one
PANEL_ARGS = sys.argv[3:] or PANELS
RATES = {"R1": 0.20, "R2": 0.30, "R3": 0.50}
HOURS = 24
SHRINK = 30.0          # ridge / James-Stein shrinkage: a bin with n probes keeps n / (n + SHRINK) of its mean


def main() -> None:
    net = m3.Net(10, m3.WIDTH).to(m3.DEV)
    net.load_state_dict(torch.load(CKPT, map_location=m3.DEV))
    net.eval()
    rng = np.random.default_rng(0)
    acc = {}
    for panel in PANEL_ARGS:
        p = m3.panel_tensors(panel)
        z = p["z"]
        days = p["te_days"]
        lanes, vf = z["lanes"], z["free_speed"]
        ts, tq = z["true_speed"], z["true_flow"]
        reg = np.asarray(z["regime"], str)
        S = p["X"].shape[-1]
        hour = (np.arange(T) * 24 // T).astype(np.int64)
        for d in days:
            x = torch.from_numpy(p["X"][d:d + 1].astype(np.float32)).to(m3.DEV)
            vis = (p["X"][d, 2] > 0.5) & p["E"][d]                     # visible eligible, i.e. not a target
            probe = vis & (rng.random(vis.shape) < PROBE)
            xp = x.clone()
            keep = torch.from_numpy(~probe).to(m3.DEV).float()
            xp[0, 0:4] = xp[0, 0:4] * keep
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xp).float()[0].cpu().numpy()
                out0 = net(x).float()[0].cpu().numpy()
            qp = p["base_q"][d] + out[1] * m3.SCALE_Q * lanes          # probe-time prediction
            vp = p["base_v"][d] + out[0] * m3.SCALE_V
            q0 = p["base_q"][d] + out0[1] * m3.SCALE_Q * lanes         # the production prediction at targets
            v0 = p["base_v"][d] + out0[0] * m3.SCALE_V
            eq = np.where(probe, (tq[d] - qp) / lanes, 0.0)            # per lane, as the metric is
            ev = np.where(probe, ts[d] - vp, 0.0)
            n = probe.sum(); n_h = np.array([probe[hour == h].sum() for h in range(HOURS)])
            aq = eq.sum() / max(n, 1) * n / (n + SHRINK)
            av = ev.sum() / max(n, 1) * n / (n + SHRINK)
            aq_h = np.array([eq[hour == h].sum() / max(n_h[h], 1) * n_h[h] / (n_h[h] + SHRINK)
                             for h in range(HOURS)])
            av_h = np.array([ev[hour == h].sum() / max(n_h[h], 1) * n_h[h] / (n_h[h] + SHRINK)
                             for h in range(HOURS)])
            tgt = p["M"][d]
            free = tgt & (np.nan_to_num(ts[d], nan=0.0) > 0.85 * vf)
            variants = {
                "CNN": (v0, q0),
                "+ daily scalar": (v0 + av, q0 + aq * lanes),
                "+ daily hourly": (v0 + av_h[hour][:, None], q0 + aq_h[hour][:, None] * lanes),
                "+ daily scalar (flow only)": (v0, q0 + aq * lanes),
            }
            for name, (pv_, pq_) in variants.items():
                for grp, m in (("all", tgt), ("free", free)):
                    a = acc.setdefault((name, grp), [0.0, 0.0, 0])
                    a[0] += float((((pv_ - ts[d])[m]) ** 2).sum())
                    a[1] += float(((((pq_ - tq[d]) / lanes)[m]) ** 2).sum())
                    a[2] += int(m.sum())
            a = acc.setdefault(("probe stats", "all"), [0.0, 0.0, 0])
            a[0] += float(aq * n); a[1] += float(abs(aq) * n); a[2] += int(n)
        print(f"  {panel}", flush=True)

    st = acc.pop(("probe stats", "all"))
    print(f"\nprobes {st[2]:,}; mean daily flow bias {st[0] / st[2]:+.3f}, mean |bias| {st[1] / st[2]:.3f} "
          f"veh/h/lane")
    base = None
    print(f"\n{'variant':<28} {'v RMSE':>8} {'q RMSE':>8} {'q RMSE free':>12} {'delta q free':>13}")
    for name in ("CNN", "+ daily scalar", "+ daily hourly", "+ daily scalar (flow only)"):
        a, f = acc[(name, "all")], acc[(name, "free")]
        qf = np.sqrt(f[1] / f[2])
        base = qf if base is None else base
        print(f"{name:<28} {np.sqrt(a[0] / a[2]):8.3f} {np.sqrt(a[1] / a[2]):8.2f} {qf:12.2f} "
              f"{qf - base:+13.3f}")


if __name__ == "__main__":
    main()
