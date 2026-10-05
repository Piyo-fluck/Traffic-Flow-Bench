"""Is the synthetic generator a cell-transmission (Godunov LWR) model with the released FD parameters?

On train truth, for interior mainline links, compare the accumulation change
over a span of m steps with the net boundary flux integrated over those
steps, under several flux models:
  zero     : dN_pred = 0
  linkflow : inflow = q_{i-1}, outflow = q_i
  topo     : inflow = q_{i-1}, outflow = q_{i+1}   (the official topology fallback)
  ctm      : Godunov flux f_{j->j+1} = min(D_j, S_{j+1}), D = min(vf k, cap), S = min(cap, w (kjam - k))
Released valid ramp flows enter every model. Truth is the noisy observation,
so longer spans (m = 3, 6, 12) raise the signal above the endpoint noise.
Also reports how far the truth sits from the triangular FD.

    py analysis/phys_eda01_ctm_check.py D12_I5_N D7_I210_W
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402
from tfb.physics import fd_params, load_ramps  # noqa: E402

DT = 5 / 60


def main() -> None:
    for panel in sys.argv[1:] or ["D12_I5_N", "D7_I210_W"]:
        z = load(panel, "train")
        prm = fd_params(panel, z["link_ids"])
        r = load_ramps(panel, "train", z)
        v, q = z["true_speed"], z["true_flow"]
        use = z["elig"] & np.isfinite(v) & np.isfinite(q)
        k = q / np.maximum(v, 1.0)
        L = prm["length"]
        N = k * L
        vf, cap, kc, kj = prm["vf"], prm["cap"], prm["kcrit"], prm["kjam"]
        w = cap / np.maximum(kj - kc, 1e-6)
        D = np.minimum(vf * k, cap)
        Sup = np.minimum(cap, w * np.maximum(kj - k, 0))
        ramp_ok = (r["on_v"] | ~r["on_att"]) & (r["off_v"] | ~r["off_att"])
        net_ramp = np.where(r["on_v"], r["on_q"], 0) - np.where(r["off_v"], r["off_q"], 0)

        # Fluxes at time t for interior links i = 1..S-2 (index 0 = the link itself after slicing).
        f_ctm_in = np.minimum(D[:, :, :-2], Sup[:, :, 1:-1])
        f_ctm_out = np.minimum(D[:, :, 1:-1], Sup[:, :, 2:])
        flux = {
            "zero": np.zeros_like(k[:, :, 1:-1]),
            "linkflow": q[:, :, :-2] - q[:, :, 1:-1],
            "topo": q[:, :, :-2] - q[:, :, 2:],
            "ctm": f_ctm_in - f_ctm_out,
        }
        ok_t = use[:, :, 1:-1] & use[:, :, :-2] & use[:, :, 2:] & ramp_ok[:, :, 1:-1]
        Ni = N[:, :, 1:-1]
        rp = net_ramp[:, :, 1:-1]
        print(f"\n== {panel}: {k.shape[2]} links; median link length {np.median(L):.3f} km ==")
        # FD fit of the truth, per lane.
        lanes = prm["lanes"]
        kl = k / lanes
        qfd = np.where(kl <= kc / lanes, vf * kl, (cap / lanes) / np.maximum((kj - kc) / lanes, 1e-9) * np.maximum(kj / lanes - kl, 0))
        dev = np.abs(q / lanes - qfd)[use] / np.maximum((q / lanes)[use], 1)
        cong = (k > kc)[use]
        print(f"truth vs triangular FD: |q - Q_FD(k)|/q median free {np.median(dev[~cong]):.4f}, congested {np.median(dev[cong]):.4f} "
              f"(share congested {cong.mean():.3f}) | free-flow v/vf median {np.median((v / vf)[use & (k <= kc)]):.4f}")
        for m in (1, 3, 6, 12):
            ok = ok_t[:, :-m].copy()
            for j in range(1, m + 1):
                ok &= ok_t[:, j:ok_t.shape[1] - m + j] if j < m else use[:, m:, 1:-1]
            dN = Ni[:, m:] - Ni[:, :-m]
            line = []
            for name, fx in flux.items():
                integ = sum((fx[:, j:fx.shape[1] - m + j] + rp[:, j:rp.shape[1] - m + j]) for j in range(m)) * DT
                if name == "zero":
                    integ = np.zeros_like(dN)
                res = np.abs(dN - integ)[ok].sum() / (np.abs(dN)[ok].sum() + 1e-9)
                line.append(f"{name} {res:.3f}")
            sig = np.median(np.abs(dN)[ok] / np.maximum(Ni[:, :-m][ok], 1e-6))
            print(f"  span {m:>2} steps ({5 * m:>2} min): n={int(ok.sum()):>9,}  median|dN|/N {sig:.4f}  |  E = " + "  ".join(line))


if __name__ == "__main__":
    main()
