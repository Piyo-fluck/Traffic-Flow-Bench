"""Can a non-parametric memory of past CNN residuals predict the residual at a new cell?

The user's proposal (note 103): rather than another parametric model, keep the CNN and look up cells whose
situation resembles the query's, using the CNN's own penultimate representation as the search key, then correct
by the retrieved residuals:

    r_hat(j) = sum_k w_k r_k over the nearest neighbours of j,     q_hat(j) = q_CNN(j) + alpha r_hat(j)

This is the oracle diagnostic before any of that is built. Database = the January target cells of the local test
block (truth known, out of fold for the CNN), queries = the February ones. If the correlation between the
retrieved residual and the true residual is near zero, the branch closes immediately.

A random-neighbour control is reported beside it, because any weighted average of residuals shrinks towards the
mean and can look helpful for the wrong reason.

    py analysis/t1_knn_residual.py [n_db] [n_query]
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
from tfb.data import PANELS  # noqa: E402

CKPT = ROOT / "work" / "models" / "t1_m3_all.pt"
N_DB = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
KS = (1, 5, 20, 100)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
CHUNK = 256


@torch.no_grad()
def embed_day(net, x):
    """(1, C, T, S) -> penultimate features (C, T, S) and the two output channels."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = net.blocks(net.inp(x))
        out = net.out(h)
    return h.float()[0], out.float()[0]


def main() -> None:
    net = m3.Net(10, m3.WIDTH).to(m3.DEV)
    net.load_state_dict(torch.load(CKPT, map_location=m3.DEV))
    net.eval()
    rng = np.random.default_rng(0)
    db_h, db_r, q_h, q_r, q_panel, db_panel = [], [], [], [], [], []
    for pi, panel in enumerate(PANELS):
        p = m3.panel_tensors(panel)
        z, lanes = p["z"], p["z"]["lanes"]
        months = np.array([d[5:7] for d in z["dates"].astype(str)])
        for d in p["te_days"]:
            x = torch.from_numpy(p["X"][d:d + 1].astype(np.float32)).to(m3.DEV)
            h, out = embed_day(net, x)
            pq = p["base_q"][d] + out[1].cpu().numpy() * m3.SCALE_Q * lanes
            r = (z["true_flow"][d] - pq) / lanes
            m = p["M"][d]
            if not m.any():
                continue
            idx = np.flatnonzero(m.ravel())
            hh = h.permute(1, 2, 0).reshape(-1, m3.WIDTH).cpu().numpy()[idx]
            rr = r.ravel()[idx]
            take = rng.random(len(idx))
            if months[d] == "01":
                sel = take < N_DB / 3.5e6
                db_h.append(hh[sel].astype(np.float16)); db_r.append(rr[sel].astype(np.float32))
                db_panel.append(np.full(int(sel.sum()), pi, np.int16))
            else:
                sel = take < N_Q / 3.5e6
                q_h.append(hh[sel].astype(np.float16)); q_r.append(rr[sel].astype(np.float32))
                q_panel.append(np.full(int(sel.sum()), pi, np.int16))
        print(f"  {panel}", flush=True)
    H = torch.from_numpy(np.concatenate(db_h)).to(m3.DEV, torch.float16)
    R = torch.from_numpy(np.concatenate(db_r)).to(m3.DEV, torch.float32)
    Q = torch.from_numpy(np.concatenate(q_h)).to(m3.DEV, torch.float16)
    qr = np.concatenate(q_r)
    dbp = torch.from_numpy(np.concatenate(db_panel)).to(m3.DEV)
    qp = np.concatenate(q_panel)
    print(f"\ndatabase {H.shape[0]:,} cells, queries {Q.shape[0]:,}; embedding dim {H.shape[1]}")
    Hn = H / H.norm(dim=1, keepdim=True).clamp(min=1e-6)
    Qn = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-6)
    out = {k: np.zeros(len(qr), np.float32) for k in KS}
    out_same = {k: np.zeros(len(qr), np.float32) for k in KS}
    for a in range(0, len(qr), CHUNK):
        sl = slice(a, a + CHUNK)
        sim = (Qn[sl] @ Hn.T).float()
        for k in KS:
            v, i = torch.topk(sim, k, dim=1)
            w = torch.softmax(v * 20.0, dim=1)
            out[k][sl] = (w * R[i]).sum(1).cpu().numpy()
        same = dbp[None, :] == torch.from_numpy(qp[sl]).to(m3.DEV)[:, None]
        sim2 = sim.masked_fill(~same, -2.0)
        for k in KS:
            v, i = torch.topk(sim2, k, dim=1)
            w = torch.softmax(v * 20.0, dim=1)
            out_same[k][sl] = (w * R[i]).sum(1).cpu().numpy()
    rnd = R[torch.randint(0, len(R), (len(qr),), device=m3.DEV)].cpu().numpy()
    base = float(np.sqrt((qr ** 2).mean()))
    print(f"\nquery flow RMSE without correction: {base:.3f} veh/h/lane "
          f"(residual sd {qr.std():.3f}, mean {qr.mean():+.3f})")
    print(f"{'retrieval':<22} {'corr':>7} {'sd':>7} " + " ".join(f"{'a=' + str(a):>9}" for a in ALPHAS))
    rows = [(f"top-{k} any panel", out[k]) for k in KS] + [(f"top-{k} same panel", out_same[k]) for k in KS] \
        + [("random neighbour", rnd)]
    for name, e in rows:
        c = float(np.corrcoef(qr, e)[0, 1])
        cells = " ".join(f"{np.sqrt(((qr - a * e) ** 2).mean()):9.3f}" for a in ALPHAS)
        print(f"{name:<22} {c:+7.3f} {e.std():7.3f} {cells}")


if __name__ == "__main__":
    main()
