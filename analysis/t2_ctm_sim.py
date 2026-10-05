"""Task 2 PoC: forecast queues by simulating the generator's physics forward from the history state.

The generator (DLSim4GMNS) is a kinematic-wave model: triangular FD per link (fd_parameters), lane-proportional
merges, fixed off-ramp split fractions, 6-s steps. Its parameters are network properties, so they transfer across
simulation runs, unlike the queue maps the learned priors memorise (notes 17-20). This script runs a
cell-transmission model (Godunov discretisation of the same LWR model) from the state at T-5 through T+30:
  initial density   last visible k = q/v in the last 3 history slots, else interpolated along the corridor,
                    else the train weekday x tod profile scaled by today's level
  upstream demand   link 0's level today x train flow profile at each future slot
  on-ramp demand    train on-ramp profile (weekday x tod) x today's corridor level (no same-day ramp data)
  off-ramps         per-link split fraction off_q / q, estimated on train
  merges            Daganzo merge, mainline priority lanes / (lanes + 1); ramp and upstream point queues
  CFL               links shorter than vf * 6 s are lengthened to that (a handful of 25-m connectors)
Queued in the simulation = FD speed <= 0.6 vf. Evaluated as a direct rule on pseudo windows (days < 14) in
train (same run), validation and private (new runs):
  onset   : top-k links by simulated k / k_queue at T+30 (k = 1, 2, 3), and the plain threshold rule
  ongoing : simulated queued cells over T+5..T+30, against persistence of the last visible queue flags
  hit@3   : the true onset link lies within +-1 link of a CTM top-3 link

    py analysis/t2_ctm_sim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import nanlast  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from tfb.data import CACHE, T, load  # noqa: E402
from tfb.features import interp_along  # noqa: E402
from tfb.physics import fd_params, load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

DT_H, SUB = 6 / 3600, 50  # 6-s steps, 50 per 5-min slot
EVAL_DAYS = 14


class Net:
    def __init__(self, panel: str):
        z = load(panel, "train")
        self.link_ids, self.vf = z["link_ids"], z["free_speed"]
        prm = fd_params(panel, z["link_ids"])
        self.cap, self.kj, self.kc, self.lanes = prm["cap"], prm["kjam"], prm["kcrit"], prm["lanes"]
        self.vf_fd = prm["vf"]
        self.w = self.cap / np.maximum(self.kj - self.kc, 1e-6)
        self.L = np.maximum(prm["length"], 1.05 * self.vf_fd * DT_H)
        self.k_q = self.w * self.kj / (0.6 * self.vf + self.w)  # density above which FD speed <= 0.6 vf
        r = load_ramps(panel, "train", z)
        wd = z["weekday"]
        with np.errstate(all="ignore"):
            on = np.where(r["on_v"], r["on_q"], np.nan)
            self.ramp_prof = np.nan_to_num(np.stack([np.nanmean(on[wd == d], 0) for d in range(7)])) * r["on_att"]
            offv = r["off_v"] & np.isfinite(z["true_flow"])
            f = np.where(offv, r["off_q"], 0).sum((0, 1)) / np.where(offv, z["true_flow"], 0).sum((0, 1))
        self.f_off = np.clip(np.nan_to_num(f), 0, 0.6) * r["off_att"]
        with np.load(CACHE / f"{panel}_t1tables.npz") as t:
            self.tv, self.tq = t["tv"], t["tq"]
        self.on_att = r["on_att"]


def init_and_demand(net: Net, v, qf, ok, t0, wd):
    N, S = len(t0), len(net.vf)
    ok3 = ok[:, -3:]
    k_hist = qf[:, -3:] / np.maximum(v[:, -3:], 1.0)
    k_last = nanlast(k_hist, ok3)
    have = np.isfinite(k_last)
    k_fill, _ = interp_along(np.where(have, k_last, np.nan), have, axis=1)
    prof_h = net.tq[wd[:, None], t0[:, None] + np.arange(-3, 0)]
    with np.errstate(all="ignore"):
        r_link = np.nanmean(np.where(ok3, qf[:, -3:], np.nan) / np.maximum(prof_h, 1.0), 1)
        r_corr = np.nan_to_num(np.nanmedian(r_link, 1), nan=1.0)
    k_prof = r_corr[:, None] * net.tq[wd, t0 - 1] / np.maximum(net.tv[wd, t0 - 1], 1.0)
    k0 = np.where(have, k_last, np.where(np.isfinite(k_fill), k_fill, k_prof))
    k0 = np.clip(np.nan_to_num(k0), 0, net.kj * 0.98)
    slots = np.clip(t0[:, None] + np.arange(0, F + 1), 0, T - 1)  # T, T+5 .. T+30
    r0 = np.where(np.isfinite(r_link[:, 0]), r_link[:, 0], r_corr)
    up = r0[:, None] * net.tq[wd[:, None], slots, 0]
    ramp = r_corr[:, None, None] * net.ramp_prof[wd[:, None], slots]
    return k0.astype(np.float64), up, ramp


def mid(a, b, c):
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def simulate(net: Net, k0, up, ramp):
    """Returns mean density per slot (N, F + 1, S): slot 0 = T, slots 1..F = T+5..T+30."""
    vf, cap, kj, w, L, f = net.vf_fd, net.cap, net.kj, net.w, net.L, net.f_off
    lanes_prev = np.concatenate([net.lanes[:1], net.lanes[:-1]])
    p = lanes_prev / (lanes_prev + 1.0)
    keep = 1.0 - np.concatenate([[0.0], f[:-1]])  # mainline share leaving link s-1 into s
    k = k0.copy()
    rq = np.zeros_like(k)
    uq = np.zeros(len(k))
    out_k = np.zeros((len(k), up.shape[1], k.shape[1]))
    for j in range(up.shape[1]):
        for _ in range(SUB):
            D = np.minimum(vf * k, cap)
            Sp = np.minimum(cap, w * np.maximum(kj - k, 0))
            Dm = np.concatenate([(up[:, j] + uq / DT_H)[:, None], D[:, :-1] * keep[None, 1:]], 1)
            Dr = ramp[:, j] + rq / DT_H
            free = Dm + Dr <= Sp
            fm = np.where(free, Dm, np.minimum(mid(Dm, Sp - Dr, p * Sp), Dm))
            fm = np.maximum(fm, 0)
            fr = np.where(free, Dr, np.clip(Sp - fm, 0, Dr))
            out = np.concatenate([fm[:, 1:] / keep[None, 1:], D[:, -1:]], 1)
            k = np.clip(k + DT_H / L * (fm + fr - out), 0, kj)
            rq = np.maximum(rq + (ramp[:, j] - fr) * DT_H, 0)
            uq = np.maximum(uq + (up[:, j] - fm[:, 0]) * DT_H, 0)
            out_k[:, j] += k
    return out_k / SUB


def evaluate(net: Net, z: dict, days_max: int):
    class C:  # pseudo_windows needs only vf
        vf = net.vf
    w = pseudo_windows(C, z)
    sel = w["d"] < days_max
    wd = z["weekday"][w["d"]].astype(int)
    res = {}
    for kind in ("onset", "ongoing"):
        idx = np.flatnonzero(sel & w[kind])
        if len(idx) == 0:
            continue
        ks = []
        for j in range(0, len(idx), 300):
            b = idx[j:j + 300]
            k0, up, ramp = init_and_demand(net, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], wd[b])
            ks.append(simulate(net, k0, up, ramp)[:, 1:])
        kr = np.concatenate(ks) / net.k_q  # (n, F, S), >= 1 means simulated queue
        fut, efut = w["fut"][idx], w["efut"][idx]
        if kind == "onset":
            score30 = kr[:, -1]
            order = np.argsort(-score30, 1)
            for kk in (1, 2, 3):
                pr = np.zeros_like(fut)
                np.put_along_axis(pr[:, -1], order[:, :kk], True, 1)
                res[f"onset_top{kk}"] = iou(pr, fut, efut).mean()
            pr = np.zeros_like(fut)
            pr[:, -1] = score30 >= 1
            res["onset_thr"] = iou(pr, fut, efut).mean()
            res["onset_thr_flag"] = pr[:, -1].sum(1).mean()
            top3 = np.zeros_like(pr[:, -1])
            np.put_along_axis(top3, order[:, :3], True, 1)
            near = top3 | np.roll(top3, 1, 1) | np.roll(top3, -1, 1)
            res["hit3"] = (near & fut[:, -1]).any(1).mean()
        else:
            res["ongoing_ctm"] = iou(kr >= 1, fut, efut).mean()
            vis_q = w["ok"][idx] & (np.nan_to_num(w["v"][idx], nan=1e9) <= 0.6 * net.vf)
            last = nanlast(vis_q.astype(np.float32), w["ok"][idx]) > 0.5
            res["ongoing_persist"] = iou(np.broadcast_to(last[:, None], fut.shape), fut, efut).mean()
            res["ongoing_ctm_or_persist"] = iou((kr >= 1) | last[:, None], fut, efut).mean()
    return res


def main() -> None:
    t_start = time.time()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for panel in T2_PANELS:
        net = Net(panel)
        for split in ("train", "validation", "private"):
            z = net_z = load(panel, split)
            r = evaluate(net, net_z, EVAL_DAYS if split != "train" else 60)  # train: first 60 days, same-run reference
            for k, v in r.items():
                out.setdefault(split, {}).setdefault(k, {})[panel] = v
            print(f"[{panel} {split}] " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()) + f" ({time.time() - t_start:.0f}s)", flush=True)
            del z
    for split, r in out.items():
        print(f"\n== {split}: CTM forward simulation as a direct rule, family mean ==")
        print("  " + "  ".join(f"{k} {family_mean(v):.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
