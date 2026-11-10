"""Per-window, per-horizon member selection (note 141), shared by the lab evaluation and the official prediction.

bank_for(i, panel, split, ps_tag)   the run's official-like ongoing pool: members' causal lab predictions, labels
                                     from the released layer, day, origin, queue-neighbourhood signature
select_p0(M, vr, nb, d, t0, bank)    for target windows (members M (K, N, F, S), signature, day, origin): the bank
                                     windows released by each origin (earlier days, or the same day with the horizon
                                     ended before it) give per-horizon Brier competences -> weighted p0
Members, in order: gs1, tfm8 (adapted), cnn3, p_PS, occ_cnn, occ_tfm, stid. AU's p0 weights are (0.08, 0.24, 0.08,
0.60, 0, 0, 0); the selection prior adds 0.02 to the last three.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
F = C.F
NAMES = ("gs1", "tfm8", "cnn3", "ps", "occ_cnn", "occ_tfm", "stid")
WAU = np.array([0.08, 0.24, 0.08, 0.60, 0.0, 0.0, 0.0])
PRIOR = np.array([0.08, 0.24, 0.08, 0.60, 0.02, 0.02, 0.02]) / 1.06
CONFIG = dict(k=60, beta=5.0, kappa=30.0)


def signature(v, ok, vf, radius=8):
    vr = np.where(ok, v / vf, 1.0)[:, -3:, :].astype(np.float32)
    q = (np.where(ok, v, 1e9)[:, -3:, :] <= 0.6 * vf).any(1)
    nb = np.zeros_like(q)
    for r in range(-radius, radius + 1):
        nb |= np.roll(q, r, axis=1)
    return vr, nb


def tfm8_lab(split, panel, tfm_tag=""):
    """The adapted transformer family on the lab pool: AU's (tfmps_s<k>_sup0.0001) or a variant (tfmps_s<k>_<tag>)."""
    from refit_eval import fam_lab
    if not tfm_tag:
        return fam_lab("tfm", split, panel)
    return np.mean([np.load(CACHE / f"tfmps_s{k}_{tfm_tag}_{split}_{panel}.npz")["og"].astype(np.float32)
                    for k in range(3, 11)], 0)


def bank_for(i, panel, split, ps_tag="solid3", tfm_tag=""):
    from online_adapter import load_split
    from refit_eval import fam_lab
    ctx, stat, W = C.windows(i, panel)
    og = W[split]["og"]
    D = load_split(i, panel, "ssl_w192ssl_avg4")[split]
    P = D["P"]
    if ps_tag:
        ps = np.mean([np.load(CACHE / f"fam_av_s{k}_{ps_tag}_{split}_{panel}.npz")["og"].astype(np.float32)
                      for k in (20, 21, 22, 23)], 0)
    else:
        ps = D["ps"]
    M = np.stack([P["gs1"], tfm8_lab(split, panel, tfm_tag), P["cnn3"], ps, P["occ_cnn"], P["occ_tfm"], P["stid"]])
    vr, nb = signature(og["v"], og["ok"], ctx.vf)
    y, e = D["y"].astype(np.float32), D["e"].astype(np.float32)
    return dict(M=M, y=y, e=e, d=D["d"], t0=og["t0"], vr=vr, nb=nb, P=P, ps_ship=D["ps"], tfm_ship=fam_lab("tfm", split, panel),
                brier=((M - y[None]) ** 2) * e[None])


def select_p0(M, vr, nb, d, t0, bank, k=None, beta=None, kappa=None):
    k = k or CONFIG["k"]; beta = beta or CONFIG["beta"]; kappa = kappa or CONFIG["kappa"]
    p0 = np.tensordot(WAU, M, 1)
    for n in range(M.shape[1]):
        past = np.flatnonzero((bank["d"] < d[n]) | ((bank["d"] == d[n]) & (bank["t0"] + F < t0[n])))
        if not len(past):
            continue
        msk = nb[n] if nb[n].any() else np.ones_like(nb[n])
        a = vr[n][:, msk].reshape(-1)
        b = bank["vr"][past][:, :, msk].reshape(len(past), -1)
        sim = (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-6)
        sim = sim - 0.02 * np.abs(bank["t0"][past] - t0[n]) / 12.0
        j = past[np.argsort(-sim)[:k]]
        den = bank["e"][j].sum((0, 2))
        B = bank["brier"][:, j].sum((1, 3)) / np.maximum(den[None], 1)
        rel = B / np.maximum(B.min(0, keepdims=True), 1e-6) - 1
        w = PRIOR[:, None] * np.exp(-beta * rel)
        w = w / w.sum(0, keepdims=True)
        sh = len(j) / (len(j) + kappa)
        w = sh * w + (1 - sh) * WAU[:, None]
        p0[n] = np.einsum("kf,kfs->fs", w, M[:, n])
    return p0
