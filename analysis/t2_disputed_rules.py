"""Decision rules for the ongoing cells where the three members disagree (note 80) - no new model.

The blend takes 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 and thresholds at 0.50 everywhere. On the 2.9% of cells where
the members disagree that decision is only 70% right, and the cells are near a coin flip (44-47% queued). This
script keeps the blend untouched on the agreed cells and only changes how the disputed ones are decided:

    tau X            the same blend probability at a different threshold
    majority         two of the three members at their own thresholds
    tfm8             follow the transformer ensemble, the most accurate member there
    max / min        the largest / smallest member probability at 0.50

Evaluated twice: on the train month (histories from the true layer, coverage 0.764 - the official windows'
coverage, note 76) and on the scored months' official-like pseudo windows (coverage 0.52).

    .venv/Scripts/python.exe analysis/t2_disputed_rules.py --cnn <cnn x3> --tfm <tfm s3..s10>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from t2_disputed_stack import BLEND_TAU, BLEND_W, TAU  # noqa: E402
from t2_eda02 import candidates  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402

C.cnn.REC = 0.0
H, F = C.H, C.F
TAUS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CAL = {}


def structure_rules(P, base, dis, og_hist):
    """Rules that use the SHAPE of the prediction instead of the probability level.
    og_hist = (v, ok, vf) of the window histories, for the queue edge visible at T."""
    out = {}
    q = base.copy()
    left, right = np.roll(q, 1, -1), np.roll(q, -1, -1)
    left[..., 0], right[..., -1] = q[..., 0], q[..., -1]
    fill = q.copy()
    fill[dis & left & right] = True                     # a hole between two queued links
    fill[dis & ~left & ~right] = False                  # an isolated link
    out["fill holes / drop isolated"] = fill
    up, dn = np.roll(q, 1, 1), np.roll(q, -1, 1)        # the same in time, across horizons
    up[:, 0], dn[:, -1] = q[:, 0], q[:, -1]
    tf = q.copy()
    tf[dis & up & dn] = True
    tf[dis & ~up & ~dn] = False
    out["temporal fill / drop"] = tf
    both = fill.copy()
    both[dis & up & dn] = True
    both[dis & ~up & ~dn & ~left & ~right] = False
    out["both fills"] = both
    v, ok, vf = og_hist
    queued_T = ok[:, -1] & (np.nan_to_num(v[:, -1], nan=1e9) <= 0.6 * vf)
    S = queued_T.shape[1]
    idx = np.arange(S)
    dist = np.full(queued_T.shape, S, np.float32)
    for j in range(len(queued_T)):
        qq = np.flatnonzero(queued_T[j])
        if len(qq):
            dist[j] = np.abs(idx[:, None] - qq[None, :]).min(1)
    h = np.arange(1, base.shape[1] + 1, dtype=np.float32)[None, :, None]
    for c in (0.5, 1.0, 1.5, 2.0):
        w = base.copy()
        inside = dist[:, None, :] <= c * h
        w[dis] = inside[dis]
        out[f"edge wave c={c}"] = w
    return out


def rules(P):
    """{name: bool array} for the disputed cells, from the three member probabilities."""
    bl = BLEND_W[0] * P["gs1"] + BLEND_W[1] * P["tfm8"] + BLEND_W[2] * P["cnn3"]
    out = {f"blend tau {t:.2f}": bl >= t for t in TAUS}
    votes = sum((P[n] >= TAU[n]).astype(int) for n in P)
    out["majority (2 of 3)"] = votes >= 2
    out["tfm8 alone"] = P["tfm8"] >= TAU["tfm8"]
    out["max member"] = np.maximum.reduce([P[n] for n in P]) >= 0.5
    out["min member"] = np.minimum.reduce([P[n] for n in P]) >= 0.5
    out["always queued"] = np.ones_like(bl, bool)
    out["never queued"] = np.zeros_like(bl, bool)
    return bl, out


def score(sets, name_prefix, acc):
    for panel, y, e, hist, dis, P in sets:
        bl, R = rules(P)
        base = bl >= BLEND_TAU
        if hist is not None:
            R.update(structure_rules(P, base, dis, hist))
        CAL.setdefault(name_prefix, []).append((y[dis].astype(float), bl[dis]))
        for name, dec in R.items():
            out = base.copy()
            out[dis] = dec[dis]
            a = acc.setdefault((name_prefix, name), dict(fam={}, ok=0, n=0, pos=0))
            a["fam"][panel] = C.iou(out, y, e).mean()
            a["ok"] += int((out[dis] == y[dis]).sum())
            a["n"] += int(dis.sum())
            a["pos"] += int(out[dis].sum())


def train_sets(args):
    """Train-month ongoing windows with true-layer histories (the official coverage)."""
    nets_cnn = [_load(p, "cnn") for p in args.cnn]
    nets_tfm = [_load(p, "tfm") for p in args.tfm]
    m_og = lgb.Booster(model_file=str(args.gs_models / "ongoing_g1.txt"))
    rng = np.random.default_rng(0)
    out, t0 = [], time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        c = candidates(z)
        d, t0w = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        if len(d) > args.per_panel:
            sel = np.sort(rng.choice(len(d), args.per_panel, replace=False))
            d, t0w = d[sel], t0w[sel]
        hi, fi = t0w[:, None] + np.arange(-H, 0), t0w[:, None] + np.arange(1, F + 1)
        v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
        ok = ctx.el[d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)
        x = C.cnn.make_x(ctx, stat, v, qf, ok, t0w, z["weekday"][d] >= 5)
        P = {"cnn3": np.mean([C.cnn.predict(n, x) for n in nets_cnn], 0),
             "tfm8": np.mean([C.cnn.predict(n, x) for n in nets_tfm], 0)}
        pz = np.load(args.gs_models / f"pca_{panel}.npz")
        grp = ctx.g[d]
        _, Xh, _ = feats_og(ctx, True, v, qf, ok, t0w, grp, zero_prior(ctx), np.stack([d, grp], 1).astype(int),
                            z["weekday"][d].astype(int))
        g = np.broadcast_to(gstate(ctx, (pz["mu"], pz["vt"]), v, qf, ok)[:, None, None], Xh.shape[:3] + (len(G_NAMES),))
        Xg = np.concatenate([Xh, g], -1)
        P["gs1"] = m_og.predict(Xg.reshape(-1, Xg.shape[-1])).reshape(Xh.shape[:3]).astype(np.float32)
        y, e = ctx.q[d[:, None], fi], ctx.el[d[:, None], fi]
        dis = ~((( P["gs1"] >= TAU["gs1"]) == (P["tfm8"] >= TAU["tfm8"])) &
                ((P["tfm8"] >= TAU["tfm8"]) == (P["cnn3"] >= TAU["cnn3"]))) & e
        out.append((panel, y, e, (v, ok, ctx.vf), dis, P))
        print(f"  train windows {panel}: {len(d)}, coverage {ok.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    return out


def _load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-models", type=Path, default=ROOT / "work" / "models_gstate")
    ap.add_argument("--per-panel", type=int, default=800)
    args = ap.parse_args()
    acc = {}
    score(train_sets(args), "train (coverage 0.76)", acc)
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = W[s]["og"], W[s]["ref"]
            if not len(og["d"]):
                continue
            P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"]}
            dis = ~(((P["gs1"] >= TAU["gs1"]) == (P["tfm8"] >= TAU["tfm8"])) &
                    ((P["tfm8"] >= TAU["tfm8"]) == (P["cnn3"] >= TAU["cnn3"]))) & og["efut"]
            score([(panel, og["fut"], og["efut"], (og["v"], og["ok"], ctx.vf), dis, P)], f"pseudo {s}", acc)
    for prefix, sets in CAL.items():
        y = np.concatenate([a for a, _ in sets]); pp = np.concatenate([b for _, b in sets])
        print(f"\n--- {prefix}: calibration of the blend probability ON DISPUTED CELLS ({len(y):,} cells) ---")
        for lo, hi in ((0.0,0.2),(0.2,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,0.8),(0.8,1.01)):
            m = (pp >= lo) & (pp < hi)
            if m.sum() > 50:
                print(f"   p in [{lo:.2f},{hi:.2f}): n {int(m.sum()):>6}  mean p {pp[m].mean():.3f}  actual {y[m].mean():.3f}")
    for prefix in sorted({k[0] for k in acc}):
        base = acc[(prefix, f"blend tau {BLEND_TAU:.2f}")]
        b = C.family_mean(base["fam"])
        print(f"\n=== {prefix}: blend at 0.50 family {b:.4f}, disputed accuracy {base['ok'] / max(base['n'], 1):.3f}, "
              f"{base['n']:,} disputed cells ===")
        for (pre, name), a in acc.items():
            if pre != prefix:
                continue
            print(f"   {name:<20} family {C.family_mean(a['fam']) - b:+.4f}  accuracy {a['ok'] / max(a['n'], 1):.3f}"
                  f"  queued share on disputed {a['pos'] / max(a['n'], 1):.2f}")


if __name__ == "__main__":
    main()
