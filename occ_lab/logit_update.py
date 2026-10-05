"""Step 2 of the user's plan after note 86: update in logit space instead of probability space.

r = y - p0 is the negative gradient of the binary cross-entropy with respect to the LOGIT, so the gradient-boosting
update is F1 = logit(p0) + eta * r_hat, p1 = sigmoid(F1), not p1 = clip(p0 + eta * r_hat). Three updates, all on
AI's 15 features and AI's training data (the tail's masked-layer windows):

    prob      p1 = clip(p0 + eta * r_hat)                       candidate AI; r_hat fitted to y - p0 (L2)
    logit     p1 = sigmoid(logit p0 + eta * r_hat)             the same r_hat, applied as a logit step
    newton    p1 = sigmoid(logit p0 + eta * z_hat)             LogitBoost: z_hat fitted to
                                                               z = (y - p0) / (p0 (1 - p0) + eps), clipped to +-4,
                                                               with weights p0 (1 - p0)

A logit step moves p by about p (1 - p) * eta * r_hat, at most a quarter of the probability step near 0.5, so the
eta grid of the logit variants runs higher. `gate` keeps the correction only where |step| >= gate, measured in
the variant's own units.

Tests as in note 87: January -> February inside the tail (obs and full coverage), and the whole tail -> March /
April on the pseudo windows. Rescue / spoil at each variant's best setting.

    python occ_lab/logit_update.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_cov import CACHE, FNAMES, PARAMS, load_env  # noqa: E402
from residual_diag import NAMES, TAU, lg  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

IDX = [FNAMES.index(n) for n in NAMES]
EPS = 1e-3
ZMAX = 4.0
GRID = {"prob": [(e, g) for e in (0.50, 0.75, 1.00) for g in (0.0, 0.05, 0.10)],
        "logit": [(e, g) for e in (1.0, 2.0, 3.0, 4.0, 6.0) for g in (0.0, 0.05, 0.10)],
        "newton": [(e, g) for e in (0.25, 0.50, 0.75, 1.00, 1.50) for g in (0.0, 0.25, 0.50)]}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_models(X, y, p0):
    m_r = lgb.train(PARAMS, lgb.Dataset(X, label=y - p0, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=400)
    w = p0 * (1 - p0)
    z = np.clip((y - p0) / (w + EPS), -ZMAX, ZMAX)
    m_z = lgb.train(PARAMS, lgb.Dataset(X, label=z, weight=w + EPS, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=400)
    return m_r, m_z


def apply(kind, step, base, eta, g):
    s = np.where(np.abs(step) >= g, step, 0.0)
    if kind == "prob":
        return np.clip(base + eta * s, 0, 1)
    return sig(lg(base) + eta * s)


def evaluate(models, panels):
    """panels: list of (panel, f15, y, e, base). Returns {kind: {(eta, g): (family gain, rescue, spoil)}}."""
    m_r, m_z = models
    fam_b, fam, rs = {}, {}, {}
    for panel, f, y, e, base in panels:
        r = np.zeros(base.shape, np.float32)
        z = np.zeros(base.shape, np.float32)
        r[e] = m_r.predict(f[e]).astype(np.float32)
        z[e] = m_z.predict(f[e]).astype(np.float32)
        fam_b[panel] = C.iou(base >= TAU, y, e).mean()
        bw = ((base >= TAU) != y)[e]
        for kind, step in (("prob", r), ("logit", r), ("newton", z)):
            for eta, g in GRID[kind]:
                p = apply(kind, step, base, eta, g)
                fam.setdefault((kind, eta, g), {})[panel] = C.iou(p >= TAU, y, e).mean()
                w = ((p >= TAU) != y)[e]
                a = rs.setdefault((kind, eta, g), [0, 0, 0, 0])
                a[0] += int((bw & ~w).sum()); a[1] += int(bw.sum())
                a[2] += int((~bw & w).sum()); a[3] += int((~bw).sum())
    b = C.family_mean(fam_b)
    out = {}
    for (kind, eta, g), d in fam.items():
        a = rs[(kind, eta, g)]
        out.setdefault(kind, {})[(eta, g)] = (C.family_mean(d) - b, a[0] / max(a[1], 1), a[2] / max(a[3], 1))
    return b, out


def report(title, b, out):
    print(f"  {title}: blend {b:.4f}")
    for kind in ("prob", "logit", "newton"):
        best = sorted(out[kind].items(), key=lambda kv: -kv[1][0])
        (eta, g), (gain, rescue, spoil) = best[0]
        ai = out["prob"][(0.75, 0.10)] if kind == "prob" else None
        line = (f"    {kind:<7} best {gain:+.4f} at eta {eta:.2f} gate {g:.2f}  rescue {rescue:.3f} spoil {spoil:.4f}")
        if ai is not None:
            line += f"   | AI's setting {ai[0]:+.4f} rescue {ai[1]:.3f} spoil {ai[2]:.4f}"
        print(line)
        top = "  ".join(f"{e:.2f}/{gg:.2f} {v[0]:+.4f}" for (e, gg), v in best[1:4])
        print(f"            next: {top}", flush=True)


def main() -> None:
    lab = {p: np.load(CACHE / f"cov_labels_{p}.npz") for p in C.T2_PANELS}
    env = {(v, p): load_env(v, p) for v in ("obs", "full") for p in C.T2_PANELS}
    jan = {p: np.char.startswith(lab[p]["date"].astype(str), "2031-01") for p in C.T2_PANELS}

    def train_set(months):
        Xs, ys, ps = [], [], []
        for p in C.T2_PANELS:
            f, base, _ = env[("obs", p)]
            m = lab[p]["e"] & (months[p][:, None, None] if months is not None else True)
            Xs.append(f[m][:, IDX]); ys.append(lab[p]["y"][m].astype(np.float32)); ps.append(base[m])
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)

    print("=== 1. fit on 2031-01 (masked-layer windows, AI's data), score 2031-02 ===", flush=True)
    models = fit_models(*train_set(jan))
    for v in ("obs", "full"):
        panels = []
        for p in C.T2_PANELS:
            f, base, _ = env[(v, p)]
            w = ~jan[p]
            panels.append((p, f[w][..., IDX], lab[p]["y"][w], lab[p]["e"][w], base[w]))
        report(f"February, {v} coverage", *evaluate(models, panels))

    print("\n=== 2. fit on the whole tail (= AI's corrector data), score March / April pseudo windows ===", flush=True)
    models = fit_models(*train_set(None))
    for s in C.SPLITS:
        panels = []
        for p in C.T2_PANELS:
            d = scored_panel(p, s)
            if d is not None:
                panels.append((p,) + d)
        report(s, *evaluate(models, panels))


if __name__ == "__main__":
    main()
