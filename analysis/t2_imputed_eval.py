"""Task 2: re-score the models on pseudo windows whose histories look like the OFFICIAL ones (note 46).

The official window histories (window_history.parquet) are excluded from the T1 mask and are ~76% observed, like the
training windows. Our official-like pseudo windows are cut from the masked layer and are only ~52% observed, so the
pseudo-eval scores every model on harsher inputs than the LB. Here the T1-masked history cells of the pseudo windows
are filled with our own Task 1 predictions (work/sub_m3ttt/pred_<split>_<panel>.npz, the CNN + month base + TTA of
submission L). The window SELECTION is unchanged (the same windows as work/cache/t2probs_*); only the inputs change.

Scored with both histories ("masked" = as before, "filled" = T1-filled), family mean, days 1-14 and all:
  onset   LightGBM H (5 bags, table priors), CNN x3, TFM x3, onset-fine-tuned TFM seeds (if present), and M's blend
  ongoing LightGBM I (stress), CNN x3, TFM x3, and M's blend (0.5 I + 0.5 CNN, tau 0.60)

    .venv/Scripts/python.exe analysis/t2_imputed_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

MR = ROOT / "work" / "models_remote"
NETS = {"cnn": ("cnn", [MR / "t2_cnn_lov.pt", MR / "t2_cnn_lov_s1.pt", MR / "t2_cnn_lov_s2.pt"]),
        "tfm": ("tfm", [MR / "t2_cnn_lov_tfm.pt", MR / "t2_cnn_lov_tfm_s1.pt", MR / "t2_cnn_lov_tfm_s2.pt"])}
FT = [p for p in (ROOT / "work" / "models" / "t2_cnn_lov_tfm_lw0.5_ft_ononly_s1.pt",
                  ROOT / "work" / "models" / "t2_cnn_lov_tfm_lw0.5_ft_ononly_s2.pt",
                  MR / "t2_cnn_lov_tfm_ft_ononly.pt") if p.exists()]
if FT:
    NETS["tfm_ft"] = ("tfm", FT)
for name, fn in (("tfm_A", "t2_cnn_lov_tfm_on0.5.pt"), ("tfm_B", "t2_cnn_lov_tfm_on0.5_lw0.5.pt"),
                 ("tfm_C2", "t2_cnn_lov_tfm_ft_ononly.pt")):  # the Ada onset-focused variants (note 46), single models
    if (MR / fn).exists():
        NETS[name] = ("tfm", [MR / fn])
# Transformers with the position + train-prior channels (T2CNN_PRIORS=1; 20 input channels), scored with table priors
for name, fn in (("tfmP_on s0", "t2_cnn_pri_lov_tfm_on0.5.pt"), ("tfmP_on s1", "t2_cnn_pri_lov_tfm_on0.5_s1.pt"),
                 ("tfmP_on s2", "t2_cnn_pri_lov_tfm_on0.5_s2.pt"), ("tfmP s0", "t2_cnn_pri_lov_tfm.pt"),
                 ("tfmP_B s0", "t2_cnn_pri_lov_tfm_on0.5_lw0.5.pt"), ("tfmP_B s1", "t2_cnn_pri_lov_tfm_on0.5_lw0.5_s1.pt")):
    if (MR / fn).exists():
        NETS[name] = ("tfm_pri", [MR / fn])
DAY_SETS = {"d1-14": 14, "all": 31}


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def load_nets():
    out = {}
    for name, (arch, paths) in NETS.items():
        nets = []
        for p in paths:
            cin = 12 + 1 + cnn.N_PRI if arch == "tfm_pri" else 12  # position + prior channels
            net = (cnn.T2Tfm if arch in ("tfm", "tfm_pri") else cnn.T2Net)(cin, 64).to(cnn.DEV)
            net.load_state_dict(torch.load(p, map_location=cnn.DEV))
            net.eval()
            nets.append(net)
        out[name] = nets
    return out


def filled(z, panel, split):
    """Copies of obs speed / flow with the eligible-but-unobserved cells filled by our T1 predictions."""
    with np.load(ROOT / "work" / "sub_m3ttt" / f"pred_{split}_{panel}.npz") as f:
        pv, pq = f["pv"], f["pq"]
    assert pv.shape == z["obs_speed"].shape, (pv.shape, z["obs_speed"].shape)
    hole = z["elig"] & ~(np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"]))
    zs = dict(z)
    zs["obs_speed"] = np.where(hole, pv, z["obs_speed"]).astype(np.float32)
    zs["obs_flow"] = np.where(hole, pq, z["obs_flow"]).astype(np.float32)
    return zs


def rehist(z, d, t0):
    hi = t0[:, None] + np.arange(-12, 0)
    v, qf = z["obs_speed"][d[:, None], hi], z["obs_flow"][d[:, None], hi]
    ok = z["elig"][d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)
    return v, qf, ok


def main() -> None:
    t_start = time.time()
    m_on = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]
    m_og = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2robust_bag5_stress" / "t2_ongoing.txt"))
    nets = load_nets()
    print(f"nets: {[(k, len(v)) for k, v in nets.items()]}; onset bags {len(m_on)}", flush=True)
    res: dict = {}
    obs_share: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        for split in ("validation", "private"):
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            og, _ = ongoing_official(ctx, z, blank)
            zf = filled(z, panel, split)
            for hname, zz in (("masked", z), ("filled", zf)):
                # ---- onset
                if len(on["d"]):
                    v, qf, ok = rehist(zz, on["d"], on["t0"])
                    obs_share.setdefault(hname, []).append(ok.mean())
                    grp = (z["weekday"][on["d"]] >= 5).astype(int)
                    key = np.stack([on["d"], grp], 1)
                    _, X = feats_on(ctx, False, v, qf, ok, on["t0"], grp, table_prior(ctx), key, z["weekday"][on["d"]].astype(int))
                    probs = {"lgbm": np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])}
                    x = cnn.make_x(ctx, stat, v, qf, ok, on["t0"], z["weekday"][on["d"]] >= 5)
                    pri = table_prior(ctx)  # scoring-time train-table priors, as LightGBM H and score_extra use
                    extra = cnn.prior_extra(ctx.S, len(on["d"]), on["t0"], lambda name, t: pri(name, key, t))
                    x_pri = cnn.make_x(ctx, stat, v, qf, ok, on["t0"], z["weekday"][on["d"]] >= 5, extra)
                    for name, ns in nets.items():
                        xx = x_pri if NETS[name][0] == "tfm_pri" else x
                        probs[name] = np.mean([cnn.predict(n, xx) for n in ns], 0)[:, F - 1]
                    probs["M blend"] = 0.7 * probs["lgbm"] + 0.3 * probs["cnn"]
                    if "tfm_ft" in probs:
                        probs["lgbm+tfm_ft 0.5"] = 0.5 * probs["lgbm"] + 0.5 * probs["tfm_ft"]
                    for name, p in probs.items():
                        pr = np.zeros(on["fut"].shape, bool)
                        pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                        s = iou(pr, on["fut"], on["efut"])
                        for ds, dmax in DAY_SETS.items():
                            m = on["d"] < dmax
                            res.setdefault(("onset", hname, split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
                # ---- ongoing
                if len(og["d"]):
                    v, qf, ok = rehist(zz, og["d"], og["t0"])
                    grp = (z["weekday"][og["d"]] >= 5).astype(int)
                    key = np.stack([og["d"], grp], 1)
                    wdo = z["weekday"][og["d"]].astype(int)
                    pl = []
                    for j in range(0, len(og["d"]), 200):
                        b = slice(j, j + 200)
                        _, Xh, _ = feats_og(ctx, True, v[b], qf[b], ok[b], og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
                        pl.append(m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3]))
                    probs = {"lgbm": np.concatenate(pl)}
                    x = cnn.make_x(ctx, stat, v, qf, ok, og["t0"], z["weekday"][og["d"]] >= 5)
                    for name in ("cnn", "tfm"):
                        probs[name] = np.mean([cnn.predict(n, x) for n in nets[name]], 0)
                    probs["M blend"] = 0.5 * probs["lgbm"] + 0.5 * probs["cnn"]
                    probs["lgbm+tfm"] = 0.5 * probs["lgbm"] + 0.5 * probs["tfm"]
                    for name, p in probs.items():
                        tau = 0.60 if "+" in name or "blend" in name else 0.65
                        s = iou(p >= tau, og["fut"], og["efut"])
                        for ds, dmax in DAY_SETS.items():
                            m = og["d"] < dmax
                            res.setdefault(("ongoing", hname, split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
        print(f"[{panel}] ({time.time() - t_start:.0f}s)", flush=True)
    print("\nobserved share of onset history cells: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in obs_share.items()))
    for kind in ("onset", "ongoing"):
        print(f"\n== {kind}: family-mean IoU, masked -> filled histories ==")
        names = list(res[(kind, "masked", "validation", "d1-14")].keys())
        for name in names:
            cells = []
            for split in ("validation", "private"):
                for ds in DAY_SETS:
                    a = fmean(res[(kind, "masked", split, ds)][name])
                    b = fmean(res[(kind, "filled", split, ds)][name])
                    cells.append(f"{split[:4]} {ds} {a:.3f}->{b:.3f}")
            print(f"  {name:<16} " + " | ".join(cells))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
