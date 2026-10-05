"""Note 163: refit the residual corrector for candidate BD (p_PS and adapted tfm8 rebuilt on the widened mask).

Same recipe as note 126's R0: OOF = the train run's 2031-01/02 ongoing windows along the causal trajectory, the p0
the corrector is applied to (0.40 (0.20 gs1 + 0.60 adapted tfm8 + 0.20 cnn3) + 0.60 p_PS), the frozen feature
columns; here p_PS and the adapted tfm8 are the okall ones (occ_lab/cache/pseudo_oof_ssl_s<k>_okall_*,
fam_tfm_s<k>okall_oof_*). Scored on the official-like pool of both months:

    AU           AU as shipped (old corrector)
    AU + R0      note 126's refit on AU's p0 (occ_lab/models/residual_lgb_R0.txt)
    BD           okall p_PS + tfm8, old corrector            (candidate BD)
    BD + R0ok    okall p_PS + tfm8, corrector refitted on the okall p0

    .venv/Scripts/python.exe occ_lab/refit_okall.py
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
from online_adapter import ETA, GATE, global_features, load_split, mean_of  # noqa: E402
from refit_eval import PAR, W_PS, fam_lab  # noqa: E402

CACHE = LAB / "cache"
PS = (20, 21, 22, 23)
TFM = range(3, 11)


def base_of(P, tfm_ad, ps):
    return (1 - W_PS) * (0.2 * P["gs1"] + 0.6 * tfm_ad + 0.2 * P["cnn3"]) + W_PS * ps


def main() -> None:
    X, r = [], []
    for panel in C.T2_PANELS:
        z = np.load(CACHE / f"og_oof_{panel}.npz")
        P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
             "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
             "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(CACHE / f"stid_oof_{panel}.npz"), "stid")}
        fp = [CACHE / f"pseudo_oof_ssl_s{k}_okall_{panel}.npz" for k in PS]
        ft = [CACHE / f"fam_tfm_s{k}okall_oof_{panel}.npz" for k in TFM]
        if not all(f.exists() for f in fp + ft):
            print(f"  OOF {panel}: missing caches, skipped", flush=True)
            continue
        ps = np.mean([np.load(f)["pseudo"].astype(np.float32) for f in fp], 0)
        ta = np.mean([np.load(f)["og"].astype(np.float32) for f in ft], 0)
        assert ps.shape == P["gs1"].shape == ta.shape, "OOF windows do not align"
        base = base_of(P, ta, ps)
        y, e = z["y"], z["e"]
        X.append(global_features(P, base)[e]); r.append(y[e].astype(np.float32) - base[e])
    g_new = lgb.train(PAR, lgb.Dataset(np.concatenate(X), label=np.concatenate(r)), num_boost_round=400)
    g_new.save_model(str(LAB / "models" / "residual_lgb_R0okall.txt"))
    print(f"refit R0okall: {sum(len(a) for a in r):,} OOF cells", flush=True)
    g_old = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    g_r0 = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb_R0.txt"))

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            t_au = fam_lab("tfm", s, panel)
            ps_ok = np.mean([np.load(CACHE / f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}_okall_{s}_{panel}.npz")["og"]
                             .astype(np.float32) for k in PS], 0)
            t_ok = np.mean([np.load(CACHE / f"tfmps_s{k}okall_sup0.0001_{s}_{panel}.npz")["og"].astype(np.float32)
                            for k in TFM], 0)
            V = {"AU": (base_of(P, t_au, ps), g_old), "AU + R0": (base_of(P, t_au, ps), g_r0),
                 "BD": (base_of(P, t_ok, ps_ok), g_old), "BD + R0ok": (base_of(P, t_ok, ps_ok), g_new)}
            for nm, (base, g) in V.items():
                Fe = global_features(P, base)
                rh = g.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(base.shape).astype(np.float32)
                q = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                fam.setdefault((s, nm, "d15"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "d15"):
        print(f"\n=== ongoing family IoU, {'all days' if sub == 'all' else 'days 1-5'} (vs AU) ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in ("AU", "AU + R0", "BD", "BD + R0ok"):
            print(f"  {nm:<10}" + "".join(f"  {s[:4]} {C.family_mean(fam[(s, nm, sub)]):.4f} "
                                          f"({C.family_mean(fam[(s, nm, sub)]) - ref[s]:+.4f})" for s in C.SPLITS))


if __name__ == "__main__":
    main()
