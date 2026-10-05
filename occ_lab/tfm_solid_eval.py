"""SOLID v3 on the transformer family (occ_lab/pseudo_adapt_tfm.py --solid3) in AU's stack and inside AX.

    AU                         shipped p_PS, shipped adapted tfm8
    tfm8 -> SOLID v3           only the transformer family replaced
    AX                         p_PS -> SOLID v3 and per-window selection (note 141)
    AX + tfm8 SOLID v3         both members SOLID v3, then the selection

    .venv/Scripts/python.exe occ_lab/tfm_solid_eval.py
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
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, single = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s in C.SPLITS:
            B0 = SB.bank_for(i, panel, s, "solid3", "")
            B1 = SB.bank_for(i, panel, s, "solid3", "solid3")
            yb, eb, d = B0["y"] > 0.5, B0["e"] > 0.5, B0["d"]
            M0 = B0["M"].copy()
            M0[3] = B0["ps_ship"]
            au = np.tensordot(SB.WAU, M0, 1)
            M_t = M0.copy()
            M_t[1] = B1["M"][1]
            V = {"AU": corr(au, B0["P"]), "tfm8 -> SOLID v3": corr(np.tensordot(SB.WAU, M_t, 1), B0["P"]),
                 "AX": corr(SB.select_p0(B0["M"], B0["vr"], B0["nb"], d, B0["t0"], B0), B0["P"]),
                 "AX + tfm8 SOLID v3": corr(SB.select_p0(B1["M"], B1["vr"], B1["nb"], d, B1["t0"], B1), B1["P"])}
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, yb, eb)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
            for nm, q in (("tfm8 shipped", B0["tfm_ship"]), ("tfm8 SOLID v3", B1["M"][1])):
                sc = C.iou(q >= 0.5, yb, eb)
                single.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                single.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "days 1-5"):
        print(f"\n=== {sub} ===")
        for nm in ("tfm8 shipped", "tfm8 SOLID v3"):
            print(f"  single {nm:<18}" + "".join(f"  {s[:4]} {C.family_mean(single[(s, nm, sub)]):.4f}" for s in C.SPLITS))
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in ("AU", "tfm8 -> SOLID v3", "AX", "AX + tfm8 SOLID v3"):
            print(f"  stack  {nm:<18}" + "".join(f"  {s[:4]} {C.family_mean(fam[(s, nm, sub)]):.4f} "
                                                 f"({C.family_mean(fam[(s, nm, sub)]) - ref[s]:+.4f})" for s in C.SPLITS))


if __name__ == "__main__":
    main()
