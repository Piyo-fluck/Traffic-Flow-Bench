"""Delta correction (note 73), checked in the exact form the official build will use.

    p_new = sigmoid(logit p_T + lambda * dt),   dt = D - median(D),   D = logit p_adapt - logit p_base   (T+30)

The mm_lab diagnostic took the median over the links eligible at T+30, which the official path does not know.
Compared here, on the official-like onset windows: median over ALL links (the official form) vs over eligible links.
p_adapt = the cached 9b model (the one candidate T ships), p_base = rec_s20 without adaptation.

    T2CNN_REC=0.3 .venv/Scripts/python.exe analysis/t2_delta_correction_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

LAMBDAS = (0.0, 0.05, 0.1, 0.15, 0.25)
CKPT = ROOT / "work" / "models_remote" / "t2_cnn_lov_rec_s20.pt"


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main() -> None:
    fam, win = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        net, _ = C.build_net(CKPT, "full")
        for split in C.SPLITS:
            w, z, ref = W[split]["on"], W[split]["z"], W[split]["ref"]
            if not len(w["d"]):
                continue
            x = C.make_x(ctx, stat, w, np.arange(len(w["d"])), z)
            pb = C.cnn.predict(net, x, None)[:, C.F - 1]
            D = logit(ref["on_ad9b"]) - logit(pb)
            el = w["efut"][:, C.F - 1]
            med_all = np.median(D, 1, keepdims=True)
            med_el = np.array([np.median(D[j][el[j]]) if el[j].any() else np.median(D[j]) for j in range(len(D))])[:, None]
            pT = C.W_ON * ref["on_H"] + (1 - C.W_ON) * ref["on_ad9b"]
            for mname, med in (("all links", med_all), ("eligible", med_el)):
                for lam in LAMBDAS:
                    p = 1 / (1 + np.exp(-(logit(pT) + lam * (D - med))))
                    s = C.onset_iou(p, w)
                    fam.setdefault((split, mname, lam), {})[panel] = s.mean()
                    win.setdefault((split, mname, lam), []).append(s)
        print(f"  {panel}", flush=True)
    for split in C.SPLITS:
        for mname in ("all links", "eligible"):
            b = C.family_mean(fam[(split, mname, 0.0)])
            b_w = np.concatenate(win[(split, mname, 0.0)])
            print(f"{split:<10} median over {mname:<9}: " + "  ".join(
                f"l={lam:g} {C.family_mean(fam[(split, mname, lam)]) - b:+.4f} "
                f"(win {C.ci(np.concatenate(win[(split, mname, lam)]) - b_w)})" for lam in LAMBDAS[1:]))


if __name__ == "__main__":
    main()
