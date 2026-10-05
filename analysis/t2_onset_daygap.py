"""Task 2 onset: why are the transformers level with LightGBM on days 1-14 but far below it on later days (note 46)?

Hypothesis: LightGBM H carries the train recurrence priors (table priors at scoring, prior dropout in training), the
transformers carry none. If later-day onset events sit mostly at train-frequent bottlenecks, the priors decide them.
From the caches (the same blank-aware official-like windows as note 46), per window:
    IoU of H, the base transformer (s0) and the onset-fine-tuned transformer (Ada C2), expected-IoU ratio rule
    'recurrent': a true T+30 cell lies within +-1 link of train's 10 most frequent first-onset run ends
Mean IoU by day group (d1-14 / d15+) x recurrent / not, pooled over panels, both splits; plus the origin hours.

    .venv/Scripts/python.exe analysis/t2_onset_daygap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_onset_instability import candidates_of  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE  # noqa: E402

MODELS = {"H": None, "tfm s0": "t2_cnn_lov_tfm", "tfm ft (C2)": "t2_cnn_lov_tfm_ft_ononly"}


def main() -> None:
    rows = []  # (split, late, recurrent, hour, {model: iou})
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        cand = candidates_of(first_onsets(ctx.vf, ctx.z, truth=True)["cells"], ctx.S)
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            if not len(c["on_d"]):
                continue
            ious = {}
            for name, tag in MODELS.items():
                p = c["on_p"] if tag is None else np.load(CACHE / f"t2cnn_{tag}_{split}_{panel}.npz")["on_p"]
                pr = np.zeros(c["on_fut"].shape, bool)
                pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                ious[name] = iou(pr, c["on_fut"], c["on_efut"])
            truth = c["on_fut"][:, F - 1]
            rec = (truth & cand[None]).any(1)
            for k in range(len(c["on_d"])):
                rows.append((split, c["on_d"][k] >= 14, bool(rec[k]), {n: v[k] for n, v in ious.items()}))
    print("mean onset IoU (n windows) by day group x recurrent location, pooled over panels")
    for split in ("validation", "private", "both"):
        for late in (False, True):
            for rec in (True, False):
                sel = [r for r in rows if (split == "both" or r[0] == split) and r[1] == late and r[2] == rec]
                if not sel:
                    continue
                line = "  ".join(f"{n} {np.mean([r[3][n] for r in sel]):.3f}" for n in MODELS)
                print(f"  {split:<10} {'d15+ ' if late else 'd1-14'} {'recurrent' if rec else 'other    '} (n={len(sel):3d}): {line}")
        sel = [r for r in rows if split == "both" or r[0] == split]
        for late in (False, True):
            s = [r for r in sel if r[1] == late]
            print(f"  {split:<10} {'d15+ ' if late else 'd1-14'} share recurrent {np.mean([r[2] for r in s]):.2f} (n={len(s)})")


if __name__ == "__main__":
    main()
