"""Fine-tune Chronos-2 on the train run, then score it with fm_lab/chronos_t2.py.

Zero-shot Chronos-2 cannot see a queue coming (onset 0.03 / 0.10, ongoing 0.59 / 0.58 against 0.84 / 0.82). Here the
pretrained model is adapted to this simulator: every fit day (< FIT_END) of every Task 2 panel of the train split
becomes one multivariate series, all links' v/vf over the 288 slots of the day, NaN where the release hides the
cell. Chronos-2's own trainer draws the forecast cut-offs. Days from FIT_END on are the validation series.

    python fm_lab/chronos_ft.py --mode full --lr 1e-5 --steps 3000 --out work/models/chronos2_ft_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import common as C  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402


def day_series(panel, i, source):
    ctx = C.Ctx(panel, i)
    vf = np.asarray(ctx.vf, np.float32)
    z = ctx.z
    if source == "truth":             # the simulator state wherever the link is eligible
        v, ok = z["true_speed"], ctx.el
    else:                             # the released layer, as the scored months look
        v = z["obs_speed"]
        ok = z["elig"] & np.isfinite(v)
    r = np.where(ok & np.isfinite(v), v / vf, np.nan).astype(np.float32)   # (days, T, S)
    fit = np.asarray(z["dates"]).astype(str) < FIT_END
    S = r.shape[2]
    # Chronos-2's trainer needs one n_targets for every series: 64-link crops with stride 32, as the CNN is trained
    offs = sorted(set(list(range(0, max(S - CROP, 0) + 1, CROP // 2)) + [max(S - CROP, 0)]))
    crops = lambda d: [r[d][:, o:o + CROP].T.copy() for o in offs]  # noqa: E731
    return ([c for d in np.flatnonzero(fit) for c in crops(d)], [c for d in np.flatnonzero(~fit) for c in crops(d)])


CROP = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=("full", "lora"))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--source", default="release", choices=("release", "truth"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from chronos import Chronos2Pipeline
    tr, va = [], []
    for i, panel in enumerate(C.T2_PANELS):
        a, b = day_series(panel, i, args.source)
        tr += [{"target": x} for x in a]
        va += [{"target": x} for x in b[::4]]
        print(f"{panel}: {len(a)} fit crops, {len(b)} held-out crops", flush=True)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    ft = pipe.fit(tr, prediction_length=C.F, validation_inputs=va, finetune_mode=args.mode,
                  learning_rate=args.lr, num_steps=args.steps, batch_size=args.bs,
                  output_dir=str(Path(args.out).with_suffix(".trainer")), min_past=C.H)
    ft.save_pretrained(args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
