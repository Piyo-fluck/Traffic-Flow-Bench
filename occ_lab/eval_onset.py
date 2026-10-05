"""Score the single-horizon onset models against the production onset half.

Candidate AI's onset is `0.7 * LightGBM H + 0.3 * past-day-adapted CNN`, decided at T+30 with the expected-IoU
ratio rule (c = 1.0). This compares the T+30-only models of `train_onset.py` in that same pipeline - as the CNN
slot, and beside it - on the blank-aware official-like onset windows, family mean over the eight panels.

    python occ_lab/eval_onset.py --ckpt occ_lab/models/onset30_cnn_s*.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"

W_ON = 0.70


def load(p):
    sd = torch.load(p, map_location=C.DEV)
    tfm = "gru.weight_ih_l0" in sd
    cin = sd["gru.weight_ih_l0"].shape[1] if tfm else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if tfm else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--adapt-new", nargs="*", default=[], help="tags of adapt_onset.py runs of the single-horizon "
                                                               "models (averaged)")
    ap.add_argument("--adapt-base", nargs="*", default=[], help="tags of adapt_onset.py runs of the production "
                                                                "checkpoints, as the reference")
    ap.add_argument("--extra-tags", nargs="*", default=[], help="another cached onset probability set")
    ap.add_argument("--extra-name", default="extra")
    args = ap.parse_args()
    C.cnn.REC = 0.0
    nets = [load(p) for p in args.ckpt]
    print(f"{len(nets)} single-horizon onset checkpoints", flush=True)
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref, z = W[s]["on"], W[s]["ref"], W[s]["z"]
            if not len(on["d"]):
                continue
            x = C.make_x(ctx, stat, on, np.arange(len(on["d"])), z)
            new = np.mean([C.cnn.predict(n, x)[:, C.F - 1] for n in nets], 0)
            H, cnn3, ad = ref["on_H"], ref["on_cnn3"], ref["on_ad9b"]   # already (windows, links) at T+30
            def adapted(tags):
                fs = [CACHE / f"onset_adapt_{tg}_{s}_{panel}.npz" for tg in tags]
                fs = [f for f in fs if f.exists()]
                return np.mean([np.load(f)["prob"] for f in fs], 0) if fs else None

            new_ad, base_ad = adapted(args.adapt_new), adapted(args.adapt_base)
            ex = adapted(args.extra_tags)
            nm = args.extra_name
            v = {"LightGBM H alone": H, "cnn3 alone": cnn3, "adapted CNN alone (9b)": ad,
                 "onset30 alone": new,
                 "T/AI onset: .7H + .3 ad9b": W_ON * H + (1 - W_ON) * ad,
                 ".7H + .3 cnn3": W_ON * H + (1 - W_ON) * cnn3,
                 ".7H + .3 onset30": W_ON * H + (1 - W_ON) * new,
                 ".7H + .15 ad9b + .15 onset30": W_ON * H + 0.15 * ad + 0.15 * new,
                 ".6H + .2 ad9b + .2 onset30": 0.6 * H + 0.2 * ad + 0.2 * new,
                 ".8H + .2 onset30": 0.8 * H + 0.2 * new,
                 ".6H + .4 onset30": 0.6 * H + 0.4 * new}
            if new_ad is not None:
                v["onset30 ADAPTED alone"] = new_ad
                v[".7H + .3 onset30 ADAPTED"] = W_ON * H + 0.3 * new_ad
                v[".6H + .4 onset30 ADAPTED"] = 0.6 * H + 0.4 * new_ad
                v[".7H + .15 ad9b + .15 onset30 ADAPTED"] = W_ON * H + 0.15 * ad + 0.15 * new_ad
            if ex is not None:
                v[f"{nm} alone"] = ex
                v[f".7H + .3 {nm}"] = W_ON * H + 0.3 * ex
                v[f".8H + .2 {nm}"] = 0.8 * H + 0.2 * ex
                v[f".7H + .15 ad9b + .15 {nm}"] = W_ON * H + 0.15 * ad + 0.15 * ex
                v[f".6H + .2 ad9b + .2 {nm}"] = 0.6 * H + 0.2 * ad + 0.2 * ex
                v[f".55H + .3 ad9b + .15 {nm}"] = 0.55 * H + 0.3 * ad + 0.15 * ex
            if base_ad is not None:
                v["base ADAPTED alone (our 9b rerun)"] = base_ad
                v[".7H + .3 base ADAPTED"] = W_ON * H + 0.3 * base_ad
            if new_ad is not None and base_ad is not None:
                v[".7H + .15 base + .15 onset30 ADAPTED"] = W_ON * H + 0.15 * base_ad + 0.15 * new_ad
            for n, p in v.items():
                fam.setdefault((s, n), {})[panel] = C.onset_iou(p, on).mean()
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        base = C.family_mean(fam[(s, "T/AI onset: .7H + .3 ad9b")])
        print(f"\n=== onset {s}: production (T/AI) {base:.4f} ===")
        for v, n in sorted(((C.family_mean(d), n) for (ss, n), d in fam.items() if ss == s), reverse=True):
            print(f"   {n:<32} {v:.4f} ({v - base:+.4f})")


if __name__ == "__main__":
    main()
