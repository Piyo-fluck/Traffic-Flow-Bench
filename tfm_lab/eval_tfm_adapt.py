"""Plan B evaluation: adapt each rec-head transformer day by day (idea 9b) and score the ongoing blend of T.

Per seed (--seed), per panel and scored month, over the days with official-like windows in ascending order:
predict that day's onset and ongoing windows with the current weights, then adapt on 16 history windows drawn from
strictly earlier days (released layer, no label), 30 AdamW steps at lr 1e-4 on the reconstruction loss only
(exactly idea 9b's protocol). The same weights without adaptation give the control.
Writes tfm_lab/out/s<seed>_{adapt,base}_<split>_<panel>.npz (og_p, on_p).

    CUDA_VISIBLE_DEVICES=0 python tfm_lab/eval_tfm_adapt.py --seed 3
    python tfm_lab/eval_tfm_adapt.py --report --seeds 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "tfm"
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from tfm_rec import MODELS, T2TfmRec, rec_loss  # noqa: E402

OUT = LAB / "out"
STEPS, LR, BS = 30, 1e-4, 16


def run_seed(seed):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    torch.manual_seed(3)
    sd = torch.load(MODELS / f"t2_tfm_rec_s{seed}.pt", map_location=C.DEV)
    t0 = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for split in C.SPLITS:
            w_on, w_og, z = W[split]["on"], W[split]["og"], W[split]["z"]
            base = T2TfmRec(sd["gru.weight_ih_l0"].shape[1], C.cnn.WIDTH).to(C.DEV)
            base.load_state_dict(sd)
            base.eval()
            live = T2TfmRec(sd["gru.weight_ih_l0"].shape[1], C.cnn.WIDTH).to(C.DEV)
            live.load_state_dict(sd)
            live.eval()
            params = [p for n, p in live.named_parameters() if not n.startswith("head.")]
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
            res = {k: np.zeros((len(w["d"]), C.F, ctx.S), np.float32) for k, w in (("og", w_og), ("on", w_on))}
            resb = {k: np.zeros_like(v) for k, v in res.items()}
            days = sorted(set(w_on["d"].tolist()) | set(w_og["d"].tolist()))
            for day in days:
                for k, w in (("og", w_og), ("on", w_on)):
                    m = np.flatnonzero(w["d"] == day)
                    if len(m):
                        x = C.make_x(ctx, stat, w, m, z)
                        res[k][m] = C.cnn.predict(live, x)
                        resb[k][m] = C.cnn.predict(base, x)
                xb = C.past_batch(ctx, stat, z, np.arange(day), BS, rng)
                if xb is not None:
                    live.train()
                    for _ in range(STEPS):
                        loss = rec_loss(live, xb)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    live.eval()
            for tag, r in (("adapt", res), ("base", resb)):
                np.savez_compressed(OUT / f"s{seed}_{tag}_{split}_{panel}.npz", og_p=r["og"], on_p=r["on"][:, C.F - 1])
            print(f"[s{seed} {panel} {split}] {len(days)} days ({time.time() - t0:.0f}s)", flush=True)


def report(seeds):
    fam, win = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for split in C.SPLITS:
            og, ref = W[split]["og"], W[split]["ref"]
            if not len(og["d"]):
                continue
            ad = np.mean([np.load(OUT / f"s{s}_adapt_{split}_{panel}.npz")["og_p"] for s in seeds], 0)
            bs = np.mean([np.load(OUT / f"s{s}_base_{split}_{panel}.npz")["og_p"] for s in seeds], 0)
            rest = 0.2 * ref["og_gs1"] + 0.2 * ref["og_cnn3"]
            V = {"REF T (tfm8 production)": ref["og_tfm8"], "tfm-rec, no adaptation": bs, "tfm-rec, adapted (9b)": ad,
                 "production tfm8 + adapted, 1:1": (ref["og_tfm8"] + ad) / 2}
            for n, p in V.items():
                for mode, pp in (("alone", p), ("T-blend", rest + 0.6 * p)):
                    sc = C.ongoing_iou(pp, og, 0.5)
                    fam.setdefault((split, mode, n), {})[panel] = sc.mean()
                    win.setdefault((split, mode, n), []).append(sc)
    for split in C.SPLITS:
        for mode in ("alone", "T-blend"):
            b = C.family_mean(fam[(split, mode, "REF T (tfm8 production)")])
            bw = np.concatenate(win[(split, mode, "REF T (tfm8 production)")])
            print(f"== ongoing {split} | {mode} | REF family {b:.4f}")
            for (s, md, n), d in fam.items():
                if s == split and md == mode and not n.startswith("REF"):
                    print(f"   {n:<34} fam {C.family_mean(d) - b:+.4f}  window {C.ci(np.concatenate(win[(s, md, n)]) - bw)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(3, 11)))
    a = ap.parse_args()
    if a.report:
        report(a.seeds)
    else:
        run_seed(a.seed)


if __name__ == "__main__":
    main()
