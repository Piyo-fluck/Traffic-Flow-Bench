"""Note 169: the ctrl CNN's T+30 onset probabilities on the generated runs' onset windows (no adaptation), so a
candidate reranker can be trained where run-level demand shifts exist. Per generated run file: d, t0 of the
earliest-onset windows, prob (N, S) of the leap ctrl_s20 / ctrl_s21 average, truth y (N, S) and eligibility e.
Saves occ_lab/cache/genonset_<run file stem>.npz.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python occ_lab/gen_onset_infer.py work/genruns_ctm_hb
"""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import genparts as GP  # noqa: E402
import leap_t2 as LT  # noqa: E402
import longctx as LC  # noqa: E402

F = LT.F


def main():
    gen_dir = Path(sys.argv[1])
    parts = LC.train_panels()
    nets = [LT.load_leap(t) for t in ("ctrl_s20", "ctrl_s21")]
    for gp, f in zip(GP.gen_parts(gen_dir, parts, max_og=0), sorted(gen_dir.glob("*_r*.npz"))):
        k = gp["n_on"]
        if not k:
            continue
        d, t0 = gp["d"][:k], gp["t0"][:k]
        x = LC.short_x(gp, d, t0)
        ps = []
        for net in nets:
            net.set_panel(gp["panel"])
            ps.append(LT.predict_ctx(net, x, None)[:, F - 1])
        fi = t0 + F
        np.savez(LAB / "cache" / (__import__("os").environ.get("OUT_SUB", "")) / f"genonset_{f.stem}.npz", d=d, t0=t0, prob=np.mean(ps, 0).astype(np.float32),
                 y=gp["q"][d, fi], e=gp["el"][d, fi])
        print(f"  {f.stem}: {k} onset windows", flush=True)


if __name__ == "__main__":
    main()
