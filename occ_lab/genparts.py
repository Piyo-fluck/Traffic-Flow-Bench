"""Note 169: training parts from DLSim-generated runs (work/genruns/<panel>_r<k>.npz), in longctx.train_panels'
format, so leap_t2 can mix them with the real train panels.

Each run shares its panel's ctx and static channels (same network); its own arrays, labels (v <= 0.6 vf on
eligible cells), first-onset windows (earliest_onset) and up to `max_og` ongoing windows. No ES windows: early
stopping stays on the real panels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "analysis"))
from t2_eda02 import candidates  # noqa: E402
from t2_eda03 import earliest_onset, labels  # noqa: E402


def gen_parts(gen_dir, real_parts, max_og=1500, seed=0):
    rng = np.random.default_rng(seed)
    by_panel = {p["panel"]: p for p in real_parts}
    out = []
    for f in sorted(Path(gen_dir).glob("*_r*.npz")):
        panel = f.name.rsplit("_r", 1)[0]
        if panel not in by_panel:
            continue
        r = by_panel[panel]
        g = dict(np.load(f, allow_pickle=True))
        z = dict(true_speed=g["true_speed"].astype(np.float32), true_flow=g["true_flow"].astype(np.float32),
                 elig=g["elig"].astype(bool), free_speed=g["free_speed"], dates=g["dates"], weekday=g["weekday"])
        q, el = labels(z)
        dd, tt = np.nonzero(earliest_onset(q))
        c = candidates(z)
        d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        if len(d_og) > max_og:
            s = rng.choice(len(d_og), max_og, replace=False)
            d_og, t_og = d_og[s], t_og[s]
        d, t0 = np.concatenate([dd, d_og]), np.concatenate([tt, t_og])
        ok = el & np.isfinite(z["true_flow"])
        out.append(dict(panel=panel, ctx=r["ctx"], stat=r["stat"], wk=z["weekday"], q=q, el=el, d=d, t0=t0,
                        es=np.zeros(len(d), bool), v=z["true_speed"], qf=z["true_flow"], ok=ok, gen=True,
                        n_on=len(dd)))
        print(f"  [gen {f.stem}] {len(dd)} onset + {len(d_og)} ongoing windows", flush=True)
    return out
