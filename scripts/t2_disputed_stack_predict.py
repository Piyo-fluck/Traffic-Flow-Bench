"""Production side of note 81: re-decide the disputed ongoing cells of the OFFICIAL windows with a stacker.

For every official ongoing window (date D, panel p) the three members of the ongoing blend are computed
(gs1 from the LightGBM probability files, the CNN x3 and the transformer x8 from their checkpoints), the cells
where the three disagree are found, and those cells are decided by a LightGBM fitted on the disputed cells of the
official-like pseudo windows of the SAME split on dates before D (labels from released observations, which the
2026-09-20 clarification allows). The cells the members agree on keep the production decision.

Writes <out-dir>/pred_<split>_<panel>.npz with pred (N, F, S) bool for every window of window_index.csv (onset
rows are left False; scripts/t2_cnn_blend_predict.py --og-pred-dir uses the ongoing rows only).

    T2CNN_REC= .venv/Scripts/python.exe scripts/t2_disputed_stack_predict.py --out-dir work/sub_t2stack \
        --cnn <cnn x3> --tfm <tfm s3..s10> --gs-dir work/sub_t2gs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402  (also puts analysis/ and scripts/ on the path)
from t2_disputed_stack import BLEND_TAU, BLEND_W, NAMES, TAU, build  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE  # noqa: E402

C.cnn.REC = 0.0          # the production checkpoints have no reconstruction head


def load_nets(paths, arch):
    nets = []
    for p in paths:
        sd = torch.load(p, map_location=C.DEV)
        net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(sd["inp.weight"].shape[1] if arch != "tfm"
                                                              else sd["gru.weight_ih_l0"].shape[1], C.cnn.WIDTH)
        net.load_state_dict(sd)
        nets.append(net.to(C.DEV).eval())
    return nets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None,
                    help="a stacker trained beforehand (scripts/t2_stack_train_month.py); without it the stacker "
                         "is refitted per date on the scored month's earlier pseudo windows")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-cells", type=int, default=2000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nets_cnn, nets_tfm = load_nets(args.cnn, "cnn"), load_nets(args.tfm, "tfm")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=args.leaves, verbose=-1,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1)

    fixed_model = lgb.Booster(model_file=str(args.model)) if args.model else None
    pseudo = {s: [] for s in C.SPLITS}          # training pool: the official-like windows of every day
    official = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = W[s]["og"], W[s]["ref"]
            if len(og["d"]) and fixed_model is None:
                P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"]}
                X, pred, dis = build(i, og, ctx, P)
                dates = np.array([str(x) for x in W[s]["z"]["dates"]])[og["d"]]
                pseudo[s].append(dict(X=X, y=og["fut"], dis=dis & og["efut"], dates=dates))
            # official windows of this panel and split
            z = W[s]["z"]
            base = RELEASE / "task2" / panel / s
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0w, grp = hist_from_release(ctx, Hh, Wi)
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0w, grp.astype(bool))
            p_cnn = np.mean([C.cnn.predict(n, x) for n in nets_cnn], 0)
            p_tfm = np.mean([C.cnn.predict(n, x) for n in nets_tfm], 0)
            gs = np.load(args.gs_dir / f"prob_{s}_{panel}.npz")
            assert (gs["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "gs window order differs"
            P = {"gs1": gs["prob"].astype(np.float32), "tfm8": p_tfm, "cnn3": p_cnn}
            X, pred, dis = build(i, dict(v=v, qf=qf, ok=ok), ctx, P)
            official.append(dict(panel=panel, split=s, X=X, pred=pred, dis=dis,
                                 is_on=(Wi.condition == "queue_onset").to_numpy(),
                                 dates=Wi.date.astype(str).to_numpy(), ids=Wi.window_id.to_numpy().astype(str)))
        print(f"[{panel}] official windows prepared ({time.time() - t0:.0f}s)", flush=True)

    models = {}
    for o in official:
        s = o["split"]
        out = o["pred"].copy()
        for date in np.unique(o["dates"][~o["is_on"]]):
            key = (s, date)
            if fixed_model is not None:
                models[key] = fixed_model
            elif key not in models:
                Xtr = np.concatenate([p["X"][p["dates"] < date][p["dis"][p["dates"] < date]] for p in pseudo[s]])
                ytr = np.concatenate([p["y"][p["dates"] < date][p["dis"][p["dates"] < date]] for p in pseudo[s]])
                if len(ytr) < args.min_cells or ytr.min() == ytr.max():
                    models[key] = None
                else:
                    models[key] = lgb.train(params, lgb.Dataset(Xtr, ytr.astype(np.float32), feature_name=NAMES),
                                            args.rounds)
                    print(f"  stacker {s} {date}: {len(ytr):,} past disputed cells ({time.time() - t0:.0f}s)",
                          flush=True)
            mdl = models[key]
            m = (o["dates"] == date) & ~o["is_on"]
            sel = o["dis"][m]
            if mdl is None or not sel.any():
                continue
            blk = out[m]
            blk[sel] = mdl.predict(o["X"][m][sel]) >= 0.5
            out[m] = blk
        n_dis = int(o["dis"][~o["is_on"]].sum())
        n_ch = int((out[~o["is_on"]] != o["pred"][~o["is_on"]]).sum())
        np.savez_compressed(args.out_dir / f"pred_{o['split']}_{o['panel']}.npz", pred=out, window_id=o["ids"])
        print(f"[{o['panel']} {o['split']}] disputed ongoing cells {n_dis}, changed {n_ch} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
