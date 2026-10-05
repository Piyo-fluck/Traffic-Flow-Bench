"""Production Task 2: blend the LightGBM probabilities with T2 CNN probabilities on the official windows.

Evaluation (notes 36, 38) on the blank-aware official-like windows:
- onset:   p = w_on * p_lgbm(H) + (1 - w_on) * p_cnn at T+30, then the expected-IoU ratio rule (c 1.0)
- ongoing: p = w_og * p_lgbm(I) + (1 - w_og) * p_cnn over the horizon, then p >= tau
p_cnn is the mean over the given CNN checkpoints (analysis/t2_cnn.py: T2Net, width 64, no prior channels).

Inputs:
- the official window histories (task2/<panel>/<split>/window_history.parquet);
- the LightGBM production probabilities `prob_<split>_<panel>.npz` from `scripts/t2_m1_predict.py` (onset from
  --onset-probs-dir, ongoing from --ongoing-probs-dir; the window order is that of window_index.csv);
- the CNN checkpoints.

Writes queue_<split>.csv for scripts/swap_task_rows.py.

    py scripts/t2_cnn_blend_predict.py --cnn work/models_remote/t2_cnn_lov.pt work/models_remote/t2_cnn_lov_s1.pt \
        --onset-probs-dir work/sub_t2robust_bag5 --ongoing-probs-dir work/sub_t2robust_bag5_stress \
        --w-on 0.85 --w-og 0.6 --tau 0.6 --out-dir work/sub_t2M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import hist_from_release, table_prior  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", nargs="+", type=Path, required=True, help="T2 CNN checkpoints (width 64, no priors)")
    ap.add_argument("--cnn-arch", choices=["cnn", "tfm", "tfm_pri"], default="cnn",
                    help="architecture of the --cnn checkpoints (tfm_pri: transformer with position + train-prior channels)")
    ap.add_argument("--og-nets", nargs="+", type=Path, default=None,
                    help="separate checkpoints for the ongoing blend (default: the --cnn ones); note 45")
    ap.add_argument("--og-arch", choices=["cnn", "tfm", "tfm_pri"], default="cnn", help="architecture of the --og-nets checkpoints")
    ap.add_argument("--og-nets2", nargs="+", type=Path, default=None,
                    help="second ongoing neural group, for the note-56 ensemble (transformer seeds beside the CNNs)")
    ap.add_argument("--og-arch2", choices=["cnn", "tfm", "tfm_pri"], default="tfm", help="architecture of --og-nets2")
    ap.add_argument("--w-og2", type=float, default=None,
                    help="weight of --og-nets2; ongoing = w_og*lgbm + w_og2*og_nets2 + (1 - w_og - w_og2)*og_nets")
    ap.add_argument("--onset-probs-dir", type=Path, required=True)
    ap.add_argument("--ongoing-probs-dir", type=Path, required=True)
    ap.add_argument("--w-on", type=float, default=0.85)
    ap.add_argument("--w-og", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--analog-dir", type=Path, default=None,
                    help="state-conditioned analog onset maps from scripts/t2_analog_probs.py (note 47)")
    ap.add_argument("--w-analog", type=float, default=0.0,
                    help="onset: p = (1 - w_analog) * (w_on * LGBM + (1 - w_on) * CNN) + w_analog * map")
    ap.add_argument("--onset-cnn-probs-dir", type=Path, default=None,
                    help="precomputed onset CNN probabilities (note 67: the past-day adapted model, "
                         "whose weights differ per day and cannot be recomputed from one checkpoint)")
    ap.add_argument("--onset-delta-base-dir", type=Path, default=None,
                    help="note 73 delta correction: probabilities of the UNADAPTED checkpoint (t2_adapt_official.py "
                         "--steps 0); p = sigmoid(logit p + lambda * (D - median D)), D = logit p_adapt - logit p_base")
    ap.add_argument("--onset-delta-lambda", type=float, default=0.0)
    ap.add_argument("--ratio-c", type=float, default=1.0, help="c of the expected-IoU ratio rule for onset")
    ap.add_argument("--og-pred-dir", type=Path, default=None,
                    help="note 81: ready-made ongoing decisions (pred_<split>_<panel>.npz from "
                         "scripts/t2_disputed_stack_predict.py); the ongoing threshold is then not used")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.onset_delta_lambda and (args.onset_delta_base_dir is None or args.onset_cnn_probs_dir is None):
        raise SystemExit("--onset-delta-lambda needs --onset-delta-base-dir and --onset-cnn-probs-dir")
    if cnn.PRIORS:
        raise SystemExit("T2CNN_PRIORS must be off: these checkpoints take the 12 base channels")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    def load_nets(paths, arch):
        out = []
        for path in paths:
            cin = 12 + 1 + cnn.N_PRI if arch == "tfm_pri" else 12   # position + train-prior channels
            net = (cnn.T2Tfm if arch in ("tfm", "tfm_pri") else cnn.T2Net)(cin, cnn.WIDTH).to(cnn.DEV)
            net.load_state_dict(torch.load(path, map_location=cnn.DEV))
            net.eval()
            out.append(net)
        return out

    nets = load_nets(args.cnn, args.cnn_arch)
    nets_og = load_nets(args.og_nets, args.og_arch) if args.og_nets else None
    if (args.og_nets2 is None) != (args.w_og2 is None):
        raise SystemExit("--og-nets2 and --w-og2 go together")
    nets_og2 = load_nets(args.og_nets2, args.og_arch2) if args.og_nets2 else None
    print(f"{len(nets)} {args.cnn_arch} checkpoints (onset{'' if nets_og else ' + ongoing'})"
          + (f", {len(nets_og)} {args.og_arch} checkpoints (ongoing)" if nets_og else "")
          + f"; w_on {args.w_on}, w_og {args.w_og}, tau {args.tau}", flush=True)
    if nets_og2:
        print(f"second ongoing group: {len(nets_og2)} {args.og_arch2} checkpoints at w_og2 {args.w_og2}; "
              f"the first group takes the remainder {1 - args.w_og - args.w_og2:.2f}", flush=True)
    for split in ("validation", "private"):
        frames = []
        for i, panel in enumerate(T2_PANELS):
            ctx = Ctx(panel, i)
            stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            x = cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
            x_pri = None
            if "tfm_pri" in (args.cnn_arch, args.og_arch, args.og_arch2):  # position + train-table priors, as in the evaluation
                pri = table_prior(ctx)
                key = np.stack([np.zeros(len(W), int), grp.astype(int)], 1)  # table_prior reads the weekday type only
                x_pri = cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool),
                                   cnn.prior_extra(ctx.S, len(W), t0, lambda name, t: pri(name, key, t)))
            xin = lambda arch: x_pri if arch == "tfm_pri" else x  # noqa: E731
            p_cnn = np.mean([cnn.predict(net, xin(args.cnn_arch)) for net in nets], 0)       # (N, F, S)
            p_og = np.mean([cnn.predict(net, xin(args.og_arch)) for net in nets_og], 0) if nets_og else p_cnn
            p_og2 = np.mean([cnn.predict(net, xin(args.og_arch2)) for net in nets_og2], 0) if nets_og2 else None
            if args.onset_cnn_probs_dir is not None:  # the onset half comes from the adapted model instead
                fad = np.load(args.onset_cnn_probs_dir / f"prob_{split}_{panel}.npz")
                assert (fad["window_id"] == W.window_id.to_numpy().astype(str)).all(), "adapted window order differs"
                p_cnn = fad["prob"].astype(p_cnn.dtype)
            dcorr = None
            if args.onset_delta_lambda:
                fb = np.load(args.onset_delta_base_dir / f"prob_{split}_{panel}.npz")
                assert (fb["window_id"] == W.window_id.to_numpy().astype(str)).all(), "base window order differs"
                lg = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))  # noqa: E731
                dD = lg(p_cnn[:, F - 1].astype(np.float64)) - lg(fb["prob"][:, F - 1].astype(np.float64))
                dcorr = args.onset_delta_lambda * (dD - np.median(dD, 1, keepdims=True))
            fon = np.load(args.onset_probs_dir / f"prob_{split}_{panel}.npz")
            fog = np.load(args.ongoing_probs_dir / f"prob_{split}_{panel}.npz")
            ids = W.window_id.to_numpy().astype(str)
            assert (fon["window_id"] == ids).all() and (fog["window_id"] == ids).all(), "window order differs"
            is_on = (W.condition == "queue_onset").to_numpy()
            fan = None
            if args.analog_dir is not None and args.w_analog > 0:
                fan = np.load(args.analog_dir / f"prob_{split}_{panel}.npz")
                assert (fan["window_id"] == ids).all(), "analog window order differs"
            pred = np.zeros((len(W), F, ctx.S), bool)
            for j in np.flatnonzero(is_on):
                p = args.w_on * fon["prob"][j, F - 1] + (1 - args.w_on) * p_cnn[j, F - 1]
                if fan is not None:  # state-conditioned analog prior, blended after the models (note 47)
                    p = (1 - args.w_analog) * p + args.w_analog * fan["prob"][j, F - 1]
                if dcorr is not None:  # note 73: push along the direction the past-day adaptation moved
                    pc = np.clip(p.astype(np.float64), 1e-5, 1 - 1e-5)
                    p = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + dcorr[j])))
                pred[j, F - 1] = select(p, "ratio", args.ratio_c)
            og = ~is_on
            if args.og_pred_dir is not None:
                fp = np.load(args.og_pred_dir / f"prob_{split}_{panel}.npz".replace("prob_", "pred_"))
                assert (fp["window_id"] == ids).all(), "stacked ongoing window order differs"
                pred[og] = fp["pred"][og]
                print(f"[{split} {panel}] ongoing decisions taken from {args.og_pred_dir}", flush=True)
            elif p_og2 is None:
                p_mix = args.w_og * fog["prob"][og] + (1 - args.w_og) * p_og[og]
            else:   # note 56: w_og on the LightGBM (gs1), w_og2 on the second neural group, the rest on the first
                p_mix = (args.w_og * fog["prob"][og] + args.w_og2 * p_og2[og]
                         + (1 - args.w_og - args.w_og2) * p_og[og])
            if args.og_pred_dir is None:
                pred[og] = p_mix >= args.tau
            s_of = {l: k for k, l in enumerate(ctx.z["link_ids"])}
            w_of = {w: k for k, w in enumerate(W.window_id)}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = np.asarray(slot(Q.timestamp)) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy()
            if (hi < 0).any() or (hi >= F).any() or np.isnan(si.astype(float)).any():
                raise ValueError(f"{panel}/{split}: template rows outside the windows")
            Q["queue_pred"] = pred[wi, hi, si.astype(int)].astype(int)
            frames.append(Q)
            print(f"[{split} {panel}] onset cells {int(pred[is_on].sum())}, ongoing cells {int(pred[og].sum())}", flush=True)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"queue_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'queue_{split}.csv'}: {len(out):,} rows, share 1 = {out.queue_pred.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
