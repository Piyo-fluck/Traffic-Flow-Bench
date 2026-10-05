"""Task 1 production inference from saved CNN checkpoints, optionally with a month-specific base (note 17).

The base profile is the weekday x tod table over every train day, read from
work/cache/<panel>_t1tables.npz (scripts/t1_tables_cache.py). With --month-kappa / --month-kappa-q, the
split month's own deviation is added: the weekday-type x slot x station mean of (visible observation -
train table), shrunk by K pseudo-days. Task 1 is offline, so the whole month may be used. The checkpoints'
predictions are averaged and written as pred_<split>_<panel>.npz. Unless --npz-only is given, the state CSVs
are written as well (scripts/t1_npz_to_state.py can do that step in a torch-free process instead).

Windows note: a python.exe native crash (0xc000070a in ntdll) hit the long single-process run. Run one panel
per process (--panels) and keep the thread count low.

    py scripts/t1_predict_ckpt.py --ckpt work/sub_m3v1/t1_m3_seed0.pt:64 work/sub_m3v1_s1/t1_m3_seed1.pt:64 \
        work/sub_m3w128/t1_m3_seed0.pt:128 --month-kappa 2 --month-kappa-q 10 --out-dir work/sub_m3ens3_month --panels D7_I10_E --npz-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.submit import append_csv, check_against_template, state_frame  # noqa: E402

torch.set_num_threads(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="path:width")
    ap.add_argument("--month-kappa", type=float, default=None, help="shrinkage of the speed deviation (pseudo-days)")
    ap.add_argument("--month-kappa-q", type=float, default=None, help="shrinkage of the flow deviation; omit = no flow deviation")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    ap.add_argument("--npz-only", action="store_true", help="write pred_<split>_<panel>.npz only (assemble CSVs later)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets = []
    for spec in args.ckpt:
        path, width = spec.rsplit(":", 1)
        net = m3.Net(10, int(width)).to(m3.DEV)
        net.load_state_dict(torch.load(path, map_location=m3.DEV))
        net.eval()
        nets.append(net)
    print(f"{len(nets)} checkpoints; month kappa speed {args.month_kappa} flow {args.month_kappa_q}", flush=True)
    for split in args.splits:
        out = args.out_dir / f"state_{split}.csv"
        for i, panel in enumerate(args.panels):
            with np.load(CACHE / f"{panel}_t1tables.npz") as f:
                tv, tq = f["tv"], f["tq"]
            z = load(panel, split)
            bv, bq = tv[z["weekday"]], tq[z["weekday"]]
            if args.month_kappa is not None or args.month_kappa_q is not None:
                dv, dq = month_deviation(z, bv, bq, np.ones(len(z["dates"]), bool), args.month_kappa,
                                         kappa_q=args.month_kappa_q)
                print(f"[{split} {panel}] month deviation: mean |dv| {np.abs(dv).mean():.2f} km/h, "
                      f"mean |dq/lane| {np.abs(dq / z['lanes']).mean():.1f} veh/h", flush=True)
                bv, bq = bv + dv, bq + dq
            p = dict(X=m3.make_inputs(z, bv, bq), base_v=bv, base_q=bq, z=z)
            preds = [m3.predict_days(net, p, range(len(z["dates"]))) for net in nets]
            pv = np.mean([x[0] for x in preds], 0)
            pq = np.mean([x[1] for x in preds], 0)
            np.savez(args.out_dir / f"pred_{split}_{panel}.npz", pv=pv, pq=pq)
            print(f"[{split} {panel}] predictions saved", flush=True)
            if not args.npz_only:
                frame = state_frame(panel, z, pv, pq)
                check_against_template(panel, split, frame)
                append_csv(frame, out, first=(i == 0))
        if not args.npz_only:
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
