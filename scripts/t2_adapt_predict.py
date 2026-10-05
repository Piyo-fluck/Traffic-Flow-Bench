"""Task 2 production with test-time adaptation on the scored month (notes 18-19), past days only.

For each split (validation, private), the month is cut into blocks starting at BLOCK_STARTS days. The
official windows of a block are predicted by a model trained on:
  - every train window (priors leave-one-day-out over train), plus
  - the split's pseudo windows on days strictly before the block start (weight W_ADAPT).
Their prior features come from the month-past prior (kappa 4, days < window day). Block 0 has no past days,
so it uses the saved production models (work/sub_t2m1) with the month-past prior. Features of the official
windows come only from window_history.parquet. Decisions: onset top round(1.2 * sum p) at T+30,
ongoing p >= 0.65.

    py scripts/t2_adapt_predict.py --out-dir work/sub_t2adapt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, onset_decide  # noqa: E402
from t2_m1_predict import hist_from_release, month_prior, ongoing_X, onset_X, train_prior  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from t2_pseudo_eval import pseudo_windows  # noqa: E402
from t2_shift_remedies import fit, pseudo_rows, train_rows  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402

BLOCK_STARTS = (0, 7, 14, 21, 28)
W_ADAPT, ROUNDS_ON, ROUNDS_OG, ALPHA, TAU = 5.0, 600, 2000, 1.2, 0.65


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    m_on_E = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_onset.txt"))
    m_og_E = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_ongoing.txt"))

    ctxs, tr = [], []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        names_on, on_p, names_og, og_p = train_rows(ctx, train_prior(ctx))
        tr.append((on_p, og_p))
        # Later steps need only link ids, static link attributes and the smoothed prior arrays
        # (month_prior -> train_tables). Release the large train arrays so 8 panels fit in RAM.
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[train rows {p}] ({time.time() - t_start:.0f}s)", flush=True)

    for split in ("validation", "private"):
        per_panel = {}
        for ctx in ctxs:
            z = load(ctx.panel, split)
            w = pseudo_windows(ctx, z)
            grp_w = (z["weekday"][w["d"]] >= 5).astype(int)
            key_w = np.stack([w["d"], grp_w], 1)
            pri_m = month_prior(ctx, z, 4.0)
            base_dir = RELEASE / "task2" / ctx.panel / split
            W = pd.read_csv(base_dir / "window_index.csv")
            Hh = pd.read_parquet(base_dir / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            d_of = {d: i for i, d in enumerate(z["dates"])}
            dW = W.date.astype(str).map(d_of).to_numpy()
            per_panel[ctx.panel] = dict(z=z, w=w, grp_w=grp_w, key_w=key_w, pri_m=pri_m, W=W, v=v, qf=qf, ok=ok,
                                        t0=t0, grp=grp, dW=dW, key=np.stack([dW, grp], 1).astype(int))
        prob = {p: np.zeros((len(per_panel[p]["W"]), F, c.S), np.float32) for p, c in zip(per_panel, ctxs)}
        for bi, start in enumerate(BLOCK_STARTS):
            end = BLOCK_STARTS[bi + 1] if bi + 1 < len(BLOCK_STARTS) else 10 ** 6
            if start == 0:
                m_on, m_og = m_on_E, m_og_E
            else:
                ad = []
                for ctx in ctxs:
                    pp = per_panel[ctx.panel]
                    ad.append(pseudo_rows(ctx, pp["w"], pp["w"]["d"] < start, pp["grp_w"], pp["pri_m"], pp["key_w"]))
                parts_on = [(x[0][0], x[0][1], 1.0) for x in tr] + [(a[0][0], a[0][1], W_ADAPT) for a in ad if len(a[0][1])]
                parts_og = [(x[1][0], x[1][1], 1.0) for x in tr] + [(a[1][0], a[1][1], W_ADAPT) for a in ad if a[1][0] is not None]
                m_on, m_og = fit(names_on, parts_on, ROUNDS_ON, 63), fit(names_og, parts_og, ROUNDS_OG, 127)
                m_on.save_model(str(args.out_dir / f"t2_onset_{split}_b{start}.txt"))
                m_og.save_model(str(args.out_dir / f"t2_ongoing_{split}_b{start}.txt"))
            for ctx in ctxs:
                pp = per_panel[ctx.panel]
                blk = (pp["dW"] >= start) & (pp["dW"] < end)
                is_on = (pp["W"].condition == "queue_onset").to_numpy()
                for mask, is_onset in ((blk & is_on, True), (blk & ~is_on, False)):
                    if not mask.any():
                        continue
                    args_x = (ctx, pp["v"][mask], pp["qf"][mask], pp["ok"][mask], pp["t0"][mask], pp["grp"][mask],
                              pp["pri_m"], pp["key"][mask])
                    if is_onset:
                        _, X = onset_X(*args_x)
                        prob[ctx.panel][mask, F - 1] = m_on.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
                    else:
                        _, Xh, _ = ongoing_X(*args_x)
                        prob[ctx.panel][mask] = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
            print(f"[{split}] block {start}: done ({time.time() - t_start:.0f}s)", flush=True)

        frames = []
        for ctx in ctxs:
            pp = per_panel[ctx.panel]
            W, t0, pr = pp["W"], pp["t0"], prob[ctx.panel]
            is_on = (W.condition == "queue_onset").to_numpy()
            pred = np.zeros_like(pr, bool)
            if is_on.any():
                pred[is_on, F - 1] = onset_decide(pr[is_on, F - 1], ("sum", ALPHA))
            pred[~is_on] = pr[~is_on] >= TAU
            np.savez(args.out_dir / f"prob_{split}_{ctx.panel}.npz", prob=pr, is_onset=is_on,
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            Q = pd.read_csv(RELEASE / "task2" / ctx.panel / split / "sample_submission_queue.csv",
                            usecols=["window_id", "timestamp", "link_id"])
            s_of = {l: i for i, l in enumerate(ctx.z["link_ids"])}
            w_of = {wid: i for i, wid in enumerate(W.window_id)}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = slot(Q.timestamp) - t0[wi] - 1
            if (hi < 0).any() or (hi >= F).any():
                raise ValueError(f"{ctx.panel}/{split}: template timestamps outside the horizon")
            Q["queue_pred"] = pred[wi, hi, Q.link_id.astype(str).map(s_of).to_numpy()].astype(int)
            frames.append(Q)
            print(f"[{split} {ctx.panel}] onset cells {int(pred[is_on].sum())}, ongoing cells {int(pred[~is_on].sum())}", flush=True)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"queue_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'queue_{split}.csv'}: {len(out):,} rows, share 1 = {out.queue_pred.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
