"""Note 165 for the official windows: the onset STID's FiLM supervised by recovered hidden onsets.

An official onset window on day d hides its horizon and the next hour; the released layer shows the queue again
65 min after T, and that is before the origin of every window on a LATER day. Each earlier official onset window
gets a pseudo T+30 label = the head of that post-blank queue (occ_lab/hidden_onset_recover.py, ranker fitted on the
training months, H's official T+30 probability as the prior). The FiLM (history branch of the 4 frozen onset STIDs)
is adapted day by day on the windows of the 4 most recent earlier days, 20 steps, lr 3e-3, carried forward - the
same schedule as occ_lab/film_scored_eval.py. Only released data before each window's origin is used.

Writes <out-dir>/prob_<split>_<panel>.npz: window_id, prob (FiLM STID, (W, S) at T+30), plain (the frozen STIDs).

    .venv/Scripts/python.exe scripts/t2_onset_film_official.py --out-dir work/sub_t2film \
        --h-dir work/sub_t2robust_bag5 --profiles work/models_remote/occ_profiles.npz
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "occ_lab"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
import hidden_onset_recover as HR  # noqa: E402
import onset_film_recov as OF  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from t2_occ_blend_predict import occ_from_release  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE  # noqa: E402
from train_occ import occ_channels  # noqa: E402

C.cnn.REC = 0.0
STEPS, LR, BS, CAP = 20, 3e-3, 16, 4


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--h-dir", type=Path, required=True, help="LightGBM onset probabilities (the recovery prior)")
    ap.add_argument("--profiles", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranker = HR.fit_ranker("9999")
    profs = np.load(args.profiles)
    md = ROOT / "occ_lab" / "models"
    for split in C.SPLITS:
        loff = 0
        for i, panel in enumerate(C.T2_PANELS):
            ctx = C.Ctx(panel, i)
            ctx.loff = 0
            stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
            S = ctx.S
            lid = np.arange(S) + loff
            loff += S
            base = RELEASE / "task2" / panel / split
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            ids = Wi.window_id.to_numpy().astype(str)
            fh = np.load(args.h_dir / f"prob_{split}_{panel}.npz")
            assert (fh["window_id"] == ids).all(), "window order differs"
            pH = fh["prob"][:, C.F - 1]
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
            xo = np.concatenate([x, occ_channels(occ_from_release(ctx, Hh, Wi, t0), ok, profs[panel])], 1)
            dow = np.array([pd.Timestamp(str(d)).weekday() for d in Wi.date], np.int64)
            # loss mask: the window's scored T+30 cells (the sample submission's rows)
            w_of = {w: k for k, w in enumerate(Wi.window_id)}
            s_of = {l: k for k, l in enumerate(ctx.z["link_ids"])}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = np.asarray(slot(Q.timestamp)) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy().astype(int)
            msk = np.zeros((len(Wi), S), np.float32)
            sel = hi == C.F - 1
            msk[wi[sel], si[sel]] = 1.0
            # recovered labels of the official onset windows (usable only by windows of later days)
            z = C.load(panel, split)
            vf = np.asarray(ctx.vf, np.float32)
            vz, okz = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"])
            qd = okz & (np.nan_to_num(vz, nan=1e9) <= 0.6 * vf)
            vr = np.where(okz, vz / vf, np.nan)
            d_of = {str(x_): k for k, x_ in enumerate(z["dates"])}
            dd = np.array([d_of[str(d)] for d in Wi.date])
            is_on = (Wi.condition == "queue_onset").to_numpy()
            lab = np.zeros((len(Wi), S), np.float32)
            has = np.zeros(len(Wi), bool)
            for j in np.flatnonzero(is_on):
                r = HR.recover(ranker, pH[j], qd, vr, int(dd[j]), int(t0[j]), panel not in HR.UP_HIGHER, width=0)
                if r is not None:
                    lab[j] = r; has[j] = True
            nets, opts = [], []
            for k in range(4):
                d_ = torch.load(md / f"onset_stid_s{k}.pt", map_location=C.DEV)
                n_ = OF.OnsetSTID(d_["cin"], d_["n_links"], d_["n_slots"], ident=d_.get("ident", "id")).to(C.DEV)
                n_.load_state_dict(d_["state"])
                f_ = OF.Film(n_.eval(), "series").to(C.DEV)
                nets.append(f_); opts.append(torch.optim.Adam(f_.film(), lr=LR))
            T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(C.DEV, dt)  # noqa: E731
            P0 = np.zeros((len(Wi), S), np.float32); P1 = np.zeros_like(P0)
            rng = np.random.default_rng(11 + i)
            days = np.unique(dd)
            n_adapt = 0
            for k, day in enumerate(days):
                pdays = days[max(0, k - CAP):k]
                past = np.flatnonzero(np.isin(dd, pdays) & is_on & has)
                if len(past) >= 2:
                    n_adapt += 1
                    for f_, opt in zip(nets, opts):
                        for _ in range(STEPS):
                            m = rng.choice(past, min(BS, len(past)), replace=len(past) < BS)
                            lg_ = f_(T(xo[m]), T(np.broadcast_to(lid, (len(m), S)), torch.long),
                                     T(t0[m].astype(np.int64), torch.long), T(dow[m], torch.long))[:, 0]
                            mm = T(msk[m])
                            loss = (nn.functional.binary_cross_entropy_with_logits(lg_, T(lab[m]), reduction="none")
                                    * mm).sum() / mm.sum().clamp(min=1)
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                cur = np.flatnonzero(dd == day)
                a_, b_, c_ = (T(np.broadcast_to(lid, (len(cur), S)), torch.long),
                              T(t0[cur].astype(np.int64), torch.long), T(dow[cur], torch.long))
                with torch.no_grad():
                    xc = T(xo[cur])
                    P0[cur] = np.mean([torch.sigmoid(f_.net(xc, a_, b_, c_))[:, 0].cpu().numpy() for f_ in nets], 0)
                    P1[cur] = np.mean([torch.sigmoid(f_(xc, a_, b_, c_))[:, 0].cpu().numpy() for f_ in nets], 0)
            np.savez(args.out_dir / f"prob_{split}_{panel}.npz", window_id=ids, prob=P1, plain=P0)
            print(f"[{split} {panel}] onset windows {int(is_on.sum())}, recovered {int(has.sum())}, days {len(days)}, "
                  f"adapted days {n_adapt}, mean |FiLM - plain| on onset {np.abs(P1 - P0)[is_on].mean():.4f}",
                  flush=True)
            ctx.z = ctx.q = ctx.el = None


if __name__ == "__main__":
    main()
