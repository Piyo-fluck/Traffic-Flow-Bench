"""Ongoing decisions for the official windows with the residual corrector (note 86).

    p0        = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3                     the frozen production blend
    r_hat     = corrector(member probabilities, disagreement, horizon, position)
    p_final   = clip(p0 + eta * r_hat where |r_hat| >= gate, 0, 1),  then p_final >= tau

The corrector is `occ_lab/models/residual_lgb.txt`, fitted on the 2031-01/02 windows the members were never
fitted on (`occ_lab/cache_oof.py`, `fit_corrector.py`) - earlier than every scored origin, so it is admissible
for both months. Feature order must stay exactly `residual_diag.NAMES`.

Writes ongoing_<split>.csv (window_id, timestamp, link_id, queue_pred) for the ongoing windows only, so the onset
rows of the candidate it patches stay untouched.

    .venv/Scripts/python.exe scripts/t2_residual_predict.py --out-dir work/sub_t2res --eta 0.75 --gate 0.10 \
        --cnn <cnn x3> --tfm <tfm s3..s10> --occ-cnn <occ_cnn s0..s3> --occ-tfm <occ_tfm s0..s3> \
        --stid <stid s0..s3> --profiles work/models_remote/occ_profiles.npz --gs-dir work/sub_t2gs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "occ_lab"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from online_adapter import COLS, adapter_cols  # noqa: E402
from residual_diag import NAMES, lg  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from t2_occ_blend_predict import occ_from_release  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID, predict as stid_predict  # noqa: E402

C.cnn.REC = 0.0


def load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def features(P, base, cov=None):
    """AI's 15 features; with cov (per-window observed share of the history) the 16th of note 87."""
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None], base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    members = np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
    return np.stack([base, lg(base), np.abs(base - 0.5), members.std(0), members.max(0) - members.min(0),
                     P["gs1"], P["tfm8"], P["cnn3"], P["occ_cnn"], P["occ_tfm"], P["stid"],
                     lg(P["stid"]) - lg(base), lg(P["occ_cnn"]) - lg(base), hor, pos]
                    + ([np.broadcast_to(cov[:, None, None].astype(np.float32), base.shape)] if cov is not None
                       else []), -1)


def og_mask(Wi):
    return (Wi.condition != "queue_onset").to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--occ-cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--occ-tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--stid", type=Path, nargs="+", required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--gs-dir", type=Path, required=True)
    ap.add_argument("--corrector", type=Path, default=ROOT / "occ_lab" / "models" / "residual_lgb.txt",
                    help="one corrector for both months, or a path with {split} in it for one per month")
    ap.add_argument("--eta", type=float, default=0.75)
    ap.add_argument("--gate", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument("--stab-models", type=Path, nargs="*", default=[],
                    help="note 90: block correctors; the correction is kept only where >= --stab-min of them "
                         "give r_hat the same sign as --corrector")
    ap.add_argument("--stab-min", type=int, default=0)
    ap.add_argument("--pseudo-dir", type=Path, default=None,
                    help="note 106: the pseudo-supervised adapted CNN's ongoing probabilities")
    ap.add_argument("--w-pseudo", type=float, default=0.0)
    ap.add_argument("--tfm-dir", type=Path, default=None,
                    help="the transformer family ADAPTED to the scored run (t2_pseudo_adapt_official.py --arch tfm, "
                         "seed mean): replaces the frozen tfm8 in the blend")
    ap.add_argument("--stid-dir", type=Path, default=None,
                    help="the STID family ADAPTED to the scored run (occ_lab/pseudo_adapt_stid.py --mode official, "
                         "seed mean): replaces the frozen STID in the corrector's feature columns")
    ap.add_argument("--occ-cnn-dir", type=Path, default=None,
                    help="occ_cnn adapted to the scored run (occ_lab/adapt_family.py --mode official, seed mean)")
    ap.add_argument("--occ-tfm-dir", type=Path, default=None,
                    help="occ_tfm adapted to the scored run (occ_lab/adapt_family.py --mode official, seed mean)")
    ap.add_argument("--tfm-feat", choices=("frozen", "adapted"), default="frozen",
                    help="which tfm8 the corrector's feature columns see when --tfm-dir is given")
    ap.add_argument("--adapt-dir", type=Path, default=None,
                    help="item 1: per-day coefficients from scripts/t2_online_adapt_beta.py, a run-specific "
                         "linear adapter on top of the global corrector's output")
    ap.add_argument("--select", action="store_true",
                    help="note 141: per-window, per-horizon member selection on the run's official-like ongoing pool "
                         "(occ_lab/select_bank.py) replaces p0 on the ongoing windows")
    ap.add_argument("--select-ps-tag", default="solid3",
                    help="the p_PS the bank's lab predictions use (fam_av_s<k>_<tag> caches)")
    ap.add_argument("--update", choices=("prob", "newton"), default="prob",
                    help="prob: clip(p0 + eta r_hat) (AI); newton: sigmoid(logit p0 + eta z_hat) (note 88)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets = {k: [load(p, "tfm" if "tfm" in k else "cnn") for p in v] for k, v in
            (("cnn3", args.cnn), ("tfm8", args.tfm), ("occ_cnn", args.occ_cnn), ("occ_tfm", args.occ_tfm))}
    stids = []
    for p in args.stid:
        d = torch.load(p, map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        stids.append(net.eval())
    per_split = "{split}" in str(args.corrector)
    correctors = {s: lgb.Booster(model_file=str(args.corrector).format(split=s)) for s in C.SPLITS}         if per_split else None
    gbm = correctors[C.SPLITS[0]] if per_split else lgb.Booster(model_file=str(args.corrector))
    stab = [lgb.Booster(model_file=str(m)) for m in args.stab_models]
    assert all(m.feature_name() == gbm.feature_name() for m in stab), "block correctors use other features"
    use_cov = gbm.feature_name() == NAMES + ["coverage"]
    assert use_cov or gbm.feature_name() == NAMES, "the corrector was fitted on a different feature order"
    profs = np.load(args.profiles)
    print(f"corrector {args.corrector.name} ({len(gbm.feature_name())} features{', coverage' if use_cov else ''}), "
          f"update {args.update}, eta {args.eta}, gate {args.gate}, tau {args.tau}", flush=True)
    for split in C.SPLITS:
        if per_split:
            gbm = correctors[split]
            print(f"[{split}] corrector {Path(str(args.corrector).format(split=split)).name}", flush=True)
        frames, changed = [], 0
        loff = 0
        for i, panel in enumerate(C.T2_PANELS):
            ctx = C.Ctx(panel, i)
            ctx.loff = 0
            stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
            lid = np.arange(ctx.S) + loff
            loff += ctx.S
            base_dir = RELEASE / "task2" / panel / split
            Wi = pd.read_csv(base_dir / "window_index.csv")
            Hh = pd.read_parquet(base_dir / "window_history.parquet")
            Q = pd.read_csv(base_dir / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
            xo = np.concatenate([x, occ_channels(occ_from_release(ctx, Hh, Wi, t0), ok, profs[panel])], 1)
            dow = np.array([pd.Timestamp(str(d)).weekday() for d in Wi.date], np.int64)
            P = {k: np.mean([C.cnn.predict(n, xo if k.startswith("occ") else x) for n in ns], 0)
                 for k, ns in nets.items()}
            P["stid"] = np.mean([stid_predict(n, xo, lid, t0.astype(np.int64), dow) for n in stids], 0)
            for key, dd in (("occ_cnn", args.occ_cnn_dir), ("occ_tfm", args.occ_tfm_dir)):
                if dd:
                    fo = np.load(dd / f"prob_{split}_{panel}.npz")
                    assert (fo["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), f"{key} order differs"
                    P[key] = fo["prob"].astype(np.float32)
            if args.stid_dir:
                fs = np.load(args.stid_dir / f"prob_{split}_{panel}.npz")
                assert (fs["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "stid order differs"
                P["stid"] = fs["prob"].astype(np.float32)
            gs = np.load(args.gs_dir / f"prob_{split}_{panel}.npz")
            assert (gs["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "gs1 window order differs"
            P["gs1"] = gs["prob"].astype(np.float32)
            if args.tfm_dir:
                ft = np.load(args.tfm_dir / f"prob_{split}_{panel}.npz")
                assert (ft["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "tfm order differs"
                tad = ft["prob"].astype(np.float32)
                old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
                if args.tfm_feat == "adapted":
                    P["tfm8"] = tad
            else:
                old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
            p0, ps = old, None
            if args.w_pseudo:
                fp = np.load(args.pseudo_dir / f"prob_{split}_{panel}.npz")
                assert (fp["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "pseudo order differs"
                ps = fp["prob"].astype(np.float32)
                p0 = (1 - args.w_pseudo) * old + args.w_pseudo * ps
            if args.select:
                import select_bank as SB
                assert args.tfm_dir is not None and ps is not None, "--select needs the adapted tfm8 and p_PS"
                bank = SB.bank_for(i, panel, split, args.select_ps_tag)
                zs = C.load(panel, split)
                d_of = {str(x): k for k, x in enumerate(zs["dates"])}
                dday = np.array([d_of[str(x)] for x in Wi.date], int)
                ogm = (Wi.condition != "queue_onset").to_numpy()
                Mt = np.stack([P["gs1"], tad, P["cnn3"], ps, P["occ_cnn"], P["occ_tfm"], P["stid"]])[:, ogm]
                vr, nb = SB.signature(v[ogm], ok[ogm], ctx.vf)
                p0_sel = SB.select_p0(Mt, vr, nb, dday[ogm], t0[ogm], bank)
                base_og = np.tensordot(SB.WAU, Mt, 1)
                assert np.abs(base_og - p0[ogm]).max() < 1e-4, "the selection's AU weights do not reproduce p0"
                n_moved = int((np.abs(p0_sel - p0[ogm]) > 1e-6).any((1, 2)).sum())
                p0 = p0.copy()
                p0[ogm] = p0_sel
                print(f"[{split} {panel}] selection moved p0 on {n_moved} of {int(ogm.sum())} ongoing windows "
                      f"(bank {len(bank['d'])} windows)", flush=True)
            cov = ok.mean((1, 2)) if use_cov else None             # note 76's definition: eligible and observed
            feat = features(P, p0, cov)
            rh = gbm.predict(feat.reshape(-1, feat.shape[-1])).reshape(p0.shape).astype(np.float32)
            if stab:
                flat = feat.reshape(-1, feat.shape[-1])
                agree = sum((np.sign(m.predict(flat)).reshape(p0.shape) == np.sign(rh)).astype(np.int32)
                            for m in stab)
                dropped = int(((agree < args.stab_min) & (np.abs(rh) >= args.gate))[og_mask(Wi)].sum())
                rh = np.where(agree >= args.stab_min, rh, 0.0).astype(np.float32)
            step = args.eta * np.where(np.abs(rh) >= args.gate, rh, 0.0)
            p = np.clip(p0 + step, 0, 1) if args.update == "prob" else 1 / (1 + np.exp(-(lg(p0) + step)))
            n_ad = 0
            if args.adapt_dir is not None:
                bz = np.load(args.adapt_dir / f"beta_{split}_{panel}.npz")
                assert [str(c) for c in bz["cols"]] == COLS, "the adapter was fitted on other columns"
                assert args.update == "prob", "the adapter was fitted against clip(p0 + eta r_hat)"
                for k, v in (("w_pseudo", args.w_pseudo), ("eta", args.eta), ("gate", args.gate)):
                    assert abs(float(bz[k]) - v) < 1e-9, f"the adapter was fitted with {k} {float(bz[k])}"
                assert ps is not None, "the adapter's columns need the pseudo-supervised member"
                X = adapter_cols({"ps": ps, "old": old, "P": {"tfm8": P["tfm8"]}}, p0, p, rh)
                di = {str(x): k for k, x in enumerate(bz["dates"])}
                dd = np.array([di[str(x)] for x in Wi.date], int)
                delta = np.einsum("nfsk,nk->nfs", X.astype(np.float64), bz["beta"][dd])
                lam_d = bz["lam"][dd][:, None, None]
                p = np.clip(p + lam_d * delta, 0, 1).astype(np.float32)
                n_ad = int((lam_d[:, 0, 0] > 0).sum())
            og = (Wi.condition != "queue_onset").to_numpy()
            w_of = {w: k for k, w in enumerate(Wi.window_id)}
            s_of = {l: k for k, l in enumerate(ctx.z["link_ids"])}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = np.asarray(slot(Q.timestamp)) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy().astype(int)
            keep = og[wi]
            Qk = Q[keep].copy()
            Qk["queue_pred"] = (p[wi[keep], hi[keep], si[keep]] >= args.tau).astype(int)
            ch = int(((p[og] >= args.tau) != (p0[og] >= args.tau)).sum())
            changed += ch
            frames.append(Qk)
            print(f"[{split} {panel}] ongoing rows {int(keep.sum()):,}, corrected cells {ch}, "
                  f"|r| mean {np.abs(rh[og]).mean():.4f}"
                  + (f", coverage {cov[og].mean():.3f}" if use_cov else "")
                  + (f", gated cells vetoed by the blocks {dropped}" if stab else "")
                  + (f", windows the run adapter reaches {n_ad}" if args.adapt_dir is not None else ""),
                  flush=True)
            ctx.z = ctx.q = ctx.el = None
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"ongoing_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'ongoing_{split}.csv'}: {len(out):,} rows, share 1 = "
              f"{out.queue_pred.mean():.4f}, corrected window cells {changed}", flush=True)


if __name__ == "__main__":
    main()
