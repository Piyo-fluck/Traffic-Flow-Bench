"""Onset decisions for the official windows with the onset STID as a third member (note 93).

    p(T+30) = w_H * LightGBM H + w_ad * past-day-adapted CNN (9b) + w_st * onset STID
    decision = the expected-IoU ratio rule (c = 1.0) over the window's links, at T+30 only

Production (candidates T .. AM) uses w_H 0.7, w_ad 0.3, w_st 0. Writes onset_<split>.csv (window_id, timestamp,
link_id, queue_pred) for the ONSET windows only, so `scripts/t2_patch_ongoing.py` can drop them into a finished
upload without touching its ongoing half.

    .venv/Scripts/python.exe scripts/t2_onset_blend_predict.py --out-dir work/sub_t2onset \
        --h-dir work/sub_t2robust_bag5 --ad-dir work/sub_t2adaptcnn \
        --stid occ_lab/models/onset_stid_s0.pt ... --profiles work/models_remote/occ_profiles.npz \
        --w-h 0.35 --w-ad 0.35 --w-st 0.30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "occ_lab"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from graph_ident import adj as chain_adj, lappe  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from t2_occ_blend_predict import occ_from_release  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID, predict as stid_predict  # noqa: E402

C.cnn.REC = 0.0


def cbm_official(models, panel, split, v, qf, ok, t0, dates):
    """note 171: the onset CBM probability (mean over seeds) of the official windows. Input = the window history
    (mainline) + the split's released ramp flows at the same history slots (slots < T only)."""
    import cbm_onset as M
    stc = M.panel_static(panel)
    vf, q95 = stc["vf"], stc["q95"]
    z = C.load(panel, split)
    rp = M.load_ramps(panel, split, z)
    d_of = {str(x): k for k, x in enumerate(z["dates"])}
    d = np.array([d_of[str(x)] for x in dates])
    idx = t0[:, None] + np.arange(-M.H, 0)[None]
    vv = np.where(ok, np.nan_to_num(v) / vf, 0).clip(0, 1.5)
    qq = np.where(ok, np.nan_to_num(qf) / q95, 0).clip(0, 3)
    on = np.where(rp["on_v"][d[:, None], idx], np.log1p(np.nan_to_num(rp["on_q"][d[:, None], idx]).clip(0)) / 6, 0)
    off = np.where(rp["off_v"][d[:, None], idx], np.log1p(np.nan_to_num(rp["off_q"][d[:, None], idx]).clip(0)) / 6, 0)
    X = np.stack([vv, qq, ok.astype(np.float32), on, off], 2).astype(np.float16).astype(np.float32)  # (N, H, CH, S)
    N, S = len(t0), len(vf)
    X = X.reshape(N, M.H * M.CH, S)
    tod = t0 / 288 * 2 * np.pi
    wk = np.array([pd.Timestamp(str(x)).weekday() >= 5 for x in dates], np.float32)
    g = np.broadcast_to(np.stack([np.sin(tod), np.cos(tod), wk], 1)[:, :, None], (N, 3, S))
    out = []
    with torch.no_grad():
        for ck in models:
            st = stc["st"][ck.get("st", list(range(8)))]
            x = np.concatenate([X, np.broadcast_to(st[None], (N,) + st.shape), g], 1).astype(np.float32)
            dil = tuple(ck.get("dil", (1, 2, 4, 8, 16, 1, 2, 4)))
            cn = M.ConceptNet(ck["cin"], dil=dil).to(M.DEV); cn.load_state_dict(ck["cn"]); cn.eval()
            gh = M.OnsetHead(resid=False).to(M.DEV); gh.load_state_dict(ck["g"]); gh.eval()
            xt = torch.from_numpy(x).to(M.DEV)
            out.append(torch.sigmoid(gh(M.concept_feats(cn(xt)))).cpu().numpy())
    return np.mean(out, 0).astype(np.float32)


def cbm_rank_swap(Pj, cand, reg, cbj, lam, BN):
    """note 171: rank = region mass of Pj x (CBM region max) ^ lam; swap the peaks where the top changes.
    Returns (new Pj, old top, new top)."""
    m1 = np.bincount(reg, weights=Pj, minlength=len(cand))
    cb = np.array([cbj[reg == r].max() for r in range(len(cand))])
    rk = m1 / max(m1.sum(), 1e-9) * np.maximum(cb, 1e-6) ** lam
    return BN.swap(Pj, cand, reg, m1, rk), int(np.argmax(m1)), int(np.argmax(rk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--h-dir", type=Path, required=True, help="LightGBM onset probabilities")
    ap.add_argument("--ad-dir", type=Path, required=True, help="past-day-adapted CNN probabilities (idea 9b)")
    ap.add_argument("--ad2-dir", type=Path, default=None,
                    help="note 106: the pseudo-supervised adapted CNN, as a fourth member")
    ap.add_argument("--w-ad2", type=float, default=0.0)
    ap.add_argument("--stid", type=Path, nargs="*", default=[])
    ap.add_argument("--st-dir", type=Path, default=None,
                    help="precomputed STID probabilities (scripts/t2_onset_film_official.py), used in place of "
                         "--stid; the FiLM member is adapted per run so it cannot be evaluated inline")
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--w-h", type=float, default=0.35)
    ap.add_argument("--w-ad", type=float, default=0.35)
    ap.add_argument("--w-st", type=float, default=0.30)
    ap.add_argument("--ratio-c", type=float, default=1.0)
    ap.add_argument("--bn-model", type=Path, default=None,
                    help="note 129: the candidate-bottleneck classifier (occ_lab/bottleneck_fit.py); where its top "
                         "candidate differs from the blend's, the two peaks' heights are swapped before the ratio rule")
    ap.add_argument("--bn-cap", action="store_true",
                    help="note 145: the classifier also sees the learned discharge-capacity headroom "
                         "(occ_lab/bottleneck_capacity.py; fit with bottleneck_fit.py --cap)")
    ap.add_argument("--bn-prior4", action="store_true",
                    help="note 145: + training-month capacity and the month's last <= 4 earlier days "
                         "(fit with bottleneck_fit.py --prior4)")
    ap.add_argument("--setdec", action="store_true",
                    help="notes 152-153: the onset set decoder - IoU-LambdaRank candidate ranking + swap, then a "
                         "second region added when its predicted marginal IoU exceeds the fitted threshold "
                         "(occ_lab/onset_setdec_fit.py); replaces --bn-model")
    ap.add_argument("--setdec-tag", default="", help="'_plus': the ranker with the new candidate evidence (note 155)")
    ap.add_argument("--g-dirs", type=Path, nargs="*", default=[],
                    help="note 169: w192 SSL CNN adapted on the official windows (co-trained with generated runs); "
                         "the mean over these dirs is a fifth member G")
    ap.add_argument("--g-alt-dirs", type=Path, nargs="*", default=[], help="G for the --g-alt-panels instead")
    ap.add_argument("--g-alt-panels", nargs="*", default=[])
    ap.add_argument("--w-g", type=float, default=0.0)
    ap.add_argument("--online-w", type=Path, default=None,
                    help="note 169: run-adaptive member weights per (split, panel, day) from scripts/onset_online_table.py; "
                         "onset p = (1 - lam) * the fixed blend + lam * sum_k w_k member_k over H, ad9b, pseudo, STID, G")
    ap.add_argument("--online-lam", type=float, default=0.75)
    ap.add_argument("--cbm", type=Path, nargs="*", default=[],
                    help="note 171: onset CBM checkpoints (occ_lab/cbm_onset.py); a second opinion on the candidate "
                         "choice after the set decoder's stage 1 (or on the blend without --setdec)")
    ap.add_argument("--cbm-lam", type=float, default=1.0)
    ap.add_argument("--cbm-replace", type=float, default=None,
                    help="note 171 rule: no swap; replace the final decision by the CBM set around the CBM's top peak "
                         "(+-3 links, ratio rule on the CBM) when cbm(peak) - max cbm(decision) >= this margin")
    args = ap.parse_args()
    bn = None
    if args.setdec:
        import lightgbm as lgb
        import bottleneck_cls as BN
        import bottleneck_capacity as BK
        import onset_rerank as RR
        import onset_stage2_du as S2
        md = ROOT / "occ_lab" / "models"
        tg = args.setdec_tag
        rankers = [lgb.Booster(model_file=str(f)) for f in sorted(md.glob(f"onset_rank{tg}_s*.txt"))]
        dus = [lgb.Booster(model_file=str(f)) for f in sorted(md.glob(f"onset_du{tg}_s*.txt"))]
        tau = float((md / f"onset_du{tg}_tau.txt").read_text())
        print(f"set decoder: {len(rankers)} rankers, {len(dus)} dU regressors, threshold {tau}", flush=True)
        logs = []
    if args.bn_model is not None:
        import lightgbm as lgb
        sys.path.insert(0, str(ROOT / "occ_lab"))
        import bottleneck_cls as BN
        bn = lgb.Booster(model_file=str(args.bn_model))
    assert abs(args.w_h + args.w_ad + args.w_st + args.w_ad2 + args.w_g - 1) < 1e-6, "the onset weights must sum to 1"
    assert args.stid or args.st_dir or args.w_st == 0, "--w-st needs --stid checkpoints or --st-dir"
    assert not (args.stid and args.st_dir), "--stid and --st-dir are two ways to supply the same member"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets, ident, k_lap = [], "id", 12
    for p in args.stid:
        d = torch.load(p, map_location=C.DEV)
        ident, k_lap = d.get("ident", "id"), d.get("k_lap", 12)
        net = OnsetSTID(d["cin"], d["n_links"], d["n_slots"], d=d.get("d", 96), ds=d.get("ds", 32), blocks=d.get("blocks", 3),
                        ident=ident, k_lap=k_lap).to(C.DEV)
        net.load_state_dict(d["state"])
        nets.append(net.eval())
    cbm_models = [torch.load(p_, map_location=C.DEV) for p_ in args.cbm]
    profs = np.load(args.profiles)
    print(f"onset = {args.w_h:.2f} H + {args.w_ad:.2f} ad9b + {args.w_ad2:.2f} pseudo + {args.w_g:.2f} G + {args.w_st:.2f} STID "
          f"({len(nets)} seeds, identity '{ident}'), ratio c {args.ratio_c}", flush=True)
    for split in C.SPLITS:
        frames, loff = [], 0
        for i, panel in enumerate(C.T2_PANELS):
            ctx = C.Ctx(panel, i)
            ctx.loff = 0
            stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
            lid = np.arange(ctx.S) + loff
            loff += ctx.S
            base = RELEASE / "task2" / panel / split
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            ids = Wi.window_id.to_numpy().astype(str)
            fh, fa = (np.load(d / f"prob_{split}_{panel}.npz") for d in (args.h_dir, args.ad_dir))
            assert (fh["window_id"] == ids).all() and (fa["window_id"] == ids).all(), "window order differs"
            P = args.w_h * fh["prob"][:, C.F - 1] + args.w_ad * fa["prob"][:, C.F - 1]
            mem_h, mem_a = fh["prob"][:, C.F - 1], fa["prob"][:, C.F - 1]
            mem_p = np.zeros_like(mem_h)
            mem_s = np.zeros_like(mem_h)
            if args.w_ad2:
                f2 = np.load(args.ad2_dir / f"prob_{split}_{panel}.npz")
                assert (f2["window_id"] == ids).all(), "second adapted window order differs"
                P = P + args.w_ad2 * f2["prob"][:, C.F - 1]
                mem_p = f2["prob"][:, C.F - 1]
            mem_g = None
            if args.w_g or args.online_w is not None:
                gd = args.g_alt_dirs if panel in args.g_alt_panels else args.g_dirs
                fg = [np.load(d_ / f"prob_{split}_{panel}.npz") for d_ in gd]
                assert all((f_["window_id"] == ids).all() for f_ in fg), "G window order differs"
                mem_g = np.mean([f_["prob"][:, C.F - 1] for f_ in fg], 0)
                P = P + args.w_g * mem_g
            if args.st_dir is not None and args.w_st:
                fs = np.load(args.st_dir / f"prob_{split}_{panel}.npz")
                assert (fs["window_id"] == ids).all(), "STID window order differs"
                P = P + args.w_st * fs["prob"][:, C.F - 1]
            if nets:
                x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
                xo = np.concatenate([x, occ_channels(occ_from_release(ctx, Hh, Wi, t0), ok, profs[panel])], 1)
                dow = np.array([pd.Timestamp(str(d)).weekday() for d in Wi.date], np.int64)
                lap = lappe(ctx, k_lap) if ident == "lappe" else None
                adjm = chain_adj(ctx) if ident == "ab" else None
                mem_s = np.mean([stid_predict(n, xo, lid, t0.astype(np.int64), dow, lap=lap, adj=adjm)
                                 for n in nets], 0)
                P = P + args.w_st * mem_s
            is_on = (Wi.condition == "queue_onset").to_numpy()
            n_swap = 0
            n_cbm = 0
            cbm_p = cbm_official(cbm_models, panel, split, v, qf, ok, t0, Wi.date.to_numpy()) if cbm_models else None
            n_online = 0
            if args.online_w is not None:
                import json
                tab = json.loads(args.online_w.read_text())[f"{split}|{panel}"]
                zd = C.load(panel, split)
                d_of_ = {str(x): k for k, x in enumerate(zd["dates"])}
                for j in np.flatnonzero(is_on):
                    wv = tab[d_of_[str(Wi.date[j])]]
                    if wv is None:
                        continue
                    mix = wv[0] * mem_h[j] + wv[1] * mem_a[j] + wv[2] * mem_p[j] + wv[3] * mem_s[j] + wv[4] * mem_g[j]
                    P[j] = (1 - args.online_lam) * P[j] + args.online_lam * mix
                    n_online += 1
            if bn is not None:
                z = C.load(panel, split)
                vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
                qd, nd = BN.month_share(z, vf)
                if args.bn_cap:
                    import bottleneck_capacity as BK
                    A_cap = BK.day_arrays(z, vf)
                if args.bn_prior4:
                    import bottleneck_capacity as BK
                    A_cap = BK.day_arrays(z, vf)
                    Pr = BK.prior_stats(panel, vf)
                d_of = {str(x): k for k, x in enumerate(z["dates"])}
                for j in np.flatnonzero(is_on):
                    r = BN.cand_rows(P[j], (mem_h[j], mem_a[j], mem_p[j], mem_s[j]), v[j], qf[j], ok[j], vf, cap,
                                     np.asarray(ctx.lanes, np.float32), stat, qd, nd, d_of[str(Wi.date[j])],
                                     int(t0[j]))
                    if r is None:
                        continue
                    X, cand, reg, mass = r
                    if args.bn_cap:
                        X = np.concatenate([X, BK.cap_rows(A_cap, d_of[str(Wi.date[j])], int(t0[j]), qf[j], ok[j],
                                                           cand)], 1)
                    if args.bn_prior4:
                        dj = d_of[str(Wi.date[j])]
                        lo = max(0, dj - 4)
                        As = {k: w[lo:dj + 1] for k, w in A_cap.items()}
                        X = np.concatenate([X, BK.prior_rows(Pr, int(t0[j]), qf[j], ok[j], cand),
                                            BK.cap_rows(As, dj - lo, int(t0[j]), qf[j], ok[j], cand)], 1)
                    pc = bn.predict(X)
                    q = BN.swap(P[j], cand, reg, mass, pc / max(pc.sum(), 1e-9))
                    n_swap += int(not np.array_equal(q, P[j]))
                    P[j] = q
            add = np.zeros((len(Wi), ctx.S), bool)
            if args.setdec:
                z = C.load(panel, split)
                vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
                kcrit = np.asarray(ctx.kcrit, np.float32)
                qd, nd = BN.month_share(z, vf)
                d_of = {str(x): k for k, x in enumerate(z["dates"])}
                A_cap = BK.day_arrays(z, vf)
                Pr = BK.prior_stats(panel, vf)
                for j in np.flatnonzero(is_on):
                    dj = d_of[str(Wi.date[j])]
                    mems = (mem_h[j], mem_a[j], mem_p[j], mem_s[j]) + ((mem_g[j],) if mem_g is not None else ())
                    r = BN.cand_rows(P[j], mems, v[j], qf[j], ok[j], vf, cap,
                                     np.asarray(ctx.lanes, np.float32), stat, qd, nd, dj, int(t0[j]))
                    if r is None:
                        continue
                    X, cand, reg, mass = r
                    lo = max(0, dj - 4)
                    As = {k: w[lo:dj + 1] for k, w in A_cap.items()}
                    Xp = np.concatenate([X, BK.prior_rows(Pr, int(t0[j]), qf[j], ok[j], cand),
                                         BK.cap_rows(As, dj - lo, int(t0[j]), qf[j], ok[j], cand)], 1)
                    p0 = P[j].copy()
                    nf = RR.new_feats(p0, cand, reg, v[j], qf[j], ok[j], vf, cap, kcrit)
                    Xr = Xp if not args.setdec_tag.startswith("_plus") else np.concatenate(
                        [Xp, nf, Xp[:, :5] - Xp[:, :5].max(0, keepdims=True)], 1)
                    sc = np.mean([m.predict(Xr) for m in rankers], 0)
                    P[j] = BN.swap(p0, cand, reg, mass, sc - sc.min() + 1e-3)
                    if cbm_p is not None and args.cbm_replace is None:
                        q, a_, b_ = cbm_rank_swap(P[j], cand, reg, cbm_p[j], args.cbm_lam, BN)
                        if not np.array_equal(q, P[j]):
                            n_cbm += 1
                            P[j] = q
                            ta, tb = int(np.argmax(sc)), b_
                            sc = sc.copy(); sc[ta], sc[tb] = sc[tb], sc[ta]
                    Sb, rows = S2.pair_rows(P[j], sc, Xp, nf, cand, reg)
                    rec = dict(window_id=Wi.window_id[j], top_prod=int(np.argmax(mass)), top_s1=int(np.argmax(sc)),
                               stage1_changed=int(np.argmax(mass)) != int(np.argmax(sc)), add_region=-1, add_du=np.nan,
                               add_cells=0)
                    if rows:
                        du = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in dus], 0)
                        k = int(np.argmax(du))
                        rec["add_du"] = float(du[k])
                        if du[k] > tau:
                            add[j] = rows[k][1]
                            rec["add_region"], rec["add_cells"] = int(rows[k][2]), int(rows[k][1].sum())
                    logs.append({**rec, "split": split, "panel": panel})
            pred = np.zeros((len(Wi), C.F, ctx.S), bool)
            for j in np.flatnonzero(is_on):
                pred[j, C.F - 1] = select(P[j], "ratio", args.ratio_c) | add[j]
                if cbm_p is not None and args.cbm_replace is not None:
                    cbj, A0 = cbm_p[j], pred[j, C.F - 1]
                    pk = int(np.argmax(cbj))
                    m = np.zeros(len(cbj), bool); m[max(0, pk - 3):pk + 4] = True
                    Ac = select(cbj * m, "ratio", 1.0) & m
                    c0 = cbj[A0].max() if A0.any() else 0.0
                    if Ac.any() and (Ac & ~A0).any() and cbj[pk] - c0 >= args.cbm_replace:
                        pred[j, C.F - 1] = Ac
                        n_cbm += 1
            w_of = {w: k for k, w in enumerate(Wi.window_id)}
            s_of = {l: k for k, l in enumerate(ctx.z["link_ids"])}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = np.asarray(slot(Q.timestamp)) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy().astype(int)
            keep = is_on[wi]
            Qk = Q[keep].copy()
            Qk["queue_pred"] = pred[wi[keep], hi[keep], si[keep]].astype(int)
            frames.append(Qk)
            print(f"[{split} {panel}] onset rows {int(keep.sum()):,}, cells predicted {int(pred[is_on].sum())}, "
                  f"bottleneck swaps {n_swap}, online-weighted windows {n_online}, CBM swaps {n_cbm}",
                  flush=True)
            ctx.z = ctx.q = ctx.el = None
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"onset_{split}.csv", index=False)
        if args.setdec:
            pd.DataFrame([r for r in logs if r["split"] == split]).to_csv(args.out_dir / f"setdec_log_{split}.csv",
                                                                           index=False)
        print(f"wrote {args.out_dir / f'onset_{split}.csv'}: {len(out):,} rows, "
              f"share 1 = {out.queue_pred.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
