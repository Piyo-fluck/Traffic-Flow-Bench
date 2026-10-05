"""Note 165 on the SCORED months: note 120's onset FiLM supervised by recovered hidden onsets.

For every lab onset window on day d of a scored month, the supervision is the lab onset events of the 4 most recent
earlier days, each labelled by the recovered bottleneck (the head of the queue visible 65 min after its breakdown,
ranked with H; `hidden_onset_recover.fit_ranker` fitted on the training months) - never by its true label. The FiLM
(history branch of the 4 frozen onset STIDs) is adapted day by day, 20 steps, lr 3e-3, carried forward.

    plain     0.35 H + 0.175 ad9b + 0.175 pseudo + 0.30 STID (the frozen STIDs)
    FiLM      the same with the FiLM-adapted STIDs
    AP        the production onset blend (STID member = onset_adapt_stid30)
Onset family IoU (ratio rule, T+30) per month, and the recovered labels' hit rate.

    .venv/Scripts/python.exe occ_lab/film_scored_eval.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
sys.path.insert(0, str(LAB.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402
import hidden_onset_recover as HR  # noqa: E402
import onset_film_recov as OF  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
STEPS, LR, BS, CAP = int(os.environ.get("FILM_STEPS", 20)), 3e-3, 16, 4
FRESH = int(os.environ.get("FILM_FRESH_ROUNDS", 0))    # >0: the official regime - a fresh FiLM per day, R rounds
OUT_TAG = os.environ.get("FILM_TAG", "")


def main() -> None:
    warnings.simplefilter("ignore")
    t_start = time.time()
    ranker = HR.fit_ranker("9999")                          # every training month
    profs = np.load(MODELS / "occ_profiles.npz")
    fam, hits = {}, [0, 0]
    loff = 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        S = ctx.S
        lid = np.arange(S) + loff
        loff += S
        vf = np.asarray(ctx.vf, np.float32)
        for s in C.SPLITS:
            on, z, ref = W[s]["on"], W[s]["z"], W[s]["ref"]
            if not len(on["d"]):
                continue
            Hq, ad = ref["on_H"], ref["on_ad9b"]
            ps = np.load(CACHE / f"pseudo_s20_{s}_{panel}.npz")["on"].astype(np.float32)
            st = np.load(CACHE / f"onset_adapt_stid30_{s}_{panel}.npz")["prob"].astype(np.float32)
            v, ok = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"])
            qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
            vr = np.where(ok, v / vf, np.nan)
            # recovered labels of every lab event (used only as supervision for LATER days)
            lab = np.zeros((len(on["d"]), S), np.float32)
            has = np.zeros(len(on["d"]), bool)
            for n in range(len(on["d"])):
                r = HR.recover(ranker, Hq[n], qd, vr, int(on["d"][n]), int(on["t0"][n]), panel not in HR.UP_HIGHER, width=0)
                if r is not None:
                    lab[n] = r; has[n] = True
                    hits[0] += 1; hits[1] += bool((r & on["fut"][n, -1] & on["efut"][n, -1]).any())
            nets, opts = [], []
            for k in range(4):
                d_ = torch.load(MODELS / f"onset_stid_s{k}.pt", map_location=C.DEV)
                n_ = OF.OnsetSTID(d_["cin"], d_["n_links"], d_["n_slots"], ident=d_.get("ident", "id")).to(C.DEV)
                n_.load_state_dict(d_["state"])
                f_ = OF.Film(n_.eval(), "series").to(C.DEV)
                nets.append(f_); opts.append(torch.optim.Adam(f_.film(), lr=LR))
            dow = z["weekday"][on["d"]].astype(np.int64)
            P0 = np.zeros((len(on["d"]), S), np.float32); P1 = np.zeros_like(P0)
            rng = np.random.default_rng(11 + i)
            days = np.unique(on["d"])
            for k, day in enumerate(days):
                pdays = days[max(0, k - CAP):k]
                past = np.flatnonzero(np.isin(on["d"], pdays) & has)
                if len(past) >= 2:
                    if FRESH:
                        for f_, o_ in zip(nets, range(len(opts))):
                            with torch.no_grad():
                                for t_ in f_.film():
                                    t_.fill_(1.0 if t_ is f_.g_s or t_ is f_.g_p else 0.0)
                            opts[o_] = torch.optim.Adam(f_.film(), lr=LR)
                    for f_, opt in zip(nets, opts):
                        for _ in range(STEPS * (FRESH or 1)):
                            m = rng.choice(past, min(BS, len(past)), replace=len(past) < BS)
                            xo = torch.from_numpy(np.ascontiguousarray(OF.build(ctx, stat, profs, panel, z, on, m))).to(C.DEV, torch.float32)
                            li = torch.from_numpy(np.broadcast_to(lid, (len(m), S)).copy()).to(C.DEV, torch.long)
                            lg_ = f_(xo, li, torch.from_numpy(on["t0"][m].astype(np.int64)).to(C.DEV),
                                     torch.from_numpy(dow[m]).to(C.DEV))[:, 0]
                            tt = torch.from_numpy(lab[m]).to(C.DEV)
                            mm = torch.from_numpy(on["efut"][m][:, C.F - 1].astype(np.float32)).to(C.DEV)
                            loss = (nn.functional.binary_cross_entropy_with_logits(lg_, tt, reduction="none") * mm).sum() / mm.sum().clamp(min=1)
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                cur = np.flatnonzero(on["d"] == day)
                xo = torch.from_numpy(np.ascontiguousarray(OF.build(ctx, stat, profs, panel, z, on, cur))).to(C.DEV, torch.float32)
                a_ = torch.from_numpy(np.broadcast_to(lid, (len(cur), S)).copy()).to(C.DEV, torch.long)
                b_ = torch.from_numpy(on["t0"][cur].astype(np.int64)).to(C.DEV)
                c_ = torch.from_numpy(dow[cur]).to(C.DEV)
                with torch.no_grad():
                    P0[cur] = np.mean([torch.sigmoid(f_.net(xo, a_, b_, c_))[:, 0].cpu().numpy() for f_ in nets], 0)
                    P1[cur] = np.mean([torch.sigmoid(f_(xo, a_, b_, c_))[:, 0].cpu().numpy() for f_ in nets], 0)
            np.savez(CACHE / f"onset_film_recov{OUT_TAG}_{s}_{panel}.npz", prob=P1, plain=P0)
            other = 0.35 * Hq + 0.175 * ad + 0.175 * ps
            for nm, q in (("STID alone", P0), ("FiLM-STID alone", P1), ("plain blend", other + 0.30 * P0),
                          ("FiLM blend", other + 0.30 * P1), ("AP (production)", other + 0.30 * st)):
                fam.setdefault((s, nm), {})[panel] = C.onset_iou(q, on).mean()
            print(f"  {panel} {s}: {len(on['d'])} windows, {has.sum()} recovered ({time.time() - t_start:.0f}s)", flush=True)
    print(f"\nrecovered labels overlap a true T+30 cell: {hits[1] / max(hits[0], 1):.3f} of {hits[0]}")
    print(f"FiLM steps {STEPS}, past-day cap {CAP}\n{'':<18}" + "".join(f"{s:>12}" for s in C.SPLITS))
    for nm in ("STID alone", "FiLM-STID alone", "plain blend", "FiLM blend", "AP (production)"):
        print(f"{nm:<18}" + "".join(f"{C.family_mean(fam[(s, nm)]):>12.4f}" for s in C.SPLITS))


if __name__ == "__main__":
    main()
