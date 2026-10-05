"""note 165 copy (--labels recovered). Onset: recalibrate the STID's evidence path per run with FiLM, leaving its identity frozen.

Candidate AP's onset blend is 0.35 H + 0.175 ad9b + 0.175 pseudo + 0.30 STID. STID carries the static link
identity that gave AN its gain; the pseudo member carries the run's own supervision. They sit in separate models,
and a convex blend cannot express "what this link is like" x "what today's run is doing".

So adapt the STID with the run's past-day causal labels, but freeze the identity: past-day queue locations used
as a prior strengthened the habitual-bottleneck bias before, so `e_link` must not move. Everything is frozen and
only a FiLM pair is learned,

    h' = gamma_d * h + beta_d,     gamma initialised to 1 and beta to 0

so day 1 of a month is the unadapted STID exactly. `--film series` modulates only the history branch, which is
the evidence; `--film pre` modulates the concatenated vector before the head; `--film both` does both.

Surface: the held-out tail's onset windows at official-like coverage (`cache_onset_oof.py --truth-hist`, history
0.764 observed against the official 0.761). Note 108 showed the tail cannot rank onset MIXTURES, because H is a
same-run memory inflated there - but this compares one member against an adapted version of itself at fixed
weights, which that confound does not touch.

`--pool` is what production can actually supervise on. `day1` is the surface's own definition, one first-morning
onset per day, which on the scored months would give an official onset window on day 3 only two past examples -
and 97.5% of the official onset windows sit on days 1-7. `broad` instead takes every origin whose history and
slot T hold no visible queue and whose T+30 does, so an episode contributes about F windows and a day with a
morning and an evening episode twice that; that is the pool `scripts/t2_pseudo_adapt_official.py` already builds
for the ongoing selector, restricted to the onset selector. `--max-past-days 6` caps the day budget to what an
official onset window has.

The layer is the TRUE one restricted to eligibility, which is not a leak: the train split's obs layer is masked
BELOW the official coverage, and truth-within-elig reproduces it (0.764 against 0.761, note 108).

    python occ_lab/onset_film.py --film series --steps 40 --pool broad --max-past-days 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
sys.path.insert(0, str(PROJ / "analysis"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402
from t2_expected_iou import select  # noqa: E402
import t2_first_onset_eval as FO  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
WB = (0.35, 0.175, 0.175, 0.30)          # H, ad9b, pseudo, STID


class Film(nn.Module):
    """A frozen OnsetSTID with one or two FiLM pairs, the only trainable tensors."""

    def __init__(self, net: OnsetSTID, where: str):
        super().__init__()
        self.net, self.where = net, where
        for p in net.parameters():
            p.requires_grad_(False)
        d = net.e_link.weight.shape[1] if hasattr(net.e_link, "weight") else 32
        c = net.head.in_features
        h_dim = c - 3 * d if where in ("series", "both") else c
        self.g_s = nn.Parameter(torch.ones(max(h_dim, 1)))
        self.b_s = nn.Parameter(torch.zeros(max(h_dim, 1)))
        self.g_p = nn.Parameter(torch.ones(c))
        self.b_p = nn.Parameter(torch.zeros(c))

    def forward(self, x, lid, tod, dow):
        n = self.net
        B, Cc, Hh, S = x.shape
        h = n.series(x.permute(0, 3, 2, 1).reshape(B, S, Hh * Cc))
        if self.where in ("series", "both"):
            h = self.g_s * h + self.b_s
        e = n.e_link(lid)
        parts = [h, e, n.e_tod(tod)[:, None].expand(-1, S, -1), n.e_dow(dow)[:, None].expand(-1, S, -1)]
        h = torch.cat(parts, -1)
        if self.where in ("pre", "both"):
            h = self.g_p * h + self.b_p
        for blk in n.blocks:
            h = h + blk(h)
        return n.head(n.norm(h)).permute(0, 2, 1)

    def film(self):
        return [self.g_s, self.b_s] if self.where == "series" else \
               [self.g_p, self.b_p] if self.where == "pre" else [self.g_s, self.b_s, self.g_p, self.b_p]


def build(ctx, stat, profs, panel, z, w, m):
    x = C.make_x(ctx, stat, w, m, z)
    hi = w["t0"][m][:, None] + np.arange(-C.H, 0)
    return np.concatenate([x, occ_channels(z["obs_occ"][w["d"][m][:, None], hi], w["ok"][m], profs[panel])], 1)


def onset_pool(vf, z):
    """Every origin with a clean history and slot T and a visible queue at T+30, in the official 04-14 h band."""
    v, qf = z["true_speed"], z["true_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(qf)
    q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    T = q.shape[1]
    ds, ts = [], []
    for d in range(q.shape[0]):
        for t0 in range(max(FO.LO, C.H + 1), min(FO.HI, T - C.F - 1)):
            if q[d, t0 - C.H:t0 + 1].any() or not q[d, t0 + C.F].any():
                continue
            ds.append(d); ts.append(t0)
    d, t0 = np.array(ds, int), np.array(ts, int)
    hi = t0[:, None] + np.arange(-C.H, 0)
    fi = t0[:, None] + np.arange(1, C.F + 1)
    return dict(d=d, t0=t0, v=v[d[:, None], hi], qf=qf[d[:, None], hi], ok=ok[d[:, None], hi],
                fut=q[d[:, None], fi], efut=ok[d[:, None], fi])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="series", choices=("series", "pre", "both"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--pool", default="day1", choices=("day1", "broad"),
                    help="day1: the surface's own one-onset-per-day windows; broad: every clean-history origin "
                         "whose T+30 is queued, which is what production can build")
    ap.add_argument("--max-past-days", type=int, default=0, help="0 = every earlier tail day")
    ap.add_argument("--labels", default="true", choices=("true", "recovered"),
                    help="recovered: the supervision windows' T+30 labels are replaced by the head of the queue "
                         "visible after the official blank (note 165), as production would see them")
    ap.add_argument("--rec-width", type=int, default=1, help="recovered label = head +- this many links")
    ap.add_argument("--rec-soft", action="store_true", help="recovered label 1 at the head, 0.5 on its neighbours")
    args = ap.parse_args()
    profs = np.load(MODELS / "occ_profiles.npz")
    ranker = None
    if args.labels == "recovered":
        import hidden_onset_recover as HR
        ranker = HR.fit_ranker("2031-01")                  # fitted on Jun-Dec only: the tail is unseen
        rec_hit = [0, 0, 0]
    fam = {}
    t0_ = time.time()
    loff = 0
    for i, panel in enumerate(C.T2_PANELS):
        hic = CACHE / f"onset_hicov_{panel}.npz"
        if not hic.exists():
            continue
        ref = np.load(hic)
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        z = ctx.z
        fo = first_onsets(ctx.vf, z, truth=True, blank=None)
        keep = np.flatnonzero(z["dates"][fo["d"]] >= FIT_END)
        w = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(fo["d"]) else v) for k, v in fo.items()}
        assert len(w["d"]) == len(ref["y"]), f"{panel}: {len(w['d'])} windows against the cache's {len(ref['y'])}"
        y, e = ref["y"], ref["e"]
        dow = z["weekday"][w["d"]].astype(np.int64)
        days = np.unique(w["d"])
        if args.pool == "broad":
            sw = onset_pool(ctx.vf, z)
            kp = np.flatnonzero(z["dates"][sw["d"]] >= FIT_END)
            sw = {k: (v[kp] if isinstance(v, np.ndarray) and len(v) == len(sw["d"]) else v)
                  for k, v in sw.items()}
        else:
            sw = w
        sdow = z["weekday"][sw["d"]].astype(np.int64)
        lab_sw = sw["fut"][:, C.F - 1].astype(np.float32)
        if ranker is not None:                               # the post-blank queue on the same true-within-elig layer
            okz = ctx.el & np.isfinite(z["true_speed"])
            qdz = okz & (np.nan_to_num(z["true_speed"], nan=1e9) <= 0.6 * ctx.vf)
            vrz = np.where(okz, z["true_speed"] / ctx.vf, np.nan)
            pos = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(w["d"], w["t0"]))}
            lab_sw = np.zeros_like(lab_sw)
            for j in range(len(sw["d"])):
                jw = pos.get((int(sw["d"][j]), int(sw["t0"][j])))
                pH = ref["H"][jw] if jw is not None else np.full(ctx.S, 1.0 / ctx.S, np.float32)
                r = HR.recover(ranker, pH, qdz, vrz, int(sw["d"][j]), int(sw["t0"][j]), panel not in HR.UP_HIGHER,
                               width=max(args.rec_width, 1 if args.rec_soft else 0))
                if r is not None:
                    if args.rec_soft:
                        idx = np.flatnonzero(r)
                        lab_sw[j, idx] = 0.5
                        lab_sw[j, idx[len(idx) // 2]] = 1.0
                    else:
                        lab_sw[j] = r
                    tr_ = sw["fut"][j, C.F - 1] & sw["efut"][j, C.F - 1]
                    rec_hit[0] += 1; rec_hit[1] += bool((r & tr_).any()); rec_hit[2] += r.sum()
        print(f"  {panel}: supervision pool '{args.pool}' holds {len(sw['d'])} windows over "
              f"{len(np.unique(sw['d']))} days", flush=True)

        sd = torch.load(MODELS / "onset_stid_s0.pt", map_location=C.DEV)
        nets = []
        for s in range(4):
            d_ = torch.load(MODELS / f"onset_stid_s{s}.pt", map_location=C.DEV)
            n_ = OnsetSTID(d_["cin"], d_["n_links"], d_["n_slots"], ident=d_.get("ident", "id")).to(C.DEV)
            n_.load_state_dict(d_["state"])
            nets.append(Film(n_.eval(), args.film).to(C.DEV))
        _ = sd
        opts = [torch.optim.Adam(f.film(), lr=args.lr) for f in nets]
        P0 = np.zeros((len(w["d"]), ctx.S), np.float32)
        P1 = np.zeros_like(P0)
        rng = np.random.default_rng(11 + i)
        for k, day in enumerate(days):
            cur = np.flatnonzero(w["d"] == day)
            pd_ = days[:k] if args.max_past_days <= 0 else days[max(0, k - args.max_past_days):k]
            past = np.flatnonzero(np.isin(sw["d"], pd_))
            if k and len(past) >= 2:
                for f_, opt in zip(nets, opts):
                    for _ in range(args.steps):
                        m = rng.choice(past, min(args.bs, len(past)), replace=len(past) < args.bs)
                        xo = torch.from_numpy(np.ascontiguousarray(build(ctx, stat, profs, panel, z, sw, m))
                                              ).to(C.DEV, torch.float32)
                        tt = torch.from_numpy(lab_sw[m]).to(C.DEV)
                        mm = torch.from_numpy(sw["efut"][m][:, C.F - 1].astype(np.float32)).to(C.DEV)
                        lid_t = torch.from_numpy(np.broadcast_to(lid, (len(m), len(lid))).copy()
                                                 ).to(C.DEV, torch.long)
                        lg_ = f_(xo, lid_t,
                                 torch.from_numpy(sw["t0"][m].astype(np.int64)).to(C.DEV),
                                 torch.from_numpy(sdow[m]).to(C.DEV))[:, 0]
                        loss = (nn.functional.binary_cross_entropy_with_logits(lg_, tt, reduction="none")
                                * mm).sum() / mm.sum().clamp(min=1)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
            if not len(cur):
                continue
            xo = torch.from_numpy(np.ascontiguousarray(build(ctx, stat, profs, panel, z, w, cur))
                                  ).to(C.DEV, torch.float32)
            a_ = torch.from_numpy(np.broadcast_to(lid, (len(cur), len(lid))).copy()).to(C.DEV, torch.long)
            b_ = torch.from_numpy(w["t0"][cur].astype(np.int64)).to(C.DEV)
            c_ = torch.from_numpy(dow[cur]).to(C.DEV)
            with torch.no_grad():
                P0[cur] = np.mean([torch.sigmoid(f_.net(xo, a_, b_, c_))[:, 0].cpu().numpy()
                                   for f_ in nets], 0)
                P1[cur] = np.mean([torch.sigmoid(f_(xo, a_, b_, c_))[:, 0].cpu().numpy() for f_ in nets], 0)

        def score(st):
            out = []
            for j in range(len(w["d"])):
                m = e[j]
                if not m.any() or not y[j][m].any():
                    continue
                out.append(len((np.flatnonzero(select(st[j][m], "ratio", 1.0) & y[j][m]))) /
                           max(int((select(st[j][m], "ratio", 1.0) | y[j][m]).sum()), 1))
            return float(np.mean(out))

        H, ad = ref["H"], ref["ad9b"]
        for nm, st in (("STID alone", P0), ("FiLM-STID alone", P1)):
            fam.setdefault(nm, {})[panel] = score(st)
        for nm, st in (("AP blend, plain STID", P0), ("AP blend, FiLM STID", P1)):
            fam.setdefault(nm, {})[panel] = score(WB[0] * H + (WB[1] + WB[2]) * ad + WB[3] * st)
        print(f"  {panel}: {len(w['d'])} tail onset windows ({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None

    if ranker is not None:
        print(f"\nrecovered labels: {rec_hit[0]} windows, overlap a true T+30 cell in "
              f"{rec_hit[1] / max(rec_hit[0], 1):.3f}, cells per label {rec_hit[2] / max(rec_hit[0], 1):.1f}")
    print(f"\nonset IoU on the held-out tail at official-like coverage, FiLM on '{args.film}', "
          f"{args.steps} steps, pool '{args.pool}', past-day cap "
          f"{args.max_past_days or 'none'}\n")
    print(f"{'setting':<24}{'IoU':>9}{'delta':>9}")
    for a, b in (("STID alone", "FiLM-STID alone"), ("AP blend, plain STID", "AP blend, FiLM STID")):
        if a not in fam:
            continue
        va, vb = C.family_mean(fam[a]), C.family_mean(fam[b])
        print(f"{a:<24}{va:>9.4f}")
        print(f"{b:<24}{vb:>9.4f}{vb - va:>+9.4f}")


if __name__ == "__main__":
    main()
