"""Evaluate the onset CBM (occ_lab/cbm_onset.py) on the lab onset windows under the official windows' condition
(BB's day-adapted members with K = 0..4 days of adaptation, averaged), next to note 129's candidate classifier.

    candidate top-1   production's top-5 peaks and regions (bottleneck_cls.build); label = the region holding the
                      most true T+30 queue cells. production = arg-max region mass; note 129 = LightGBM fitted on the
                      OTHER scored month (3 seeds); CBM = arg-max of the CBM probability inside the region (no fit on
                      scored data at all)
    onset IoU         window-weighted, diff to production with the window-level SE:
                      swap (note 129's decision rule with the CBM ranking), blend (1-a) base + a CBM

    .venv/Scripts/python.exe occ_lab/cbm_eval.py cbm_cbm_s0 cbm_direct_s0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import cbm_onset as M  # noqa: E402
import common as C  # noqa: E402

TAGS = sys.argv[1:]
LAMS = (0.25, 0.5, 1.0)
KS = range(5)
CA = LAB / "cache"
PAR = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}


class Lite:
    def __init__(self, panel):
        s = M.panel_static(panel)
        self.panel, self.vf, self.q95, self.st = panel, s["vf"], s["q95"], s["st"]
        self.S = len(self.vf)


def cbm_probs(tag, P, split, on):
    if "+" in tag:                                   # seed ensemble: mean of the members' probabilities
        return np.mean([cbm_probs(t, P, split, on) for t in tag.split("+")], 0)
    f = CA / f"cbm_{tag}_{split}_{P.panel}.npz"
    if f.exists():
        return np.load(f)["prob"]
    ck = torch.load(M.MODELS / f"{tag}.pt", map_location=M.DEV)
    z = M.load(P.panel, split)
    arrays = M.dyn(z, M.load_ramps(P.panel, split, z), P.vf, P.q95) + (z,)
    st = P.st[ck.get("st", list(range(8)))]
    dil = tuple(ck.get("dil", (1, 2, 4, 8, 16, 1, 2, 4)))
    x = torch.from_numpy(M.scored_inputs(P, split, on, st, arrays=arrays)).to(M.DEV)
    with torch.no_grad():
        if "cn" in ck:
            rs = bool(ck.get("resid", False))
            nf = ck.get("nfree", 0)
            cn = M.ConceptNet(ck["cin"], dil=dil, nfree=nf).to(M.DEV); cn.load_state_dict(ck["cn"]); cn.eval()
            g = M.OnsetHead(nc=M.NC + nf, resid=rs).to(M.DEV); g.load_state_dict(ck["g"]); g.eval()
            cf = M.concept_feats(cn(x))
            if rs:
                xp = torch.from_numpy(M.scored_inputs(P, split, on, st, shift=M.RS, arrays=arrays)).to(M.DEV)
                obs = torch.from_numpy(arrays[0][on["d"], on["t0"] - 1, :3].astype(np.float32)).to(M.DEV)
                cf = torch.cat([cf, M.residuals(cn(xp), obs)], 1)
            lg = g(cf)
        else:
            net = M.Direct(ck["cin"], dil=dil).to(M.DEV); net.load_state_dict(ck["net"]); net.eval()
            lg = net(x)
    p = torch.sigmoid(lg).cpu().numpy().astype(np.float32)
    np.savez_compressed(f, prob=p)
    return p


def main() -> None:
    lite = {}
    data = {}                                     # (K, panel) -> build output
    for K in KS:
        os.environ["ONSET_EARLY_K"] = str(K)
        for i, panel in enumerate(C.T2_PANELS):
            data[(K, panel)] = BC.build(i, panel)
        print(f"  built K={K}", flush=True)
    probs = {}
    for panel in C.T2_PANELS:
        lite[panel] = Lite(panel)
        for s, D in data[(0, panel)].items():
            for t in TAGS:
                probs[(t, s, panel)] = cbm_probs(t, lite[panel], s, D["on"])
    acc, iou = {}, {}
    for K in KS:
        # note 129 classifier, cross-month, 3 seeds
        n129 = {}
        for sd in range(3):
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                X = np.concatenate([data[(K, p)][s_fit]["X"] for p in C.T2_PANELS if s_fit in data[(K, p)]])
                y = np.concatenate([data[(K, p)][s_fit]["y"] for p in C.T2_PANELS if s_fit in data[(K, p)]])
                m = lgb.train({**PAR, "seed": sd}, lgb.Dataset(X, y), 200)
                for p in C.T2_PANELS:
                    if s_app in data[(K, p)]:
                        n129.setdefault((s_app, p), []).append(m.predict(data[(K, p)][s_app]["X"]))
        for s in C.SPLITS:
            for p in C.T2_PANELS:
                if s not in data[(K, p)]:
                    continue
                D = data[(K, p)][s]
                on, base = D["on"], D["base"]
                variants = {"production": base.copy(), "n129 swap": base.copy()}
                for t in TAGS:
                    pr = probs[(t, s, p)]
                    variants[f"{t} alone"] = pr
                    variants[f"{t} swap"] = base.copy()
                    for lam in LAMS:
                        variants[f"{t} x{lam}"] = base.copy()
                    for a_ in (0.1, 0.2, 0.3):
                        variants[f"{t} blend{a_}"] = (1 - a_) * base + a_ * pr
                pn = np.mean(n129[(s, p)], 0)
                off = 0
                for (n, cand, reg, mass), gs in zip(D["meta"], D["g"]):
                    yy = D["y"][off:off + gs]
                    pc = pn[off:off + gs]
                    off += gs
                    rank = {"production": mass, "n129": pc}
                    for t in TAGS:
                        pr = probs[(t, s, p)][n]
                        rank[t] = np.array([pr[reg == r].max() for r in range(len(cand))])
                        variants[f"{t} swap"][n] = BC.swap(base[n], cand, reg, mass, rank[t])
                        for lam in LAMS:          # production's region mass x the CBM region score ^ lam
                            rk = mass / max(mass.sum(), 1e-9) * np.maximum(rank[t], 1e-6) ** lam
                            rank[f"{t} x{lam}"] = rk
                            variants[f"{t} x{lam}"][n] = BC.swap(base[n], cand, reg, mass, rk)
                    variants["n129 swap"][n] = BC.swap(base[n], cand, reg, mass, pc)
                    if yy.any():
                        for k, v in rank.items():
                            acc.setdefault((s, k), []).append(float(np.argmax(v) == np.argmax(yy)))
                            acc.setdefault((s, k, p), []).append(float(np.argmax(v) == np.argmax(yy)))
                for k, v in variants.items():
                    iou.setdefault((s, k), []).append(C.onset_iou(v, on))
        print(f"  K={K} done", flush=True)
    print("\ncandidate top-1 (mean over K=0..4; windows with a labelled candidate)")
    for k in ["production", "n129"] + TAGS + [f"{t} x{l}" for t in TAGS for l in LAMS]:
        print(f"  {k:<22} " + "  ".join(f"{s} {np.mean(acc[(s, k)]):.3f} (n {len(acc[(s, k)]) // len(KS)})"
                                          for s in C.SPLITS))
    print("\nper panel top-1 (val / pri)")
    for p in C.T2_PANELS:
        print(f"  {p:<11} " + "  ".join(
            f"{k[-12:]} " + "/".join(f"{np.mean(acc[(s, k, p)]):.2f}" if (s, k, p) in acc else "  - " for s in C.SPLITS)
            for k in ["production", "n129"] + [f"{t} x1.0" for t in TAGS])
              + "   n " + "/".join(str(len(acc.get((s, "production", p), [])) // len(KS)) for s in C.SPLITS))
    print("\nonset IoU, window-weighted, mean over K (diff to production, SE)")
    names = sorted({k for (_, k) in iou}, key=lambda x: (x != "production", x))
    for k in names:
        cells = []
        for s in C.SPLITS:
            a = np.concatenate(iou[(s, k)]); b = np.concatenate(iou[(s, "production")])
            dd = a - b
            se = dd.std(ddof=1) / np.sqrt(len(dd) / len(KS)) if dd.std() > 0 else 0.0
            cells.append(f"{s} {a.mean():.4f} ({dd.mean():+.4f}, SE {se:.4f})")
        print(f"  {k:<26} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
