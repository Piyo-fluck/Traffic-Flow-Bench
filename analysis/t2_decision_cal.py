"""Task 2: per-group thresholds and calibration before the Bayes decision (note 42, items 6-7), from caches only.

Baseline = M's rules on the blank-aware official-like windows:
    ongoing  p = 0.5 lgbm + 0.5 cnn3, flag p >= 0.60
    onset    p = 0.7 lgbm + 0.3 cnn3, expected-IoU ratio decision (c 1.0) on the T+30 links
Variants (winners' tricks: region-specific thresholds in Weather4cast 2022; calibration before a decision):
    og_h      one tau per horizon step (coordinate ascent, grid 0.45..0.75)
    og_fam    one tau per corridor family
    og_logit  blend in logit space, one tau
    on_cal    CNN recalibrated as sigmoid(a logit p + b) before the blend, with blend weight w and ratio scale c
    on_logit  the same, blended in logit space
Every variant is FITTED on one split (all days) and SCORED on the other (a separate simulation run), both ways.
The output is the change against the baseline, with panels better/worse, for days 1-14 and all.

    .venv/Scripts/python.exe analysis/t2_decision_cal.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
SPLITS = ("validation", "private")
DAY_SETS = {"d1-14": 14, "all": 31}
TAU_GRID = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-x))


def load_all():
    data = {}
    for panel in T2_PANELS:
        for split in SPLITS:
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            cn = [np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz") for t in TAGS]
            for x in cn:
                assert (x["on_d"] == c["on_d"]).all() and (x["og_d"] == c["og_d"]).all(), "window sets differ"
            data[(panel, split)] = dict(c={k: c[k] for k in c.files},
                                        og_c=np.mean([x["og_p"] for x in cn], 0) if len(c["og_d"]) else None,
                                        on_c=np.mean([x["on_p"] for x in cn], 0) if len(c["on_d"]) else None)
    return data


def score(data, split, kind, fn):
    """fn(entry, panel) -> bool prediction. Returns {day set: {panel: mean IoU}}."""
    out = {ds: {} for ds in DAY_SETS}
    for panel in T2_PANELS:
        e = data[(panel, split)]
        c = e["c"]
        pre = "og" if kind == "ongoing" else "on"
        fut, efut, dd = c[f"{pre}_fut"], c[f"{pre}_efut"], c[f"{pre}_d"]
        if len(dd) == 0:
            continue
        s = iou(fn(e, panel), fut, efut)
        for ds, dmax in DAY_SETS.items():
            m = dd < dmax
            out[ds][panel] = s[m].mean() if m.any() else np.nan
    return out


# ---------------------------------------------------------------- ongoing
def og_fn(taus=None, fam_tau=None, logit=False, tau=0.60):
    def fn(e, panel):
        l, c = e["c"]["og_p"], e["og_c"]
        p = sg(0.5 * lg(l) + 0.5 * lg(c)) if logit else 0.5 * l + 0.5 * c
        if taus is not None:
            return p >= np.asarray(taus)[None, :, None]
        if fam_tau is not None:
            return p >= fam_tau[panel.rsplit("_", 1)[0]]
        return p >= tau
    return fn


def fit_og_h(data, split):
    taus = [0.60] * F
    for _ in range(2):
        for h in range(F):
            best = None
            for t in TAU_GRID:
                cand = taus.copy(); cand[h] = t
                s = fmean(score(data, split, "ongoing", og_fn(taus=cand))["all"])
                if best is None or s > best[0]:
                    best = (s, t)
            taus[h] = best[1]
    return taus


def fit_og_fam(data, split):
    fams = sorted({p.rsplit("_", 1)[0] for p in T2_PANELS})
    res = {t: score(data, split, "ongoing", og_fn(tau=t))["all"] for t in TAU_GRID}
    out = {}
    for f in fams:
        out[f] = max(TAU_GRID, key=lambda t: np.nanmean([v for p, v in res[t].items() if p.rsplit("_", 1)[0] == f]))
    return out


def fit_og_logit(data, split):
    return max(TAU_GRID, key=lambda t: fmean(score(data, split, "ongoing", og_fn(logit=True, tau=t))["all"]))


# ---------------------------------------------------------------- onset
def on_fn(w=0.7, a=1.0, b=0.0, cc=1.0, logit=False):
    def fn(e, panel):
        l, c = e["c"]["on_p"], e["on_c"]
        cz = a * lg(c) + b
        p = sg(w * lg(l) + (1 - w) * cz) if logit else w * l + (1 - w) * sg(cz)
        pr = np.zeros(e["c"]["on_fut"].shape, bool)
        pr[:, F - 1] = np.stack([select(q, "ratio", cc) for q in p])
        return pr
    return fn


ON_GRID = list(itertools.product((0.5, 0.6, 0.7, 0.8), (0.5, 0.75, 1.0, 1.25), (-1.0, -0.5, 0.0, 0.5), (0.9, 1.0, 1.1)))


def fit_on(data, split, logit):
    best = None
    for w, a, b, cc in ON_GRID:
        s = fmean(score(data, split, "onset", on_fn(w, a, b, cc, logit))["all"])
        if best is None or s > best[0]:
            best = (s, dict(w=w, a=a, b=b, cc=cc, logit=logit))
    return best[1]


def report(name, kind, data, split, fn, base):
    r = score(data, split, kind, fn)
    parts = []
    for ds in DAY_SETS:
        d, b0 = r[ds], base[ds]
        diffs = [d[p] - b0[p] for p in d if np.isfinite(d[p]) and np.isfinite(b0.get(p, np.nan))]
        parts.append(f"{ds} {fmean(d):.3f} ({fmean(d) - fmean(b0):+.3f}; {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"  {name:<34} " + "  ".join(parts), flush=True)


def main() -> None:
    data = load_all()
    for fit, ev in (("validation", "private"), ("private", "validation")):
        print(f"\n######## fit on {fit}, scored on {ev} ########")
        base_og = score(data, ev, "ongoing", og_fn())
        base_on = score(data, ev, "onset", on_fn())
        print("ongoing: baseline M " + "  ".join(f"{ds} {fmean(base_og[ds]):.3f}" for ds in DAY_SETS))
        taus = fit_og_h(data, fit)
        report(f"og_h taus={taus}", "ongoing", data, ev, og_fn(taus=taus), base_og)
        ft = fit_og_fam(data, fit)
        report(f"og_fam {ft}", "ongoing", data, ev, og_fn(fam_tau=ft), base_og)
        t = fit_og_logit(data, fit)
        report(f"og_logit tau={t}", "ongoing", data, ev, og_fn(logit=True, tau=t), base_og)
        print("onset: baseline M " + "  ".join(f"{ds} {fmean(base_on[ds]):.3f}" for ds in DAY_SETS))
        for logit in (False, True):
            prm = fit_on(data, fit, logit)
            report(f"on_{'logit' if logit else 'cal'} {prm}", "onset", data, ev, on_fn(**prm), base_on)


if __name__ == "__main__":
    main()
