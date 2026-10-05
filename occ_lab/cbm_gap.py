"""note 171: where does onset still lose after BB's set decoder stage 1 (official condition, K = 0..4)?
Per window (production-like decoder: rankers fitted on the full-adaptation other month):
    got      IoU of the stage-1 decision
    oracle   max over candidates of U (the IoU of that candidate's ratio-rule region) - a perfect candidate choice
    loss     choice = oracle - got;  shape = 1 - oracle (extent / multi-site / missed site)
and the shape loss split by: several true sites (sites_n > 1), the best candidate's region misses the true cells
entirely, else extent (right place, wrong size).

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_gap.py
"""
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402
from onset_setdec_fit import SEEDS, fit_rank, apply_rank  # noqa: E402
from setdec_early_check import prep  # noqa: E402

warnings.simplefilter("ignore")


def main():
    full = prep(None)
    rk = {s: [fit_rank([d[s] for d in full.values() if s in d], sd) for sd in SEEDS] for s in C.SPLITS}
    acc = {}
    for k in range(5):
        early = prep(k)
        for s_app in C.SPLITS:
            s_fit = [x for x in C.SPLITS if x != s_app][0]
            for p_, d in early.items():
                if s_app not in d:
                    continue
                D = d[s_app]; on = D["on"]
                apply_rank(rk[s_fit], D)
                got = C.onset_iou(D["P1"], on)
                off = 0
                for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                    U = D["U"][off:off + g]; off += g
                    t = on["fut"][n, -1] & on["efut"][n, -1]
                    orc = max(float(U.max()), got[n])
                    multi = RR.sites_n(t) > 1
                    miss = U.max() == 0
                    kind = "multi" if multi else ("miss" if miss else "extent")
                    a = acc.setdefault((s_app, p_), dict(n=0, got=0, choice=0, multi=0, miss=0, extent=0))
                    a["n"] += 1; a["got"] += got[n]; a["choice"] += orc - got[n]; a[kind] += 1 - orc
    for s in C.SPLITS:
        tot = {k: sum(a[k] for (s_, _), a in acc.items() if s_ == s) for k in ("n", "got", "choice", "multi", "miss", "extent")}
        n = tot["n"]
        print(f"\n{s}: IoU {tot['got'] / n:.4f}; loss per window: choice {tot['choice'] / n:.4f}, "
              f"multi-site {tot['multi'] / n:.4f}, missed site {tot['miss'] / n:.4f}, extent {tot['extent'] / n:.4f}")
        for (s_, p_), a in acc.items():
            if s_ == s:
                m = a["n"]
                print(f"  {p_:<11} n {m // 5:>3}  IoU {a['got'] / m:.3f}  choice {a['choice'] / m:.3f}  multi "
                      f"{a['multi'] / m:.3f}  miss {a['miss'] / m:.3f}  extent {a['extent'] / m:.3f}")


if __name__ == "__main__":
    main()
