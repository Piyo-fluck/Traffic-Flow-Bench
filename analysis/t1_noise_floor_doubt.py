"""Is Task 1's 'noise floor' (note 01 F4) actually noise?

Note 01 set the Task 1 ceiling at ~0.96 from the robust sigma of the second time difference of the truth,
asserting "the truth is the noisy observation". That assertion was never tested, and note 01's own F4 records a
fact that contradicts it: sigma does not change with pct_observed. Sampling noise must fall as 1/sqrt(n).

Three tests, all on the released train layers, which carry obs_* and true_* side by side:

  A  obs == true at visible cells?           note 103 proved this for Task 2; if it holds here there is no
                                             measurement noise anywhere in the release
  B  second-difference autocorrelation       i.i.d. noise of any size forces rho1 = -2/3 and rho2 = +1/6,
                                             whatever the underlying signal; real dynamics do not
  C  sigma against pct_observed              sampling noise scales as 1/sqrt(pct); signal does not

    .venv/Scripts/python.exe analysis/t1_noise_floor_doubt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import load  # noqa: E402

PANELS = ["D12_I5_N", "D7_I210_W", "D7_I10_E", "D7_I405_N"]


def rsig(e):
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def main() -> None:
    print("A. does the released observation equal the truth?")
    print(f"{'panel':<12} {'field':<6} {'cells':>12} {'max |obs-true|':>15} {'exact share':>12}")
    for p in PANELS:
        z = load(p, "train")
        for f in ("speed", "flow", "occ"):
            o, t = z[f"obs_{f}"], z[f"true_{f}"]
            m = np.isfinite(o) & np.isfinite(t)
            d = np.abs(o[m] - t[m])
            print(f"{p:<12} {f:<6} {m.sum():>12,} {d.max():>15.6f} {np.mean(d == 0):>12.6f}")
        del z

    print("\nB. second difference: i.i.d. noise forces rho1 = -0.667, rho2 = +0.167")
    print(f"{'panel':<12} {'field':<6} {'sigma_2nd':>10} {'rho1':>8} {'rho2':>8} {'verdict':>22}")
    for p in PANELS:
        z = load(p, "train")
        el = z["elig"]
        for f, s in (("speed", 1.0), ("flow", 1.0)):
            x = np.where(el, z[f"true_{f}"], np.nan).astype(np.float64)
            d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]          # (days, T-2, links)
            sg = rsig(d2.ravel()) / np.sqrt(6)                   # var(d2) = 6 sigma^2 under i.i.d. noise
            a = d2[:, :-2].ravel(); b = d2[:, 1:-1].ravel(); c = d2[:, 2:].ravel()
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            a, b, c = a[m] - a[m].mean(), b[m] - b[m].mean(), c[m] - c[m].mean()
            r1 = float(a @ b / np.sqrt((a @ a) * (b @ b)))
            r2 = float(a @ c / np.sqrt((a @ a) * (c @ c)))
            v = "i.i.d. noise" if abs(r1 + 2 / 3) < 0.05 and abs(r2 - 1 / 6) < 0.05 else "NOT noise: signal"
            print(f"{p:<12} {f:<6} {sg * s:>10.3f} {r1:>8.3f} {r2:>8.3f} {v:>22}")
        del z

    print("\nC. does the second-difference sigma fall as 1/sqrt(pct_observed)?")
    print(f"{'panel':<12} {'field':<6} " + " ".join(f"{b:>9}" for b in ("pct1-24", "25-49", "50-74", "75-99", "100")))
    for p in PANELS:
        z = load(p, "train")
        el, pc = z["elig"], z["pct"]
        for f in ("speed", "flow"):
            x = np.where(el, z[f"true_{f}"], np.nan).astype(np.float64)
            d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
            pm = pc[:, 1:-1]
            row = []
            for lo, hi in ((1, 25), (25, 50), (50, 75), (75, 100), (100, 101)):
                s = rsig(d2[(pm >= lo) & (pm < hi)]) / np.sqrt(6)
                row.append(f"{s:>9.3f}")
            print(f"{p:<12} {f:<6} " + " ".join(row))
        del z


if __name__ == "__main__":
    main()
