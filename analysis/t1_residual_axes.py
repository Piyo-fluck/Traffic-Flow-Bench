"""Where, if anywhere, does the Task 1 residual repeat itself? Three axes, each with its own control.

15.7% of the error variance is the part of the truth's own high-frequency component the model failed to capture
(it captures about 73% of it, and tracks the smooth part perfectly, corr +0.01). The other 84% is not a function
of that cell's local time profile at all, and has not been identified.

This correlates the residual with itself along three axes without assuming which one should work:

    space    same day and slot, link offset 1, 2, 3, 5, 10, 20
    time     same day and link, slot offset 1, 2, 3, 5
    days     same link and slot, day offset 1, 7, 14

Each against a control that breaks the pairing it is testing: for space and time, the partner's value taken from
a day seven days away; for days, a partner slot two hours away. Free flow and congestion reported apart, since
they behave differently and free flow is 96% of the cells.

    .venv/Scripts/python.exe analysis/t1_residual_axes.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"


class Acc:
    """Pairwise-complete correlation accumulated over panels."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        self.n += x.size
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    def rho(self):
        if self.n < 1000:
            return np.nan, 0
        n = self.n
        cxy = self.sxy - self.sx * self.sy / n
        cxx = self.sxx - self.sx ** 2 / n
        cyy = self.syy - self.sy ** 2 / n
        return (float(cxy / np.sqrt(cxx * cyy)) if cxx > 0 and cyy > 0 else np.nan), int(n)


def main() -> None:
    keys = ([("space", k) for k in (1, 2, 3, 5, 10, 20)]
            + [("time", k) for k in (1, 2, 3, 5)]
            + [("days", k) for k in (1, 7, 14)])
    A = {(kind, k, band, what): Acc() for kind, k in keys for band in ("free", "cong") for what in ("real", "ctrl")}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        d = p["days"]
        vf = np.asarray(z["free_speed"], np.float64)
        ts, tq = z["true_speed"][d].astype(np.float64), z["true_flow"][d].astype(np.float64)
        tgt = z["target"][d] & np.isfinite(ts) & np.isfinite(tq)
        k_t = tq / np.maximum(ts, 1.0)
        k_p = p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0)
        e = np.where(tgt, k_p - k_t, np.nan)
        cong = np.nan_to_num(ts, nan=1e9) < 0.6 * vf[None, None, :]
        for kind, k in keys:
            if kind == "space":
                x, y = e[:, :, :-k], e[:, :, k:]
                c = cong[:, :, :-k]
                xc = np.roll(e, 7, axis=0)[:, :, :-k]
            elif kind == "time":
                x, y = e[:, :-k], e[:, k:]
                c = cong[:, :-k]
                xc = np.roll(e, 7, axis=0)[:, :-k]
            else:
                x, y = e[:-k], e[k:]
                c = cong[:-k]
                xc = np.roll(e, 24, axis=1)[:-k]          # a partner two hours away, same day
            for band, mask in (("free", ~c), ("cong", c)):
                A[(kind, k, band, "real")].add(np.where(mask, x, np.nan), y)
                A[(kind, k, band, "ctrl")].add(np.where(mask, xc, np.nan), y)
        print(f"  {panel}", flush=True)
        del z

    print(f"\ncorrelation of the residual with itself, pairwise-complete\n")
    print(f"{'axis':<8}{'offset':>7}{'free flow':>12}{'control':>10}{'n':>14}   |{'congested':>11}{'control':>10}{'n':>12}")
    last = None
    for kind, k in keys:
        if kind != last:
            print()
            last = kind
        rf, nf = A[(kind, k, "free", "real")].rho()
        cf, _ = A[(kind, k, "free", "ctrl")].rho()
        rc, nc = A[(kind, k, "cong", "real")].rho()
        cc, _ = A[(kind, k, "cong", "ctrl")].rho()
        unit = {"space": "links", "time": "slots", "days": "days"}[kind]
        print(f"{kind:<8}{k:>4} {unit:<3}{rf:>+11.4f}{cf:>+10.4f}{nf:>14,}   |{rc:>+11.4f}{cc:>+10.4f}{nc:>12,}")


if __name__ == "__main__":
    main()
