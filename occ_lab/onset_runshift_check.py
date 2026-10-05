"""Is H's advantage on the train tail a run effect? Compare H alone on the two surfaces.

`onset_hicov_eval.py` fixed the coverage exactly (0.764 vs the official 0.761) and still got candidate AN
backwards. The suspect is now the recurrence prior: on the train tail it is built from other days of the SAME
simulation run, on the scored months from a DIFFERENT run. If that is the mechanism, H alone must be far
stronger on the tail than on the scored months.

    python occ_lab/onset_runshift_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from onset_hicov_eval import score  # noqa: E402

CACHE = LAB / "cache"
WORK = LAB.parent / "work" / "cache"


def main() -> None:
    MEMBERS = ("H", "ad9b", "stid", "cnn3")
    tail, scored = {m: {} for m in MEMBERS}, {}
    for i, panel in enumerate(C.T2_PANELS):
        f = CACHE / f"onset_hicov_{panel}.npz"
        if f.exists():
            z = np.load(f)
            for m in MEMBERS:
                if m in z.files:
                    tail[m][panel] = score(z[m], z["y"], z["e"], 1.0).mean()
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on = W[s]["on"]
            if len(on["d"]):
                y = on["fut"][:, C.F - 1]
                e = on["efut"][:, C.F - 1]
                for m in MEMBERS:
                    k = {"H": "on_H", "ad9b": "on_ad9b", "cnn3": "on_cnn3"}.get(m)
                    P = W[s]["ref"].get(k) if k else None
                    if P is None:
                        continue
                    scored.setdefault((s, m), {})[panel] = score(P, y, e, 1.0).mean()
        ctx.z = None
        print(f"  {panel}", flush=True)
    print("\nEach member alone, the expected-IoU ratio rule at c = 1.0, family mean over panels")
    print(f"{'member':<8} {'train tail (same run)':>22} {'validation':>12} {'private':>10} {'drop':>9}")
    for m in MEMBERS:
        if not tail[m]:
            continue
        a = C.family_mean(tail[m])
        vs = [C.family_mean(scored[(s, m)]) for s in C.SPLITS if (s, m) in scored]
        if not vs:
            print(f"{m:<8} {a:>22.4f} {'-':>12} {'-':>10} {'-':>9}")
            continue
        print(f"{m:<8} {a:>22.4f} {vs[0]:>12.4f} {vs[1]:>10.4f} {float(np.mean(vs)) - a:>+9.4f}")


if __name__ == "__main__":
    main()
