"""A new onset member added to the production onset blend at small weights, with rescue / spoil counts.

Production onset (AP onwards): p = 0.35 H + 0.175 ad9b + 0.175 p_PS + 0.30 STID, expected-IoU ratio rule at T+30.
A new member X enters as (1 - w) p + w X for w in 0.05 / 0.10 / 0.15. Reported per month: the family IoU change,
and the windows where the production blend scores 0 and the new one does not (rescue) and the reverse (spoil).

    .venv/Scripts/python.exe occ_lab/onset_new_eval.py --tags mult pm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
WS = (0.05, 0.10, 0.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    fam, counts = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        _, _, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = W[s]["on"], W[s]["ref"]
            if not len(on["d"]):
                continue
            ps = np.load(CACHE / f"pseudo_s20_{s}_{panel}.npz")["on"].astype(np.float32)
            st = np.load(CACHE / f"onset_adapt_stid30_{s}_{panel}.npz")["prob"].astype(np.float32)
            base = 0.35 * ref["on_H"] + 0.175 * ref["on_ad9b"] + 0.175 * ps + 0.30 * st
            b_sc = C.onset_iou(base, on)
            fam.setdefault((s, "production"), {})[panel] = b_sc.mean()
            for tag in args.tags:
                f = CACHE / f"onset_adapt_{tag}_{s}_{panel}.npz"
                if not f.exists():
                    continue
                X = np.load(f)["prob"].astype(np.float32)
                fam.setdefault((s, f"{tag} alone"), {})[panel] = C.onset_iou(X, on).mean()
                for w in WS:
                    sc = C.onset_iou((1 - w) * base + w * X, on)
                    nm = f"{tag} w{w}"
                    fam.setdefault((s, nm), {})[panel] = sc.mean()
                    c = counts.setdefault((s, nm), [0, 0, 0])
                    c[0] += int(((b_sc == 0) & (sc > 0)).sum())
                    c[1] += int(((b_sc > 0) & (sc == 0)).sum())
                    c[2] += len(sc)
        print(f"  {panel}", flush=True)
    names = ["production"] + [f"{t} alone" for t in args.tags] + [f"{t} w{w}" for t in args.tags for w in WS]
    ref = {s: C.family_mean(fam[(s, "production")]) for s in C.SPLITS}
    print(f"\n{'onset member':<16}" + "".join(f"{s:>22}" for s in C.SPLITS) + "   rescue / spoil (both months)")
    for nm in names:
        if not all((s, nm) in fam for s in C.SPLITS):
            continue
        cells = "".join(f"{C.family_mean(fam[(s, nm)]):>12.4f} ({C.family_mean(fam[(s, nm)]) - ref[s]:+.4f})"
                        for s in C.SPLITS)
        rs = sum(counts.get((s, nm), [0, 0, 0])[0] for s in C.SPLITS)
        sp = sum(counts.get((s, nm), [0, 0, 0])[1] for s in C.SPLITS)
        print(f"{nm:<16}{cells}   {rs} / {sp}" if "w0" in nm else f"{nm:<16}{cells}")


if __name__ == "__main__":
    main()
