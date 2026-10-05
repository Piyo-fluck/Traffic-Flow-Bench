"""Average the '== onset' lines of several analysis/t2_onset_hazard.py repeats (different seeds).

For each mode (alone / blend), split and day set: the mean over repeats of each variant's family-mean IoU and of its
change against the paired base, the min and max of that change, and how many repeats it was positive in.

    .venv/Scripts/python.exe analysis/t2_hazard_agg.py work/t2_onset_hazard_s0.log work/t2_onset_hazard_s1.log ...
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np

LINE = re.compile(r"^== onset (\w+) (\w+) (\S+): (.*)$")
CELL = re.compile(r"([\w+]+) ([0-9.]+) \(([-+][0-9.]+);")


def main() -> None:
    acc: dict = defaultdict(lambda: defaultdict(list))
    zero: dict = defaultdict(list)
    for path in sys.argv[1:]:
        for raw in open(path, encoding="utf-8", errors="ignore"):
            raw = raw.strip()
            m = LINE.match(raw)
            if m:
                mode, split, ds, rest = m.groups()
                for var, val, delta in CELL.findall(rest):
                    acc[(mode, split, ds)][var].append((float(val), float(delta)))
            z = re.match(r"IoU=0 share (\w+) ([\w+]+)\s+(\w+): ([0-9.]+)", raw)
            if z:
                zero[z.groups()[:3]].append(float(z.group(4)))
    n = len(sys.argv) - 1
    print(f"{n} repeats")
    for key, vs in acc.items():
        parts = []
        for var, xs in vs.items():
            a = np.array(xs)
            parts.append(f"{var} {a[:, 0].mean():.3f} (d {a[:, 1].mean():+.3f} [{a[:, 1].min():+.3f},{a[:, 1].max():+.3f}] "
                         f"{int((a[:, 1] > 1e-9).sum())}/{len(a)})")
        print(f"== {key[0]} {key[1]} {key[2]}: " + "  ".join(parts))
    for key, xs in sorted(zero.items()):
        print(f"   IoU=0 share {key[0]} {key[1]:<10} {key[2]}: {np.mean(xs):.3f}")


if __name__ == "__main__":
    main()
