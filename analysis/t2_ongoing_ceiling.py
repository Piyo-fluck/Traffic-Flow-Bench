"""Task 2 ONGOING: what is each error class worth if solved perfectly? (the ceiling per line of work)

Note 52 split the ongoing errors into grow / dissolve / stay / far; note 54 removed about 8% of the growth errors
and gained +0.004...+0.006 in M's blend. Before investing more feature work we need the ceiling of each bucket:
if perfect growth prediction is worth +0.05 ongoing (= +0.0075 total) then a +0.01 total target cannot come from
growth alone, and the plan has to change.

For M's blend (0.5 * LightGBM + 0.5 * CNN x3) at tau 0.50 and 0.60, per window, the prediction is corrected to the
truth on one class at a time:
    fix grow      cells not queued at T that become queued   (the note-54 target)
    fix dissolve  cells queued at T that clear
    fix stay      cells queued at T that stay queued
    fix far       cells never queued
    fix boundary k   every cell within k links of an edge of the queue visible at T (k = 1, 2, 4)
    fix step h    every cell of horizon step h (which step is worth most)
Each is an ORACLE: it says what the whole line of work is worth at most, not what is reachable.

    .venv/Scripts/python.exe analysis/t2_ongoing_ceiling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TAUS = (0.50, 0.60)
W_L = 0.5
boot = np.random.default_rng(41)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def edges_mask(base: np.ndarray, k: int) -> np.ndarray:
    """(N, S) cells within k links of a boundary of the queue visible at T."""
    e = np.zeros_like(base)
    e[:, :-1] |= base[:, :-1] != base[:, 1:]
    e[:, 1:] |= base[:, 1:] != base[:, :-1]
    out = e.copy()
    for j in range(1, k + 1):
        out[:, j:] |= e[:, :-j]
        out[:, :-j] |= e[:, j:]
    return out


def main() -> None:
    t0 = time.time()
    acc: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == og["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["og_p"] for t in TAGS], 0)
            pM = W_L * c["og_p"] + (1 - W_L) * cn
            truth, elig = og["fut"], og["efut"]
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            idx_last = np.where(og["ok"], np.arange(og["ok"].shape[1])[None, :, None], -1).max(1)
            base = np.take_along_axis(vis_q, np.clip(idx_last, 0, None)[:, None, :], 1)[:, 0] & (idx_last >= 0)
            b3 = np.broadcast_to(base[:, None], truth.shape)
            fixes = {"grow": ~b3 & truth, "dissolve": b3 & ~truth, "stay": b3 & truth, "far": ~b3 & ~truth}
            for k in (1, 2, 4):
                fixes[f"boundary +-{k}"] = np.broadcast_to(edges_mask(base, k)[:, None], truth.shape)
            for h in range(F):
                m = np.zeros(truth.shape, bool)
                m[:, h] = True
                fixes[f"step +{5 * (h + 1)}min"] = m
            for tau in TAUS:
                pred = pM >= tau
                r = acc.setdefault((split, tau), {})
                r.setdefault("M", []).append(iou(pred, truth, elig))
                for name, m in fixes.items():
                    fixed = np.where(m, truth, pred)
                    r.setdefault(name, []).append(iou(fixed, truth, elig))
        print(f"[{panel}] done ({time.time() - t0:.0f}s)", flush=True)
    for (split, tau), r in sorted(acc.items()):
        m = np.concatenate(r["M"])
        print(f"\n=== {split}, M blend at tau {tau:.2f}: ongoing IoU {m.mean():.4f} ({len(m)} windows) ===")
        print("   (delta x 0.5 = S_queue gain, x 0.15 = total-score gain)")
        for name in [k for k in r if k != "M"]:
            v = np.concatenate(r[name])
            d = v - m
            print(f"   fix {name:<16} IoU {v.mean():.4f}  delta {ci(d)}  -> total {0.15 * d.mean():+.4f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
