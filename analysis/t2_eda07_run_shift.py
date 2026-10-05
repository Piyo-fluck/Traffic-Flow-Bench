"""Task 2 H17: are the validation/private months a different simulation run with different recurrent queues?

Build the queue-frequency map P_q(day type, slot, link) separately from:
  train      : train truth (all eligible cells)
  train-vis  : the train masked layer (visible eligible cells), which has the same visibility as below
  validation : the validation masked layer (visible eligible cells; Task 2 horizons are blanked anyway)
  private    : same for private
and compare them. The train-vis vs train gap shows what visibility alone does; the validation/private vs
train-vis gap shows the run shift. Metrics, over weekday cells between 05:00 and 20:00:
  - Pearson correlation of the maps;
  - for each hour and panel, the overlap of the top-5 links (Jaccard);
  - the share of each map's queue mass on links that train ranks in its top-10 for that hour.

    py analysis/t2_eda07_run_shift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from tfb.data import T, load  # noqa: E402

HOURS = range(5, 20)


def qmap(speed: np.ndarray, elig: np.ndarray, vf: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """(T, S) weekday queue frequency over visible eligible cells, smoothed over +-1 slot."""
    vis = elig & np.isfinite(speed)
    q = vis & (np.nan_to_num(speed, nan=1e9) <= 0.6 * vf)
    wd = weekday < 5
    num = q[wd].sum(0).astype(float)
    den = vis[wd].sum(0).astype(float)
    num = sum(np.roll(num, k, 0) for k in (-1, 0, 1))
    den = sum(np.roll(den, k, 0) for k in (-1, 0, 1))
    return num / np.maximum(den, 1)


def hourly(m: np.ndarray) -> np.ndarray:
    return m.reshape(24, 12, -1).mean(1)  # (24, S)


def main() -> None:
    print(f"{'panel':<11} {'corr train-vis':>14} {'corr val':>9} {'corr priv':>10} | {'top5 J train-vis':>16} {'val':>6} {'priv':>6} | "
          f"{'mass@top10 train':>16} {'val':>6} {'priv':>6} | {'queue share tr/va/pr':>21}")
    for panel in T2_PANELS:
        tr = load(panel, "train")
        vf = tr["free_speed"]
        maps = {"train": qmap(tr["true_speed"], tr["elig"], vf, tr["weekday"]),
                "train-vis": qmap(tr["obs_speed"], tr["elig"], vf, tr["weekday"])}
        for split in ("validation", "private"):
            z = load(panel, split)
            maps[split] = qmap(z["obs_speed"], z["elig"], vf, z["weekday"])
        sl = np.concatenate([np.arange(h * 12, h * 12 + 12) for h in HOURS])
        ref = maps["train"][sl].ravel()
        corr = {k: float(np.corrcoef(ref, maps[k][sl].ravel())[0, 1]) for k in ("train-vis", "validation", "private")}
        H_ = {k: hourly(v) for k, v in maps.items()}
        jac, mass = {}, {}
        for k in ("train-vis", "validation", "private"):
            js, ms = [], []
            for h in HOURS:
                a = set(np.argsort(-H_["train"][h])[:5]); b = set(np.argsort(-H_[k][h])[:5])
                if H_["train"][h].max() > 0.02 or H_[k][h].max() > 0.02:
                    js.append(len(a & b) / len(a | b))
                top10 = np.argsort(-H_["train"][h])[:10]
                tot = H_[k][h].sum()
                if tot > 0:
                    ms.append(H_[k][h][top10].sum() / tot)
            jac[k], mass[k] = np.mean(js) if js else np.nan, np.mean(ms) if ms else np.nan
        mass["train"] = np.mean([H_["train"][h][np.argsort(-H_["train"][h])[:10]].sum() / max(H_["train"][h].sum(), 1e-9) for h in HOURS])
        share = "/".join(f"{maps[k][sl].mean():.3f}" for k in ("train-vis", "validation", "private"))
        print(f"{panel:<11} {corr['train-vis']:>14.3f} {corr['validation']:>9.3f} {corr['private']:>10.3f} | "
              f"{jac['train-vis']:>16.3f} {jac['validation']:>6.3f} {jac['private']:>6.3f} | "
              f"{mass['train']:>16.3f} {mass['validation']:>6.3f} {mass['private']:>6.3f} | {share:>21}")


if __name__ == "__main__":
    main()
