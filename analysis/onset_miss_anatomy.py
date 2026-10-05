"""Note 169 (evaluation): the onset windows whose location the ctrl model misses (sup+rec, s20/s21). For the true
T+30 queued links and for the model's arg-max link: how often each is a queue head (or queued at all) on train
days, in the generated runs (genruns_ctm_hb), and in that scored run's other days. Needs the ctrl caches copied
from Ada (work/leap_ctrl_on/)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import corridor
from tfb.data import load
warnings.simplefilter("ignore")


def qshare(v, vf):
    """per link: share of days on which the link is queued at least 3 slots in 06-20 h"""
    q = np.isfinite(v) & (v <= 0.6 * vf)
    return (q[:, 72:240].sum(1) >= 3).mean(0)


rows = []
for i, panel in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, panel)
    zt = load(panel, "train")
    vf = np.asarray(zt["free_speed"], float)
    s_tr = qshare(zt["true_speed"], vf)
    gens = sorted((ROOT / "work" / "genruns_ctm_hb").glob(f"{panel}_r*.npz"))
    s_gen = np.mean([qshare(np.load(f)["true_speed"], vf) for f in gens], 0) if gens else np.full(len(vf), np.nan)
    for s in C.SPLITS:
        on = W[s]["on"]
        zs = W[s]["z"]
        p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
        for n in range(len(p)):
            y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
            if not y.any():
                continue
            nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
            j = int(np.argmax(np.where(e, p[n], -1)))
            if nt[j]:
                continue
            d = int(on["d"][n])
            other = np.delete(np.arange(len(zs["dates"])), d)
            s_run = qshare(zs["obs_speed"][other], vf)
            tl = np.flatnonzero(y)
            rows.append((panel, s, s_tr[tl].max(), s_gen[tl].max(), s_run[tl].max(), s_tr[j], s_gen[j], p[n][tl].max() / p[n][j]))
print(f"location misses: {len(rows)}")
print("panel      split | true site: train / generated / scored-run other days | predicted site: train / generated | p(true)/p(pred)")
for r in rows:
    print(f"{r[0]:<10} {r[1][:3]}   | {r[2]:.2f} / {r[3]:.2f} / {r[4]:.2f} | {r[5]:.2f} / {r[6]:.2f} | {r[7]:.2f}")
r = np.array([x[2:] for x in rows], float)
print(f"\nmedian: true site train {np.median(r[:, 0]):.2f}, generated {np.nanmedian(r[:, 1]):.2f}, scored-run {np.median(r[:, 2]):.2f}; "
      f"predicted site train {np.median(r[:, 3]):.2f}; true sites queued on < 5 % of train days: {int((r[:, 0] < 0.05).sum())}, "
      f"of those queued on >= 5 % of generated days: {int(((r[:, 0] < 0.05) & (r[:, 1] >= 0.05)).sum())}")
