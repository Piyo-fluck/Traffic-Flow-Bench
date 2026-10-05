"""Note 169 (evaluation): is the race between habitual bottlenecks decidable from the history? All lab onset windows.
Candidates: the true site (the most downstream queued link at T+30) and a competitor (misses: the ctrl model's pick;
hits: the ctrl model's best link outside +-1 of the truth). Per candidate c, from the window's history only:
load = mean flow of the last 3 slots / effective capacity (the hidden-bottleneck capacity of note 169 when c is an
identified recurrent head, else the released capacity of the next sensored link), and growth = flow slope over the
last 6 slots / effective capacity. Share of windows where the true site wins on load / growth / both, misses vs hits."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
res = {"miss": [], "hit": []}
for i, panel in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, panel)
    zt = load(panel, "train")
    g = CTMGen(panel, zt)
    ceff = np.full(g.S, np.nan)
    for h in range(g.n - 1):                                     # capacity binding downstream of sensored link h
        ceff[g.order[h]] = g.cell_cap[g.sens_cell[h] + 1]
    ceff[g.order[-1]] = g.cell_cap[g.sens_cell[-1]]
    for s in C.SPLITS:
        on = W[s]["on"]
        p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
        for n in range(len(p)):
            y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
            if not y.any():
                continue
            nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
            j = int(np.argmax(np.where(e, p[n], -1)))
            kind = "hit" if nt[j] else "miss"
            if kind == "hit":
                pp = np.where(e & ~nt, p[n], -1)
                if pp.max() < 0:
                    continue
                j = int(np.argmax(pp))
            tl = np.flatnonzero(y)
            t = tl[np.argmax(g.order.tolist().index(x) if x in g.order else -1 for x in tl)] if False else tl[np.argmax([np.flatnonzero(g.order == x)[0] if (g.order == x).any() else -1 for x in tl])]
            q, ok = on["qf"][n], on["ok"][n]
            def feat(c):
                x = np.where(ok[:, c], q[:, c], np.nan)
                last = np.nanmean(x[-3:]); tt = np.arange(6); m = np.isfinite(x[-6:])
                slope = np.polyfit(tt[m], x[-6:][m], 1)[0] if m.sum() >= 3 else np.nan
                return last / ceff[c], slope / ceff[c]
            lt, gt = feat(t); lc, gc = feat(j)
            res[kind].append((lt > lc, gt > gc, (lt > lc) and (gt > gc), lt, lc))
for k, v in res.items():
    r = np.array(v, float)
    print(f"{k:<5} windows {len(r)}: true site wins on load {np.nanmean(r[:, 0]):.2f}, on growth {np.nanmean(r[:, 1]):.2f}, "
          f"on both {np.nanmean(r[:, 2]):.2f} | median load true {np.nanmedian(r[:, 3]):.2f} vs competitor {np.nanmedian(r[:, 4]):.2f}")
