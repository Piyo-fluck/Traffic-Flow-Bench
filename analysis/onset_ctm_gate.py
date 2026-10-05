"""Note 169 (evaluation): when is the forward-CTM race verdict reliable? Every lab onset window with a competitor
(misses: ctrl's pick; hits: ctrl's best link outside +-1 of the truth). CTM margin = score(true) - score(comp)
signed as a verdict; |margin| bins -> accuracy of the CTM verdict. CNN margin = p(pick) - p(second site). Gate: follow
CTM when |CTM margin| > a and the CNN margin < b; count fixed / broken windows over all 331."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
warnings.simplefilter("ignore")
rows = []
for i, panel in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, panel)
    for s in C.SPLITS:
        on = W[s]["on"]
        SC = np.load(ROOT / "occ_lab" / "cache" / f"ctmfwd_{s}_{panel}.npz")["sc"]
        p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
        for n in range(len(p)):
            y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
            if not y.any():
                continue
            nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
            j = int(np.argmax(np.where(e, p[n], -1)))
            hit = bool(nt[j])
            # the CNN's best site outside +-1 of its pick = the alternative it would switch to
            far = np.ones(len(e), bool); far[max(0, j - 1):j + 2] = False
            alt = int(np.argmax(np.where(e & far, p[n], -1)))
            nb = lambda f, c: f[max(0, c - 1):c + 2].max()  # noqa: E731
            m_ctm = nb(SC[n], alt) - nb(SC[n], j)                     # > 0: CTM prefers the alternative
            m_cnn = p[n][j] - p[n][alt]
            rows.append((hit, bool(nt[alt]), m_ctm, m_cnn))
r = np.array(rows, float)
print(f"windows {len(r)}, CNN top-1 {r[:, 0].mean():.4f}")
for a in (0.0, 0.1, 0.3, 0.5, 1.0):
    for b in (0.1, 0.3, 1.0):
        sw = (r[:, 2] > a) & (r[:, 3] < b)
        fixed = int((sw & (r[:, 0] == 0) & (r[:, 1] == 1)).sum()); broken = int((sw & (r[:, 0] == 1) & (r[:, 1] == 0)).sum())
        print(f"  switch to the CTM's alternative if CTM margin > {a:.1f} and CNN margin < {b:.1f}: switches {int(sw.sum()):3d}, fixed {fixed}, broken {broken}, net {fixed - broken:+d}")
