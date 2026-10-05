"""Note 169: (1) share of real first-onset sites within 3 links of the upstream / downstream corridor end (travel
order), and whether the downstream-most link is queued while its upstream neighbour is free (a queue entering from
outside the panel); (2) generated vs real mean flow 06-11 h at the sensored links (morning demand inflation)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import corridor
from onset_site_distribution import sites
from tfb.data import load
warnings.simplefilter("ignore")
for panel in C.T2_PANELS:
    z = load(panel, "train")
    order, *_ = corridor(panel, z)
    n = len(order)
    rank = np.full(len(z["link_ids"]), -1); rank[order] = np.arange(n)
    rs, _ = sites(z, order)
    rr = rank[rs]
    up_end, dn_end = (rr <= 2).mean(), (rr >= n - 3).mean()
    v = z["true_speed"][:, :, order]; vf = np.asarray(z["free_speed"], float)[order]
    cg = np.isfinite(v) & (v <= 0.6 * vf)
    spill = (cg[:, :, -1] & ~cg[:, :, -2]).sum() / max(cg[:, :, -1].sum(), 1)
    qr = np.nanmean(z["true_flow"][:, 72:132][:, :, order], (0, 1))
    gq = np.nanmean([np.nanmean(np.load(f)["true_flow"][:, 72:132][:, :, order], (0, 1)) for f in sorted((ROOT / "work" / __import__("os").environ.get("GEN_DIR", "genruns_ctm_hb")).glob(f"{panel}_r*.npz"))], 0)
    ratio = gq / np.maximum(qr, 1)
    print(f"{panel:<10} real onset sites within 3 links of the upstream end {up_end:.2f}, downstream end {dn_end:.2f} | "
          f"downstream-most link queued with its upstream neighbour free: {spill:.2f} of its queued slots | "
          f"morning flow gen / real median {np.nanmedian(ratio):.3f} [q10 {np.nanquantile(ratio, .1):.3f}, q90 {np.nanquantile(ratio, .9):.3f}]", flush=True)
