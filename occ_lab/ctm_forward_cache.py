"""Note 169: cache the forward-CTM onset scores (analysis/onset_ctm_forward.forward) for every lab onset window:
occ_lab/cache/ctmfwd_<split>_<panel>.npz, sc (N_on, S) in common.windows order."""
import sys
import warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from ctm_gen import CTMGen
from onset_ctm_forward import forward
from tfb.data import load
warnings.simplefilter("ignore")
for i, panel in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, panel)
    g = CTMGen(panel, load(panel, "train"))
    g.fit_profiles(g.z if hasattr(g, "z") else load(panel, "train"))
    for s in C.SPLITS:
        on, z = W[s]["on"], W[s]["z"]
        sc = np.stack([forward(g, z, int(on["d"][n]), int(on["t0"][n])) for n in range(len(on["d"]))]) if len(on["d"]) else np.zeros((0, g.S))
        np.savez(LAB / "cache" / f"ctmfwd_{s}_{panel}.npz", sc=sc.astype(np.float32))
        print(f"  {panel} {s}: {len(sc)} windows", flush=True)
