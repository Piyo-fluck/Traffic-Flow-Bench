"""Note 169: does the overlap between the generator's active bottlenecks and a scored run's explain the per-panel
pretraining gain? Per panel and scored split (evaluation only): head day-activation shares in train truth, in the
scored run (observed layer), and in the generated runs; the L1 gap generated - scored and train - scored over the
heads active in any of them; against the ghb onset gain (sup+rec, s20/s21 mean, vs ctrl) from the Ada breakdown."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from dlsim_gen2 import corridor
from tfb.data import load
warnings.simplefilter("ignore")
# ghb - ctrl onset gains per panel (Ada gen_panel_breakdown2, sup+rec)
GAIN = {("D7_I10_W", "val"): 0.341 - 0.271, ("D7_I10_W", "pri"): 0.167 - 0.056,
        ("D7_I210_E", "val"): 0.0, ("D7_I210_E", "pri"): 0.0,
        ("D7_I210_W", "val"): 0.795 - 0.833, ("D7_I210_W", "pri"): 0.899 - 0.859,
        ("D7_I405_N", "val"): 0.929 - 0.929, ("D7_I405_N", "pri"): 0.882 - 0.920,
        ("D7_I405_S", "val"): 0.823 - 0.819, ("D7_I405_S", "pri"): 0.654 - 0.519,
        ("D12_I5_N", "val"): 0.901 - 0.919, ("D12_I5_N", "pri"): 0.905 - 0.926,
        ("D12_I5_S", "val"): 0.816 - 0.752, ("D12_I5_S", "pri"): 0.708 - 0.816}


def act(v, order, vf):
    v = v[:, :, order]; vfo = vf[order]
    cg = np.isfinite(v) & (v <= 0.6 * vfo); fr = np.isfinite(v) & (v > 0.9 * vfo)
    return ((cg[:, :, :-1] & fr[:, :, 1:]).sum(1) >= 4).mean(0)


rows = []
for panel in ["D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]:
    z = load(panel, "train")
    order, *_ = corridor(panel, z)
    vf = np.asarray(z["free_speed"], float)
    a_tr = act(z["true_speed"], order, vf)
    a_gen = np.mean([act(np.load(f)["true_speed"], order, vf) for f in sorted((ROOT / "work" / "genruns_ctm_hb").glob(f"{panel}_r*.npz"))], 0)
    for s, sk in (("validation", "val"), ("private", "pri")):
        a_sc = act(load(panel, s)["obs_speed"], order, vf)
        m = (a_tr >= 0.05) | (a_sc >= 0.05) | (a_gen >= 0.05)
        g_gap = np.abs(a_gen - a_sc)[m].sum(); t_gap = np.abs(a_tr - a_sc)[m].sum()
        off = ((a_gen >= 0.3) & (a_sc < 0.1)).sum()          # generated-active heads that the scored run does not use
        rows.append((panel, sk, GAIN[(panel, sk)], g_gap, t_gap, g_gap - t_gap, off))
        print(f"{panel:<10} {sk}: gain {GAIN[(panel, sk)]:+.3f} | L1 gap generated-scored {g_gap:.2f}, train-scored {t_gap:.2f} "
              f"(excess {g_gap - t_gap:+.2f}) | generated-active heads idle in the scored run {off}", flush=True)
r = np.array([x[2:] for x in rows], float)
print(f"\ncorr(gain, generated-scored gap) {np.corrcoef(r[:, 0], r[:, 1])[0, 1]:+.2f}; corr(gain, excess over train) "
      f"{np.corrcoef(r[:, 0], r[:, 3])[0, 1]:+.2f}; corr(gain, idle generated heads) {np.corrcoef(r[:, 0], r[:, 4])[0, 1]:+.2f} (n {len(r)})")
