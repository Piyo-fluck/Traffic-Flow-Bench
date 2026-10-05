"""note 171: MixCEM-style gate. Use the CBM's second opinion (cbm_setdec_check: swap after the set decoder's stage 1)
only in windows that look OUT of the training run's distribution; elsewhere keep BB's set decoder.

Per window two decisions (production-like set decoder, early K = 0..4, cross-month): setdec and setdec+CBM. The gate
takes setdec+CBM when an out-of-distribution score passes a threshold. Scores, from data released at or before T:

    dev_lvl   median over links of log(q_now / q_train(tod)): today's demand level vs the training run's profile
    dev_abs   median |log(q_now / q_train(tod))|
    dev_mad   spread (MAD) of the log ratios over links: today's spatial demand pattern vs the training run's
    v_dev     median (v_now / v_train(tod)) - 1, speeds
    cb_prod   the CBM probability at production's top peak (low = the CBM disagrees with the location habit)
    cb_max    the CBM's own maximum (its confidence anywhere)

The threshold (and its direction) is chosen on one month and applied to the other.

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_gate.py
"""
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from setdec_early_check import prep  # noqa: E402
import os  # noqa: E402
import cbm_setdec_check as CS  # noqa: E402
from cbm_setdec_check import decode, cbm_prob  # noqa: E402
from tfb.data import load  # noqa: E402

warnings.simplefilter("ignore")
NAMES = ("dev_lvl", "dev_abs", "dev_mad", "v_dev", "cb_prod", "cb_max")


def train_profile(panel):
    z = load(panel, "train")
    vf = np.asarray(z["free_speed"], np.float32)
    ok = z["elig"] & np.isfinite(z["obs_flow"]) & np.isfinite(z["obs_speed"])
    q = np.where(ok, z["obs_flow"], np.nan); v = np.where(ok, z["obs_speed"] / vf, np.nan)
    wk = z["weekday"] >= 5
    with np.errstate(all="ignore"):
        return {w: (np.nanmedian(q[wk == w], 0), np.nanmedian(v[wk == w], 0)) for w in (False, True)}


def scores(data_app, s_app):
    out = []
    for p_, d in data_app.items():
        if s_app not in d:
            continue
        D = d[s_app]; on = D["on"]
        prof = train_profile(p_)
        z = load(p_, s_app)
        cb = cbm_prob(s_app, p_)
        vf = np.asarray(z["free_speed"], np.float32)
        top = {n: cand[int(np.argmax(mass))] for n, cand, reg, mass in D["meta"]}
        for n in range(len(on["d"])):
            t0 = int(on["t0"][n]); wk = bool(z["weekday"][on["d"][n]] >= 5)
            qp, vp = prof[wk]
            ok = on["ok"][n][-3:]
            with np.errstate(all="ignore"):
                qn = np.nanmean(np.where(ok, on["qf"][n][-3:], np.nan), 0)
                vn = np.nanmean(np.where(ok, on["v"][n][-3:] / vf, np.nan), 0)
                qr = np.nanmean(qp[t0 - 3:t0], 0); vr = np.nanmean(vp[t0 - 3:t0], 0)
                lr = np.log(np.maximum(qn, 1) / np.maximum(qr, 1))
                lr = lr[np.isfinite(qn) & np.isfinite(qr) & (qr > 50)]
                vv = (vn / vr - 1)[np.isfinite(vn) & np.isfinite(vr)]
            j = top.get(n, int(np.argmax(D["base"][n])))
            out.append([np.median(lr) if len(lr) else 0.0, np.median(np.abs(lr)) if len(lr) else 0.0,
                        np.median(np.abs(lr - np.median(lr))) if len(lr) else 0.0,
                        np.median(vv) if len(vv) else 0.0,
                        cb[n][max(0, j - 2):j + 3].max(), cb[n].max()])
    return np.asarray(out, np.float32)


def main():
    lams = [float(x) for x in os.environ.get("GATE_LAMS", "1").split(",")]
    full = prep(None)
    R = {s_: {"sd": [], "sco": [], **{f"sc{l}": [] for l in lams}} for s_ in C.SPLITS}
    for k in range(5):
        early = prep(k)
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            _, sd, _ = decode(full, s_fit, early, s_app, False)
            R[s_app]["sd"].append(sd)
            for l in lams:
                CS.LAM = l
                _, sc, _ = decode(full, s_fit, early, s_app, True)
                R[s_app][f"sc{l}"].append(sc)
            R[s_app]["sco"].append(scores(early, s_app))
        print(f"  K={k}", flush=True)
    for s_ in C.SPLITS:
        for key in R[s_]:
            R[s_][key] = np.stack(R[s_][key])
    np.savez_compressed(LAB / "cache" / "cbm_gate_rows.npz",
                        **{f"{s_}_{k}": v for s_ in C.SPLITS for k, v in R[s_].items()})

    def gain(s_, f, th, sign, lam):
        g = np.where(sign * (R[s_]["sco"][:, :, f] - th) > 0, R[s_][f"sc{lam}"] - R[s_]["sd"], 0.0)
        return g.mean(0)
    for lam in lams:
        print(f"always CBM lambda {lam}: " + "  ".join(f"{s_} {np.mean(R[s_][f'sc{lam}'] - R[s_]['sd']):+.4f}"
                                                  for s_ in C.SPLITS))
    for f, nm in enumerate(NAMES):
        for lam in lams:
            line = []
            for s_sel, s_app in (("validation", "private"), ("private", "validation")):
                vals = R[s_sel]["sco"][:, :, f].ravel()
                best = (0.0, None, None)
                for q in np.linspace(0.05, 0.95, 19):
                    th = np.quantile(vals, q)
                    for sign in (1, -1):
                        gm = gain(s_sel, f, th, sign, lam).mean()
                        if gm > best[0] + 1e-9:
                            best = (gm, th, sign)
                if best[1] is None:
                    line.append(f"sel {s_sel[:3]}: nothing > 0"); continue
                ga = gain(s_app, f, best[1], best[2], lam)
                fired = (best[2] * (R[s_app]["sco"][:, :, f] - best[1]) > 0).mean()
                line.append(f"sel {s_sel[:3]} ({best[0]:+.4f}, {'>' if best[2] > 0 else '<'}{best[1]:.3f}) -> "
                            f"{s_app[:3]} {ga.mean():+.4f} (SE {ga.std(ddof=1) / np.sqrt(len(ga)):.4f}, on {fired:.0%})")
            print(f"  {nm:<8} lam {lam:<4} " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
