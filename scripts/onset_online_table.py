"""note 169: run-adaptive onset member weights for the official windows. For every (split, panel, day index d) the
members' onset IoU on the lab onset windows of the same scored run on days d' < d (released data only, labels from the
released layer at those windows' T+30, all before the official window's day) gives Hedge weights
w_k ~ exp(eta * (sum IoU_k - max)). Writes work/onset_online_w.json: {"split|panel": [null | [w_H, w_ad9b, w_ps, w_ST, w_G], ...]}.

    .venv/Scripts/python.exe scripts/onset_online_table.py --eta 4
"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "occ_lab")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
CA = ROOT / "occ_lab" / "cache"
MEMS = ["H", "ad9b", "ps", "ST", "G"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=ROOT / "work" / "onset_online_w.json")
    a = ap.parse_args()
    table = {}
    for i, p in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, p)
        for s in C.SPLITS:
            on, ref = W[s]["on"], W[s]["ref"]
            nd = W[s]["z"]["obs_speed"].shape[0]
            if not len(on["d"]):
                table[f"{s}|{p}"] = [None] * nd
                continue
            g_tag, g_suf = ("ctrl", ("", "_rb")) if p == "D7_I405_N" else ("ct10d", ("_q1", "_q2"))
            mem = {"H": ref["on_H"], "ad9b": ref["on_ad9b"], "ps": np.load(CA / f"pseudo_s20_{s}_{p}.npz")["on"],
                   "ST": np.load(CA / f"onset_adapt_stid30_{s}_{p}.npz")["prob"],
                   "G": np.mean([np.load(CA / f"fam_leap_{g_tag}_s{k}{x}_sup+rec_{s}_{p}.npz")["on"] for k in (20, 21) for x in g_suf], 0)}
            iou = np.stack([C.onset_iou(mem[k], on) for k in MEMS])          # (5, windows)
            rows = []
            for d in range(nd):
                past = on["d"] < d
                if not past.any():
                    rows.append(None); continue
                sc = iou[:, past].sum(1)
                w = np.exp(a.eta * (sc - sc.max())); w /= w.sum()
                rows.append([float(x) for x in w])
            table[f"{s}|{p}"] = rows
            last = next((r for r in reversed(rows) if r is not None), None)
            print(f"{s:<10} {p:<10} lab windows {len(on['d'])}, last-day weights " + (" ".join(f"{k} {v:.2f}" for k, v in zip(MEMS, last)) if last else "none"), flush=True)
    a.out.write_text(json.dumps(table))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
