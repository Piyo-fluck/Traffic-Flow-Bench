"""Judge the DCRNN by error orthogonality, not by its own IoU (the user's criterion).

For the official-like ongoing windows it reports, per scored month:
    IoU of each member alone, and of the production blend
    cell-level error correlation between the DCRNN and each member, and the share of cells the three members get
      wrong while the DCRNN gets right (and the reverse)
    the same split by horizon (T+5..T+30) and by error type (extension of a visible queue / dissolution /
      persistence), because note 52 put the error mass on the extension at T+25 and T+30
    the blend with the DCRNN added at 0.10 / 0.15 / 0.20 (taken from the other members in proportion), and a
      horizon-gated variant that only uses it at T+25 and T+30

    python dcrnn_lab/eval_dcrnn.py --ckpt dcrnn_lab/models/dcrnn_s0.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["T2CNN_ARCH"] = "tfm"
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from dcrnn import DCRNN  # noqa: E402
from spatial_ssm import SpatialSSM  # noqa: E402

W_GS, W_TFM, W_CNN, TAU = 0.20, 0.60, 0.20, 0.50


@torch.no_grad()
def predict(net, x):
    out = []
    for a in range(0, len(x), 32):
        xb = C.to_dev(x[a:a + 32])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, C.F, x.shape[-1]), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", default="dcrnn")
    ap.add_argument("--arch", default="dcrnn", choices=("dcrnn", "ssm"))
    args = ap.parse_args()
    sd = torch.load(args.ckpt, map_location=C.DEV)
    if args.arch == "ssm":
        net = SpatialSSM(sd["embed.weight"].shape[1] // C.H, C.cnn.WIDTH,
                         sum(1 for k in sd if k.endswith(".to_a.weight"))).to(C.DEV)
    else:
        net = DCRNN(sd["enc.weight_ih"].shape[1], C.cnn.WIDTH).to(C.DEV)
    net.load_state_dict(sd)
    net.eval()
    acc = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = W[s]["og"], W[s]["ref"]
            if not len(og["d"]):
                continue
            p_dc = predict(net, C.make_x(ctx, stat, og, np.arange(len(og["d"])), W[s]["z"]))
            P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"], "dcrnn": p_dc}
            bl = W_GS * P["gs1"] + W_TFM * P["tfm8"] + W_CNN * P["cnn3"]
            y, e = og["fut"], og["efut"]
            vis_q = og["ok"][:, -1] & (np.nan_to_num(og["v"][:, -1], nan=1e9) <= 0.6 * ctx.vf)
            kind = np.where(np.broadcast_to(vis_q[:, None, :], y.shape), "persist", "grow")
            variants = {"blend (T)": bl, "dcrnn alone": P["dcrnn"], "gs1": P["gs1"], "tfm8": P["tfm8"],
                        "cnn3": P["cnn3"]}
            for w in (0.10, 0.15, 0.20):
                variants[f"blend + dcrnn {w:.2f}"] = (1 - w) * bl + w * P["dcrnn"]
            g = bl.copy()
            g[:, 4:] = 0.85 * bl[:, 4:] + 0.15 * P["dcrnn"][:, 4:]
            variants["blend + dcrnn 0.15 at T+25/30"] = g
            for n, p in variants.items():
                a = acc.setdefault(("iou", s, n), {})
                a[panel] = C.iou(p >= TAU, y, e).mean()
            # error structure
            err = {n: ((P[n] >= TAU) != y)[e] for n in P}
            err["blend"] = ((bl >= TAU) != y)[e]
            a = acc.setdefault(("err", s), dict(pairs={}, only_dc=0, only_rest=0, n=0, byh={}, bykind={}))
            for n in ("gs1", "tfm8", "cnn3", "blend"):
                a["pairs"].setdefault(n, []).append((err["dcrnn"] & err[n]).mean())
                a["pairs"].setdefault(n + "_corr", []).append(float(np.corrcoef(err["dcrnn"], err[n])[0, 1]))
            rest_wrong = err["gs1"] & err["tfm8"] & err["cnn3"]
            a["only_dc"] += int((rest_wrong & ~err["dcrnn"]).sum())
            a["only_rest"] += int((~rest_wrong & err["dcrnn"]).sum())
            a["n"] += int(e.sum())
            for h in range(C.F):
                m = e[:, h]
                a["byh"].setdefault(h, [0, 0, 0])
                a["byh"][h][0] += int((((bl[:, h] >= TAU) != y[:, h]) & m).sum())
                a["byh"][h][1] += int((((P["dcrnn"][:, h] >= TAU) != y[:, h]) & m).sum())
                a["byh"][h][2] += int(m.sum())
            for k in ("grow", "persist"):
                m = e & (kind == k)
                b = a["bykind"].setdefault(k, [0, 0, 0])
                b[0] += int((((bl >= TAU) != y) & m).sum())
                b[1] += int((((P["dcrnn"] >= TAU) != y) & m).sum())
                b[2] += int(m.sum())
        print(f"  {panel}", flush=True)

    for s in C.SPLITS:
        base = C.family_mean(acc[("iou", s, "blend (T)")])
        print(f"\n=== ongoing {s}: blend family {base:.4f} ===")
        for key, d in acc.items():
            if key[0] == "iou" and key[1] == s:
                n = key[2]
                print(f"   {n:<30} {C.family_mean(d):.4f} ({C.family_mean(d) - base:+.4f})")
        a = acc[("err", s)]
        print("   error overlap with dcrnn: " + "  ".join(
            f"{n} both {np.mean(a['pairs'][n]):.4f} corr {np.mean(a['pairs'][n + '_corr']):.2f}"
            for n in ("gs1", "tfm8", "cnn3", "blend")))
        print(f"   all three members wrong but dcrnn right: {a['only_dc'] / a['n'] * 100:.3f}% of cells; "
              f"dcrnn wrong where they are not: {a['only_rest'] / a['n'] * 100:.3f}%")
        print("   error rate by horizon (blend / dcrnn): " + "  ".join(
            f"T+{(h + 1) * 5} {v[0] / max(v[2], 1):.3f}/{v[1] / max(v[2], 1):.3f}" for h, v in sorted(a["byh"].items())))
        print("   error rate by cell type (blend / dcrnn): " + "  ".join(
            f"{k} {v[0] / max(v[2], 1):.3f}/{v[1] / max(v[2], 1):.3f} (n={v[2]:,})" for k, v in a["bykind"].items()))


if __name__ == "__main__":
    main()
