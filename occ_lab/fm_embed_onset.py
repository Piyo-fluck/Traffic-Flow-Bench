"""Idea (4): frozen time-series foundation models as FEATURE EXTRACTORS for the onset candidates (not forecasters).

Per candidate bottleneck of every lab onset window (note 129's peaks, the rows of note 152's ranker): the
candidate link's released series over the L = 512 slots before T (the previous day included), slot T excluded.
The models only embed, weights frozen:

    chronos2   amazon/chronos-2 (Apache-2.0), two variates (v/vf, q/cap), encoder output averaged over the patches
    moment     AutonLab/MOMENT-1-large (MIT), v/vf with the missing slots masked, the embedding head
Saved to occ_lab/cache/fmemb_<model>_<split>_<panel>.npz, rows in the order of bottleneck_cls.build's candidate rows.

    .venv/Scripts/python.exe occ_lab/fm_embed_onset.py --model chronos2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
os.environ.setdefault("HF_HOME", str(LAB.parent / ".hf_cache"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402

L = 512
CACHE = LAB / "cache"


def series(z, vf, cap, d, t0, j):
    """(2, L) v/vf and q/cap of link j over the L slots before t0 (nan where unobserved)."""
    T = z["obs_speed"].shape[1]
    v = np.concatenate([z["obs_speed"][d - 1, :, j] if d > 0 else np.full(T, np.nan), z["obs_speed"][d, :t0, j]])
    q = np.concatenate([z["obs_flow"][d - 1, :, j] if d > 0 else np.full(T, np.nan), z["obs_flow"][d, :t0, j]])
    ok = np.concatenate([z["elig"][d - 1, :, j] if d > 0 else np.zeros(T, bool), np.ones(t0, bool)])
    v = np.where(np.isfinite(v), v, np.nan)[-L:]
    q = np.where(np.isfinite(q), q, np.nan)[-L:]
    out = np.stack([v / vf[j], q / cap[j]]).astype(np.float32)
    if out.shape[1] < L:
        out = np.concatenate([np.full((2, L - out.shape[1]), np.nan, np.float32), out], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("chronos2", "moment", "ttm", "timesfm"))
    ap.add_argument("--pool", default="mean", choices=("mean", "last"),
                    help="chronos2: average over the patches, or the last 4 patch tokens (the state at T)")
    args = ap.parse_args()
    t_start = time.time()
    if args.model == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

        def embed(S):                                           # S (N, 2, L) with nans
            embs, _ = pipe.embed(torch.from_numpy(S), batch_size=256, context_length=L)
            if args.pool == "last":                             # tokens: [REG?] patches ... [special] -> the 4 before the last
                return np.stack([e.float()[:, -5:-1].mean(1).reshape(-1).cpu().numpy() for e in embs])
            return np.stack([e.float().mean(1).reshape(-1).cpu().numpy() for e in embs])
    elif args.model == "timesfm":
        import timesfm
        tf = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        mod = tf.model.to("cuda").eval()
        cap_ = {}
        mod.stacked_xf[-1].register_forward_hook(lambda m, i, o: cap_.__setitem__("h", o[0] if isinstance(o, tuple) else o))

        def fill1(x):
            t = np.arange(len(x)); ok = np.isfinite(x)
            return np.interp(t, t[ok], x[ok]) if ok.any() else np.ones_like(x)

        def embed(S):
            X = np.stack([fill1(r) for r in S[:, 0]]).astype(np.float32)          # v/vf, (N, L)
            out = []
            with torch.no_grad():
                for a in range(0, len(X), 64):
                    xb = torch.from_numpy(X[a:a + 64]).cuda()
                    mod.decode(32, xb, torch.zeros_like(xb, dtype=torch.bool))
                    out.append(cap_["h"].float().mean(1).cpu().numpy())
            return np.concatenate(out)
    elif args.model == "ttm":
        from tsfm_public import TinyTimeMixerForPrediction
        ttm = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2").to("cuda").eval()

        def fill(x):                                             # (N, C, L): linear in time, edges nearest, all-nan -> 1
            out = x.copy()
            t = np.arange(x.shape[-1])
            for n in range(x.shape[0]):
                for c in range(x.shape[1]):
                    v = x[n, c]; ok = np.isfinite(v)
                    out[n, c] = np.interp(t, t[ok], v[ok]) if ok.any() else 1.0
            return out

        def embed(S):
            X = torch.from_numpy(fill(S).transpose(0, 2, 1).copy()).float()      # (N, L, C)
            out = []
            with torch.no_grad():
                for a in range(0, len(X), 128):
                    o = ttm(past_values=X[a:a + 128].cuda(), output_hidden_states=True)
                    h = o.backbone_hidden_state                               # (B, C, patches, d)
                    out.append(h.float().mean(2).reshape(h.shape[0], -1).cpu().numpy())
            return np.concatenate(out)
    else:
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"})
        model.init()
        model = model.to("cuda").eval()

        def embed(S):
            x = S[:, :1]
            m = np.isfinite(x[:, 0]).astype(np.float32)
            xs = np.nan_to_num(x)
            out = []
            with torch.no_grad():
                for a in range(0, len(xs), 64):
                    o = model(x_enc=torch.from_numpy(xs[a:a + 64]).cuda(), input_mask=torch.from_numpy(m[a:a + 64]).cuda())
                    out.append(o.embeddings.float().cpu().numpy())
            return np.concatenate(out)

    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        for s, D in B.items():
            z, on = W[s]["z"], D["on"]
            S = np.stack([series(z, vf, cap, int(on["d"][n]), int(on["t0"][n]), int(j))
                          for n, cand, reg, mass in D["meta"] for j in cand])
            assert len(S) == len(D["X"])
            E = embed(S)
            tag = args.model + ("" if args.pool == "mean" else "last")
            np.savez(CACHE / f"fmemb_{tag}_{s}_{panel}.npz", emb=E.astype(np.float16))
            print(f"  {panel} {s}: {len(S)} candidates -> {E.shape} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
