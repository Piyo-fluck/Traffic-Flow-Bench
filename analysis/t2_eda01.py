"""Task 2 EDA 01: window structure, local queue labels from train truth, and the persistence baseline.

Labels follow score_task2 / queue_utils: queued when speed <= v_cut = 0.6 x
links.csv free speed (fd_parameters carries no v_cut). The organizer computes
the label from the noise-free state; train only releases the noisy
observation, so the local label is an approximation near the threshold.

    py analysis/t2_eda01.py            # the 8 Task 2 panels
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, RELEASE, T, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

T2_PANELS = [p for p in PANELS if not p.startswith("D12_I405")]


def slot(ts: pd.Series) -> np.ndarray:
    s = ts.astype(str)
    return (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy()


def main() -> None:
    panels = sys.argv[1:] or T2_PANELS
    per_panel = {}
    rows = []
    for panel in panels:
        z = load(panel, "train")
        base = RELEASE / "task2" / panel / "train"
        W = pd.read_csv(base / "window_index.csv")
        H = pd.read_parquet(base / "window_history.parquet")
        Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
        links = pd.read_csv(RELEASE / "corridors" / panel / "network" / "links.csv", dtype={"link_id": str})
        fd = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv", dtype={"link_id": str})
        vf_links = links.set_index("link_id").free_speed_kmh.reindex(z["link_ids"]).to_numpy(float)
        vf_fd = fd.groupby("link_id").free_speed_kmh.mean().reindex(z["link_ids"]).to_numpy(float)
        vcut = 0.6 * vf_links
        s_of = {l: i for i, l in enumerate(z["link_ids"])}
        d_of = {d: i for i, d in enumerate(z["dates"])}
        ts, el, ov = z["true_speed"], z["elig"], z["obs_speed"]
        if panel == panels[0]:
            print("window_index columns:", list(W.columns))
            print("history columns:", list(H.columns))
        print(f"\n== {panel}: {len(W)} train windows, conditions {W.condition.value_counts().to_dict()}, "
              f"max|vf_links - vf_fd| {np.nanmax(np.abs(vf_links - vf_fd)):.3f} ==")
        hist_steps, fut_steps, blank_share, truth_share, hist_match = [], [], [], [], []
        for w in W.itertuples(index=False):
            d = d_of[str(w.date)]
            t0 = int(slot(pd.Series([w.forecast_origin]))[0])
            h = H[H.window_id == w.window_id]
            q = Q[Q.window_id == w.window_id]
            th, tq_ = slot(h.timestamp), slot(q.timestamp)
            hist_steps.append((th.min() - t0, th.max() - t0, len(np.unique(th))))
            fut_steps.append((tq_.min() - t0, tq_.max() - t0, len(np.unique(tq_))))
            hs = h.link_id.astype(str).map(s_of).to_numpy()
            hv = h.speed_kmh.to_numpy(float)
            tv = ts[d, th, hs]
            ok = np.isfinite(hv)
            hist_match.append(np.allclose(hv[ok], tv[ok]) if ok.any() else True)
            fut = np.arange(t0 + 1, min(t0 + 19, T))  # horizon + 60-min buffer
            blank_share.append(float(np.isnan(ov[d, fut]).mean()))
            truth_share.append(float(np.isfinite(ts[d, t0 + 1:t0 + 7]).mean()))
            # Local labels and persistence (the official baseline): queue at the last history step, carried forward.
            last = th.max()
            hl = h[th == last]
            qnow = np.zeros(len(s_of), bool)
            m_last = hl.is_score_eligible.astype(bool).to_numpy() & (hl.speed_kmh.to_numpy(float) <= vcut[hl.link_id.map(s_of).to_numpy()])
            qnow[hl.link_id.map(s_of).to_numpy()[m_last]] = True
            hz = np.arange(t0 + 1, t0 + 7)
            elig = el[d, hz] & np.isfinite(ts[d, hz])
            truth = (ts[d, hz] <= vcut) & elig
            pred = np.broadcast_to(qnow, truth.shape) & elig
            union = (pred | truth).sum()
            iou = 1.0 if union == 0 else (pred & truth).sum() / union
            q_hist = (h.speed_kmh.to_numpy(float) <= vcut[hs]) & h.is_score_eligible.astype(bool).to_numpy()
            rows.append(dict(panel=panel, window=w.window_id, cond=w.condition, tod=t0 / 12, iou=iou,
                             n_true=int(truth.sum()), n_pred=int(pred.sum()), n_elig=int(elig.sum()),
                             hist_queued=int(q_hist.sum()), q_at_T=int(qnow.sum()),
                             true_links_h6=int(truth[-1].sum())))
        print(f"history steps rel. to origin (min,max,n): {sorted(set(hist_steps))} | horizon: {sorted(set(fut_steps))}")
        print(f"masked layer blank in [T+5, T+90]: mean {np.mean(blank_share):.3f} | truth valued in horizon: "
              f"{np.mean(truth_share):.3f} | history equals unmasked truth: {all(hist_match)}")
        r = pd.DataFrame([x for x in rows if x["panel"] == panel])
        per_panel[panel] = r.groupby("cond").iou.mean().mean()
        print(r.groupby("cond")[["iou", "n_true", "n_pred", "hist_queued", "q_at_T", "true_links_h6", "tod"]].mean().round(2).to_string())
    allr = pd.DataFrame(rows)
    print(f"\npersistence on train windows (local labels): family mean S_queue {family_mean(per_panel):.4f}; "
          f"by condition: {allr.groupby('cond').iou.mean().round(4).to_dict()}")
    print("official references: persistence 0.2518 (repo README), 0.3017 (validation, BASELINES.md)")
    allr.to_csv(ROOT / "work" / "t2_eda01_windows.csv", index=False)


if __name__ == "__main__":
    main()
