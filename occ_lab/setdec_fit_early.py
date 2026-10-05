"""note 169: production fit of the onset set decoder under the official windows' condition. Base = H .35 + ps .175 +
G0 .175 + ST .30 (ad9b slot -> w192 ctrl), members adapted over K = 0..4 days (pseudo_adapt --hist K caches); the data
of both scored months and all five K are pooled. Stage 1: IoU-LambdaRank x6 seeds; stage 2 rows from each (K, month)'s
cross-month stage-1 output; dU regressors x6; threshold from 5-fold OOF (largest within 0.001 of the best gain).
Saves occ_lab/models/onset_rank_ge_s<k>.txt, onset_du_ge_s<k>.txt, onset_du_ge_tau.txt."""
import os
import sys
import warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
os.environ["ONSET_G"] = "G0"
os.environ["ONSET_W"] = "0.35,0,0.175,0.175,0.30"
import lightgbm as lgb
import numpy as np
import common as C
import onset_rerank as RR
import onset_stage2_du as S2
from onset_setdec_fit import SEEDS, fit_rank, apply_rank, rows_of, choose_tau
warnings.simplefilter("ignore")
MODELS = LAB / "models"
TAG = "_ge"


def main():
    datas = []
    for k in range(5):
        os.environ["ONSET_EARLY_K"] = str(k)
        datas.append(RR.prepare())
        print(f"prepared K {k}", flush=True)
    allD = [d[s] for data in datas for d in data.values() for s in C.SPLITS if s in d]
    for sd in SEEDS:
        fit_rank(allD, sd).save_model(str(MODELS / f"onset_rank{TAG}_s{sd}.txt"))
    Ts = []
    for data in datas:                      # stage-2 rows: each month's stage 1 fitted on the other month (same K)
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            rk = [fit_rank([d[s_fit] for d in data.values() if s_fit in d], sd) for sd in SEEDS]
            for d in data.values():
                if s_app in d:
                    apply_rank(rk, d[s_app])
        Ts += [S2.pairs(d[s]) for d in data.values() for s in C.SPLITS if s in d]
    X, y, win = rows_of(Ts)
    tau, gain = choose_tau(X, y, win)
    for sd in SEEDS:
        lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150).save_model(
            str(MODELS / f"onset_du{TAG}_s{sd}.txt"))
    (MODELS / f"onset_du{TAG}_tau.txt").write_text(f"{tau}\n")
    print(f"saved rankers + dU x{len(SEEDS)}; stage-2 rows {len(y)}; threshold {tau}; OOF gain "
          + ", ".join(f"{t}: {g:+.2f}" for t, g in gain.items()), flush=True)


if __name__ == "__main__":
    main()
