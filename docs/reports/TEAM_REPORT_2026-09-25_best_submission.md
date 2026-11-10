# 現在の最高スコア提出（LB 0.88405）の全体説明

提出物: `work/sub_AS_sslens5/submission.csv`。**Task 2 の詳細は別文書
[`TEAM_REPORT_2026-09-25_task2_method.md`](TEAM_REPORT_2026-09-25_task2_method.md) にあります**ので、ここでは
Task 1・Task 3・Task 4 と、提出の組み立て、スコアの構成を扱います。

---

## 1. スコアの構成（公式定義）

出典: `trafficflowbench-public/docs/SCORING_SPEC.md`、`docs/BASELINES.md`。

```text
S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
```

| | naive baseline | 完全な答え |
|---|---:|---:|
| `S_state` | 0.6929 | 1.0000 |
| `S_queue` | 0.3017 | 1.0000 |
| `S_physics` | 0.3549 | 0.9636 |
| `S_ODME` | 0.8359 | 1.0000 |
| **`S_total`** | **0.5534** | **0.9945** |

`S_physics` は完全な答えでも 1.0 に届きません。保存則の残差が測定ノイズを含む観測に対して検査されるためで、
実用上限は 0.95 程度。**この上限は全員に共通**です。

### 提出物は 3 つ、採点は 4 つ

```text
task1  state   speed_kmh, flow_vph        → S_state を決め、同時に S_physics も決める
task2  queue   queue_pred (0/1)           → S_queue
task4  odme    path_flow                  → S_ODME
```

**Task 3 には独自の提出がありません。** 公式ドキュメントの表現では「Task 3 は、あなたが Task 1 で提出した
速度と流量に対して採点される」。評価器が自分で物理フレームを導出します。

---

## 2. Task 1（S_state）

### 何を出すか

`MASK_SPEC.md` の 3 つのマスク率 `R1 = 0.20`、`R2 = 0.30`、`R3 = 0.50` について、伏せられたセルの
`speed_kmh` と `flow_vph` を出します。採点は

```text
S_speed(r) = max(0, 1 - RMSE_speed(r) / 25)
S_flow(r)  = max(0, 1 - RMSE_flow_per_lane(r) / 600)
S_state(r) = 0.54*S_speed(r) + 0.46*S_flow(r)
S_state    = 3 つの regime の平均
```

`RMSE_flow_per_lane` は提出値と真値の両方を `fd_parameters.csv` の車線数で割ってから残差を取ります。
**RMSE に線形**なので、改善幅がそのままスコアになります。

### モデル

`analysis/t1_m3_cnn.py` の CNN。入力は (時刻 × リンク) の格子で、マスクされたセルを周囲の文脈から復元します。

| 設定 | 値 |
|---|---|
| width | **192**（production は 64 だった） |
| 受容野 | **`all1`** — 全ブロックの dilation を 1 に。パラメータ数は同じ |
| パラメータ数 | 7.98M |
| seed | **5 本の平均** |

### 学習は 2 段階

```bash
# stage 1: 自前で生成したマスクでの自己教師あり再構成（マスク率 U(0.10, 0.70)）、25 epoch
# stage 2: 本物の Task 1 マスクで 25 epoch。seed 0 は --tag ssl_wide、seed k (1..4) は M3_SEED=k --tag ssl_wide_sk
M3_WIDTH=192 M3_RF=all1 python analysis/t1_ssl.py --mode wide --pre-epochs 25 --ft-epochs 25 --ft-lr 7e-4 --tag ssl_wide
```

stage 1 は **本物の Task 1 マスクではなく、自分で作ったマスク**を使います。stage 2 で本物のタスクに
25 epoch、lr 7e-4 で微調整します。

### テスト時適応（TTT）

```bash
C="work/models/t1_ssl_wide_all.pt:192,work/models/t1_ssl_wide_s1_all.pt:192,work/models/t1_ssl_wide_s2_all.pt:192,work/models/t1_ssl_wide_s3_all.pt:192,work/models/t1_ssl_wide_s4_all.pt:192"
M3_WIDTH=192 M3_RF=all1 TTT_STEPS=1200 TTT_CKPTS="$C" \
  python scripts/t1_ttt_train.py --split validation --out-dir work/sub_sslttt   # private も同様
```

採点対象の split の released 層に対して、**マスク再構成の損失で 1,200 step 適応**させます。真値もラベルも
使いません。5 本の member それぞれを、validation と private で別々に適応させます。

**再現用のコード一式と詳細な手順は `share/task1_AS/`（`README_task1_AS.md`、`run_task1_AS.sh`）にあります。**

**ここが効きの中心です。** 同じ TTT が、

| 土台 | TTT による S_state | TTT による S_LWR |
|---|---|---|
| SSL 事前学習なし | +0.0012 | **−0.0024** |
| SSL 事前学習あり | +0.0020 | **+0.0026** |

**マスク再構成を一度も解いたことがないモデルは、TTT で分布から引き離される。一度解いたモデルは精錬される。**
stage 1 の課題と TTT の課題が同じものだからです。マスク分布（`iso` / `prod` / `wide`）の違いは
測定限界内で、**効いているのは「テスト時適応と両立するか」だけ**でした。

### 推論

```bash
M3_WIDTH=192 M3_RF=all1 python scripts/t1_predict_ckpt.py \
  --ckpt work/sub_sslttt/ttt_validation_{0,1,2,3,4}.pt:192 \
  --month-kappa 2 --month-kappa-q 10 --out-dir work/sub_sslens5 --splits validation --npz-only
python scripts/t1_npz_to_state.py --pred-dir work/sub_sslens5
```

`--month-kappa` は、採点月の過去日から作る月別の偏差ベースの強さです。

### 積み上げの内訳

| 段階 | S_LWR |
|---|---|
| production width 64、9 seed 平均 | 0.63635 |
| width 192 | 0.65135 |
| + 受容野 `all1`（パラメータ数は同じ） | 0.65428 |
| + SSL 事前学習 | 0.65729 |
| + 5 seed、+ TTT | TTT の符号が反転する |

---

## 3. Task 3（S_physics）

### 採点のされ方

**提出は無く、Task 1 の答えがそのまま採点されます。** 評価器が導出するもの:

```text
k = q/v                 密度は恒等式であって推定値ではない
N = k*L                 滞留量は密度と区間長から出る
inflow, outflow         organizer が導出、参加者のものではない
r_on, r_off             released のランプ観測
```

```text
S_FD  : 車線あたりで評価。基本図に対する妥当性の検査
S_LWR : N(t+dt) - N(t) = dt*(q_in + r_on - q_out - r_off)
        S_LWR = max(0, 1 - sum|residual| / sum|rhs|)
S_physics = (1/3)*S_FD + (2/3)*S_LWR
```

公式ドキュメントの言葉では「**`S_physics` を動かす唯一の方法は、より良い Task 1 の答えを提出すること**」。
密度を `q/v_f` と置いて `S_FD` を 0.9999 にするような自由度は、評価器側が全て塞いでいます。

### 実務上の要点

- **境界流量は organizer が保持し、公開されません。** released 観測に射影されているので、公開された値を
  再現できた参加者は保存則を厳密に満たします。
- **トポロジーから自分で境界流量を作るのは機能しません。** 保存則の信号は累積量の 0.75% しかないのに、
  トポロジー由来のフラックスはシミュレーション値を約 0.22% 外します。信号の 1/3 に相当し、この方法では
  真値でさえ `S_LWR` 0.0 になります。
- したがって `S_LWR` は実質的に **`ΔN` の相対 L1 誤差**です。`k = q/v` の**時間差分**を当てる問題であり、
  水準を当てる問題ではありません。
- ローカルでは `tfb/physics.py:physics_scores` が同じ形で計算します。

### 現状

`S_FD` は 0.98〜0.99 でほぼ埋まっています。残っているのは `S_LWR`（regime 別に R1 約 0.78、R2 約 0.67、
R3 約 0.48〜0.51）で、**マスク率が上がるほど落ちます**。

---

## 4. Task 4（S_ODME）

```text
S_od   = max(0, 1 - sum|fhat - f*| / max(sum f*, eps))
S_link = max(0, 1 - sum|A*fhat - c| / max(sum c, eps))
S_dev  = exp(-|Dhat/Dstar - 1|),  Dhat = sum|fhat - b|,  Dstar = sum|f* - b|
S_attr = max(0, 1 - 0.5*L1(a_hat, a_star))
S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr
```

`f*` は**withheld な参照解**、`A` は released の path-link incidence、`c` は released のリンク交通量、
`b` は released の weak prior です。

提出しているのは**組織側の参照解法そのもの**です。

```text
f = nnls([A ; sqrt(0.05) I], [c ; sqrt(0.05) b])
```

先験値 `b` への L2 リッジ付き非負最小二乗（`REG_LAMBDA = 0.05`）。`src/task4/` の成果物構築スクリプトに
実装されているものを使っています。

**4 項のうちローカルで計算できるのは `S_link` だけ**で、`S_od` / `S_dev` / `S_attr` は withheld な `f*` を
必要とします。`S_link` はローカルで 0.9999 付近です。

---

## 5. 提出の組み立て

```bash
python scripts/assemble_submission.py \
  --state work/sub/state_validation.csv work/sub/state_private.csv \
  --queue work/sub/queue_validation.csv work/sub/queue_private.csv \
  --odme  work/sub/t4_validation/baseline_submission.csv work/sub/t4_private/baseline_submission.csv \
  --output work/sub/submission.csv
```

- 採点対象は 2 ヶ月なので、各タスクが split ごとに 1 ファイル
- 連結は**生テキストのまま**（timestamp をバイト単位で保存する必要があるため）
- 組織側の `merge_submissions.py` で `submission_key.csv` にマージし、`sample_submission.csv` と
  行単位で突き合わせる
- 最終形は `submission_id, task, speed_kmh, flow_vph, queue_pred, path_flow` の 1 ファイル

部分的な差し替えには `scripts/swap_task_rows.py` と `scripts/t2_patch_ongoing.py` を使います。触らない行は
バイト単位で保存されます。

---

## 6. 全体像

| タスク | 提出 | モデル | 備考 |
|---|---|---|---|
| Task 1 | state | w192 / RF `all1` の CNN、SSL 事前学習 + 微調整、**5 seed**、各 split で TTT 適応 | 7.98M × 5 |
| Task 2 onset | queue | LightGBM 5 bag + 適応 CNN 2 種 + onset STID 4 seed、期待 IoU の ratio rule | 別文書 |
| Task 2 ongoing | queue | LightGBM + CNN 3 + Transformer 8 + 適応 CNN、残差 corrector、τ 0.50 | 別文書 |
| Task 3 | **無し** | Task 1 の答えがそのまま採点される | |
| Task 4 | odme | 組織側の参照解法（先験値への L2 リッジ付き NNLS） | |

### 参考: ローカルで測れるもの / 測れないもの

| | ローカル |
|---|---|
| `S_state` | 測れる（train split は unmasked 観測層を持つ） |
| `S_physics` | `tfb/physics.py` で同形に測れる |
| `S_queue` | queue ラベルが全 split で withheld。official-like 窓での代理評価のみ |
| `S_ODME` | `S_link` のみ。他 3 項は withheld な `f*` が要る |
