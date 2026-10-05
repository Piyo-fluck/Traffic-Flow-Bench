# Task 1 の手法（候補 AS）— 再現用の詳細説明

対象は、LB 0.88405 を出した候補 AS の Task 1（state）です。この Task 1 は、現在の最高提出 BB（0.88697）でも
そのまま使われています（AS 以降の改善はすべて Task 2）。

このフォルダには、Task 1 を最初から作り直すのに必要なコードがすべて入っています。

- 実行の手順：`run_task1_AS.sh`
- 中身の説明：この文書

---

## 0. 準備

### フォルダ構成

このフォルダ（`task1_AS/`）を作業ディレクトリにして、次の 2 つを同じ階層に置いてください。

```text
task1_AS/
  data/kaggle_public/          コンペの配布データ（corridors/, task1/, task2/, task4/, submission_key.csv など）
  trafficflowbench-public/     運営のリポジトリ（scripts/assemble_submission.py が merge_submissions.py を使う）
  tfb/                         データ読み込み・特徴・評価・提出の共通部品
  analysis/t1_m3_cnn.py        CNN 本体、入力の作り方、損失、学習ループ、推論
  analysis/t1_ssl.py           学習（stage 1 自己教師あり → stage 2 本番タスク）
  analysis/t1_ttt.py           TTT の部品（マスクの掛け方、学習率、ステップ数の既定値）
  scripts/t1_tables_cache.py   ベース表（曜日 × 時刻 × 地点の平均）の作成
  scripts/t1_ttt_train.py      採点対象の split への TTT
  scripts/t1_predict_ckpt.py   推論（5 モデル平均 + 月別の偏差ベース）
  scripts/t1_npz_to_state.py   予測 → state_<split>.csv（テンプレートと照合）
  scripts/swap_task_rows.py    既存の提出ファイルの state 行だけを差し替える
  scripts/assemble_submission.py  提出ファイルを最初から組み立てる
  run_task1_AS.sh              上の手順を順に実行するスクリプト
```

- `tfb/data.py` は、自分の位置から見て `data/kaggle_public` を読みます。
- キャッシュ（`work/cache/`）、モデル（`work/models/`）、出力（`work/sub_*`）は自動で作られます。

### 環境（AS を作ったときの版）

| | 版 |
|---|---|
| Python | 3.12 |
| torch | 2.x（CUDA 必須。bf16 autocast を使う） |
| numpy | 2.x |
| pandas | 3.x |
| pyarrow | 25.x |

### 実行

```bash
GPU=0 PY=python bash run_task1_AS.sh
```

各段階は環境変数 `M3_WIDTH=192 M3_RF=all1` を前提にしています。スクリプト内で export 済みです。
**この 2 つを付け忘れると、構造の違うネットワークでチェックポイントを読もうとして失敗します。**

---

## 1. 何を予測するか（Task 1 と Task 3）

### Task 1

`MASK_SPEC.md` のマスク率 R1 = 0.20、R2 = 0.30、R3 = 0.50 で伏せられたセル（日 × 5 分枠 × 地点）について、
`speed_kmh` と `flow_vph` を出します。

```text
S_speed(r) = max(0, 1 - RMSE_speed(r) / 25)
S_flow(r)  = max(0, 1 - RMSE_flow_per_lane(r) / 600)       提出値と真値の両方を車線数で割る
S_state(r) = 0.54 S_speed(r) + 0.46 S_flow(r)
S_state    = 3 つの regime の平均
```

### Task 3

Task 3 には独自の提出がなく、**Task 1 の答えがそのまま採点されます**。

```text
k = q / v, N = k L
S_LWR = max(0, 1 - Σ|ΔN 誤差| / Σ|ΔN|)
S_physics = (S_FD + 2 S_LWR) / 3
```

全体のスコアに占める重みは 0.35 S_state + 0.05 S_FD + 0.10 S_LWR です。Task 1 の改善は、S_LWR を通じて
Task 3 にも効きます。

---

## 2. データの持ち方（`tfb/data.py`）

1 パネル × 1 split を、(日 D, 5 分枠 T = 288, 地点 S) の配列にします。

- **地点の順番：** LWR トポロジーの `order_index` 順（上流 → 下流）。
- **キャッシュ：** `work/cache/<panel>_<split>.npz` に保存します。

| キー | 中身 |
|---|---|
| `obs_speed`, `obs_flow`, `obs_occ` | 配布された観測層（NaN = 空白。Task 1 の予測対象と、もともとの欠測） |
| `elig` | `is_score_eligible` |
| `pct` | `pct_observed` |
| `target` | Task 1 テンプレートに載っている（= 予測を求められている）セル |
| `true_speed`, `true_flow`, `true_occ` | 正解（train のみ） |
| `lanes` | 採点器と同じ規則で決めた車線数（links.csv を fd_parameters.csv で上書き、1〜16 の範囲外は 1） |
| `free_speed` | 自由流速度 |
| `weekday`, `regime` | 曜日、マスク率の regime |

パネルは 10 本で、validation と private の 2 か月ぶんを予測します。

---

## 3. ベース（予測の土台）

ネットワークは速度・流量そのものではなく、**ベースからの偏差**を出力します。

### 学習時（`t1_m3_cnn.panel_tensors`）

- train の fit 期間（`dates < 2031-01-01`）で、曜日 × 時刻 × 地点の平均をとります。
- 平均に使うセルは `pct >= 75` かつ真値が両方あるものだけです（`tfb.features.profile` / `truth_valid`）。

### 推論時と TTT 時

1. **ベース表：** `scripts/t1_tables_cache.py` が、train の全日で同じ平均表 (7, T, S) を作ります
   （`work/cache/<panel>_t1tables.npz`）。
2. **月別の偏差（`tfb/month.month_deviation`）：** 採点対象の月の見えているセル（`elig` かつ値あり）から、
   ベースとの差を平日と週末に分けて平均し、ベースに足します。
   - 時間方向に ±1 枠で平滑化します。
   - 縮小の強さは、速度が `kappa = 2`（擬似日数）、流量が `kappa_q = 10` です。
   - 採点対象の run は train と違うシミュレーション run なので、その run の水準のずれをまず吸収します。

---

## 4. 入力と出力（`t1_m3_cnn.make_inputs`）

入力は 10 チャンネルで、形は (チャンネル, 時刻 288, 地点 S) です。「見えている」は観測値があるセルを指し、
`elig` は問いません（スコア対象外のセルでも、値が残っていれば使います）。

| ch | 中身 |
|---|---|
| 0 | (観測速度 − ベース速度) / 10。見えないセルは 0 |
| 1 | (観測流量 − ベース流量) / 車線数 / 100。見えないセルは 0 |
| 2 | 見えているかどうか（0/1） |
| 3 | 占有率 × 10。見えないセルは 0 |
| 4 | ベース速度 / 100 |
| 5 | ベース流量 / 車線数 / 1000 |
| 6 | 車線数 / 5 |
| 7 | 自由流速度 / 100 |
| 8, 9 | 時刻の sin / cos |

出力は 2 チャンネルです。

- `(速度 − ベース) / 10`
- `(流量 − ベース) / 車線数 / 100`

予測値は `ベース + 出力 × スケール` に戻します（流量は車線数を掛け戻します）。

---

## 5. ネットワーク（`t1_m3_cnn.DilNet`、`M3_WIDTH=192 M3_RF=all1`）

```text
入力 1x1 conv (10 → 192)
→ 残差ブロック × 12   各ブロック: GroupNorm(8) → GELU → 3x3 conv (dilation 1)
                                  → GroupNorm(8) → GELU → 3x3 conv
→ GroupNorm(8) → GELU → 1x1 conv (192 → 2)
```

- **パラメータ数：** 7.98M。
- **受容野：** `all1` は 12 ブロックすべての dilation を (1, 1) にした設定で、時刻・地点とも 49 です。
  以前の設定（`prod`）は dilation を最大 32 まで広げていました。パラメータ数は同じです。
- **計算の向き：** 時刻軸と地点軸の両方に畳み込みます。1 日ぶんを丸ごと 1 枚の画像として扱います。

---

## 6. 学習（`analysis/t1_ssl.py`、1 モデルあたり）

### 共通の設定

| 項目 | 値 |
|---|---|
| バッチ | 16 枚。各枚は (1 パネル, 1 日, 地点方向にランダムに切り出した 64 地点) |
| ステップ数 | 1 エポック 400 ステップ |
| optimizer | AdamW、weight decay 1e-4 |
| 学習率の制御 | OneCycle（warmup 10%）。勾配クリップ 1.0、bf16 autocast |
| 損失 | 予測対象セルの二乗誤差（スケール後の単位）を 0.75 × 速度 + 0.25 × 流量 で足す |
| 早期終了 | train の 2030-12-01 以降の日（ES 期間）の S_state が最良のエポックを採用 |
| 学習に使う日 | train の 2030-12-01 より前（ES 期間と 2031 年以降のテスト期間は使わない） |

### stage 1：自己教師あり（`--mode wide`、25 エポック、lr 2e-3）

1. 入力は、train の**正解**から作ります。「配布されていれば見えていたはずのセル」をすべて見えている状態にします。
2. そこに、**自分で作ったマスク**を掛けます。マスクのかかる範囲は `elig` かつ真値ありのセルで、マスク率は
   1 枚ごとに U(0.10, 0.70) で引きます。
3. 隠したセルを当てさせます。

本物の Task 1 マスクより広い範囲・強さの再構成を、毎エポック違うマスクで解かせることになります。

### stage 2：本番タスク（25 エポック、lr 7e-4）

入力を本物の Task 1 マスク（`target`）で隠し、`target` のセルを当てさせます。stage 1 の重みから始めます。

### 保存先

seed 0 は `--tag ssl_wide`（`work/models/t1_ssl_wide_all.pt`）、seed 1〜4 は `M3_SEED=<k> --tag ssl_wide_s<k>` です。
**seed だけを変えた 5 モデル**を作ります。

---

## 7. テスト時適応（TTT、`scripts/t1_ttt_train.py`）

採点対象の split（validation / private）それぞれについて、5 モデルを**別々に**適応させます。

| 項目 | 値 |
|---|---|
| 対象データ | その split の配布観測層の全日・全 10 パネル（パネルはまとめて 1 本の適応） |
| 入力 | 推論と同じもの（ベース表 + 月別の偏差、§3） |
| マスク | 1 枚ごとにマスク率を U(0.10, 0.50) で引き、見えていて `elig` のセルをその率で隠す |
| 損失 | 隠したセルの再構成誤差（§6 と同じ損失。観測値がそのまま正解になる） |
| ステップ数 | 1,200（バッチ 16、64 地点の切り出しは学習時と同じ） |
| optimizer | AdamW、lr 2e-4 で一定、weight decay 1e-4、勾配クリップ 1.0、bf16 |
| 乱数 | マスクは torch generator seed 0、バッチは numpy seed 1 で固定 |
| 出力 | `work/sub_sslttt/ttt_<split>_<i>.pt`（i はモデル番号 0〜4） |

正解ラベルは一切使いません。使うのは、採点対象の月の配布層に見えている値だけです。Task 1 はオフライン
タスクなので、月全体を使えます。

### ステップ数を 1,200 にした理由

旧構成（w64 × 2、w128 × 1）で、手元の月内評価を比べた結果です。

| ステップ数 | 合計スコアの変化 |
|---|---|
| 400 | +0.0010 |
| 1,200 | +0.0013 |
| 3,600 | +0.0013 |

1,200 で頭打ちと判断しました。AS の 5 モデル × 2 split は、計算時間の都合もあって 1,200 にしています。

ただし旧構成の LB では、**3,600 ステップのほうが +0.00026 良い**結果でした（0.87845 → 0.87871）。
手元の評価では、この差は見分けられていません。3,600 にすると、わずかに伸びる可能性があります。

### これが効く理由

TTT の課題（見えているセルを隠して当てる）は、stage 1 の課題と同じです。

| 土台 | TTT による S_state の変化 | TTT による S_LWR の変化 |
|---|---|---|
| stage 1 なし（同じ w192 / all1） | +0.0012 | **−0.0024** |
| stage 1 あり | +0.0020 | **+0.0026** |

- マスク再構成を解いたことのないモデルは、TTT で分布から引き離され、S_LWR が下がります。
- 解いたことのあるモデルは、TTT でさらに精錬されます。
- stage 1 のマスクの形（`iso` / `prod` / `wide`）による差は、測定の誤差の範囲でした。

---

## 8. 推論と提出（`scripts/t1_predict_ckpt.py` → `t1_npz_to_state.py`）

1. split ごとに、その split で適応させた 5 モデルで全日を予測し、速度と流量をそれぞれ**単純平均**します。
2. ベースは §3 の推論時のもの（`--month-kappa 2 --month-kappa-q 10`）です。
3. 結果を `work/sub_sslens5/pred_<split>_<panel>.npz` に保存します。
4. `t1_npz_to_state.py` が、パネルごとに Task 1 テンプレートと行を照合しながら、`state_<split>.csv` を書きます。
5. 既存の提出ファイルの state 行だけを差し替えます。ほかのタスクの行は、バイト単位でそのまま残ります。

```bash
python scripts/swap_task_rows.py --base <既存の submission.csv> --task state \
  --files work/sub_sslens5/state_validation.csv work/sub_sslens5/state_private.csv --out <新しい submission.csv>
```

AS の提出では、約 670 万の state 行（1,346 万値）を差し替えました。速度の最小値は 4.54 km/h で、
非正の値や NaN はありませんでした。

---

## 9. 期待される結果

### 手元の評価

月内の疑似評価（validation の見えているセルの 20% を隠して当てる）で、TTT は同じ 1,200 ステップです。

| | S_state~（TTT 前 → 後） | S_LWR~（TTT 前 → 後） |
|---|---|---|
| 旧構成（w64 × 2、w128 × 1）+ TTT | 0.9494 → 0.9518 | 0.7709 → 0.7752 |
| **AS（5 モデル）+ TTT** | 0.9501 → **0.9521** | 0.7783 → **0.7819** |

旧構成に対する差は合計 +0.00078 と予想し、LB では **+0.00087**（0.88318 → 0.88405）でした。

### 構成ごとの寄与（train のテスト期間での S_LWR、TTT なし）

| 構成 | S_LWR |
|---|---|
| 幅 64、9 seed 平均 | 0.63635 |
| 幅 192 | 0.65135 |
| + 受容野 `all1` | 0.65428 |
| + stage 1（SSL） | 0.65729 |
| + 5 seed 平均、+ TTT | TTT が S_LWR を上げる側に変わる |

### 計算時間の目安

TTT は RTX 4090 で 1 ステップ約 0.3 秒です。1,200 ステップ × 5 モデル × 2 split で、約 1 時間かかります。

---

## 10. 再現で間違えやすい点

- **`M3_WIDTH=192 M3_RF=all1`：** 学習・TTT・推論のすべてで必要です。
- **seed 0 のタグ：** `ssl_wide`（`_s0` は付きません）。TTT の `TTT_CKPTS` は 5 本を
  `パス:192` 形式・カンマ区切りで、0〜4 の順に並べます。推論の `--ckpt` も同じ順です。
- **TTT と推論の組み合わせ：** split ごとに別のチェックポイントになります。validation の推論には
  `ttt_validation_*`、private には `ttt_private_*` を使います。
- **ベースの作り方の違い：** 学習時のベースは fit 期間の平均、推論時は全 train 日の表 + 月別の偏差です。
  どちらも §3 のとおりで、意図した違いです。
- **ラベルの扱い：** TTT は採点対象の月の正解を使いません。使うのは配布層に見えている値だけです。
