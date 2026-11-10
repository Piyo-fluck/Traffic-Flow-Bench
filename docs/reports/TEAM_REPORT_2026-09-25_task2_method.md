# Task 2（queue forecasting）現行手法の全体説明

対象は現在の最高スコア提出 `work/sub_AS_sslens5/submission.csv`（LB `S_total` 0.88405）の **Task 2 部分のみ**。
この提出の queue 行 174,000 行は、`work/sub_t2AP_onset`（onset 87,000 行）と `work/sub_t2AR`（ongoing 87,000 行）
の中身と全行一致することを確認済み。

---

## 1. 解いている問題（公式定義）

出典: `trafficflowbench-public/docs/SCORING_SPEC.md`, `docs/DATA.md`,
`config/synthetic_release_v1.json`, `src/task2/score_task2.py`。

- forecast origin `T` までの **60 分の履歴**を見て、`T+5 … T+30`（5 分刻みで 6 ステップ）の
  **binary queue indicator** を link ごとに予測する。
- 正解の定義は `speed <= v_cut`, `v_cut = 0.60 * free_speed`。**観測値ではなく underlying state** から作られる
  （観測ノイズで閾値近傍が反転しないようにするため）。
- window ごとのスコアは

  ```
  IoU = |Qpred AND Qtrue| / |Qpred OR Qtrue|
  ```

  `is_score_eligible` が真のセルのみが対象。union が空なら 1.0、eligible セルが 1 つも無い window は 0.0。
- 集約は `score_task2.py` の通り、window → (panel, condition) 平均 → panel 平均（2 condition の `equal_mean`）
  → family 平均 → overall 平均 = **`S_queue`**。
- `S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME`。
- 公式 baseline（persistence）の `S_queue` は **0.3017**、満点が 1.0000（`docs/BASELINES.md`）。

### window の 2 condition

`config/synthetic_release_v1.json` の `task2` ブロックに条件が明記されている。

| 項目 | 値 |
|---|---|
| `conditions` | `queue_onset`, `queue_ongoing` |
| 各 condition の window 数 | panel × split ごとに 5 本ずつ |
| `requires_queue_in_horizon` | `true`（両 condition とも horizon に queue がある） |
| `minimum_spacing_minutes` | 360 |
| `minimum_history_coverage` | 0.7 |
| `maximum_persistence_iou` | 0.9 |

- `queue_onset`: 履歴に queue が見えず、horizon に queue が出る window。
- `queue_ongoing`: origin の時点で既に queue が見えている window。
- condition ラベルは**見える履歴だけから決まる**（`docs/SCORING_SPEC.md`）。
- 2 condition は**等ウェイト**。したがって onset 5 本と ongoing 5 本が同じ重みを持つ。

`D12_I405_N` と `D12_I405_S` は **Task 2 のみ対象外**なので、Task 2 の panel は 8 本。

### 入力ファイル

`data/kaggle_public/task2/<PANEL>/<split>/` に 3 つ。

| ファイル | 中身 |
|---|---|
| `window_index.csv` | `window_id, panel, family_id, split, date, forecast_origin, history_start, history_end, forecast_start, forecast_end, condition, history_coverage, future_coverage` |
| `window_history.parquet` | `window_id, timestamp, link_id, speed_kmh, flow_vph, occupancy, pct_observed, is_score_eligible` |
| `sample_submission_queue.csv` | `window_id, timestamp, link_id, queue_pred` — 予測すべきセルの完全なリスト |

提出は `queue_pred` が **0/1 のみ**（確率不可）、`(window_id, timestamp, link_id)` の重複不可
（`src/task2/score_task2.py` が弾く）。

---

## 2. 全体構成

onset と ongoing で**完全に別のパイプライン**を使い、最後に 1 本の提出にまとめる。
2 condition が等ウェイトなので、両者は同じ価値を持つ。

```
                      ┌─ onset  windows (5/panel/split) ─→ 4 member の重み付き平均 ─→ ratio rule ─→ 0/1
  official windows ───┤
                      └─ ongoing windows (5/panel/split) ─→ 6 member → 2 段の blend → residual corrector → τ=0.50 → 0/1
```

共通の前処理として、履歴 `window_history.parquet` を `(N, H=12, S)` のテンソルに直す
（`scripts/t2_m1_predict.py:hist_from_release`、occupancy は `scripts/t2_occ_blend_predict.py:occ_from_release`）。
`S` は panel の link 数。欠測は `is_score_eligible` と有限性の AND で `ok` マスクにする。

---

## 3. ニューラル系 member 共通の入力表現

`analysis/t2_cnn.py:make_x` が全ての CNN / Transformer member の入力 `(N, C, H, S)` を作る。

| チャネル群 | 内容 |
|---|---|
| 動的 4ch | `speed/v_free`、`flow/capacity`、`ok` マスク、`ok AND (speed <= 0.6*v_free)` |
| 静的 5ch | `lanes/5`、`capacity/lanes/2500`、`v_free/120`、on-ramp 接続、off-ramp 接続（履歴軸にブロードキャスト） |
| 時刻 3ch | `sin(2π t0/T)`、`cos(2π t0/T)`、weekend フラグ |

occupancy を使う member（`occ_cnn` / `occ_tfm` / STID）は `occ_lab/train_occ.py:occ_channels` で 3ch 追加する:
occupancy そのもの、リンク別プロファイルとの比、履歴方向の階差。プロファイルは `work/models_remote/occ_profiles.npz`。

### アーキテクチャ

- **`T2Net`**（`t2_cnn.py`）: `1x1 Conv` で width 64 に上げ、`(1,2,4,8,16,1,2,4)` の dilation を link 軸に
  持つ残差ブロック 8 段（各ブロックは GroupNorm → GELU → dilated 3x3 → GroupNorm → GELU → 3x3）。
  最後に履歴軸を `(H,1)` の Conv で潰して `(B, F, S)` のロジットを出す。
  `T2CNN_REC>0` のとき再構成ヘッド（2ch）が付く。
- **`T2Tfm`**（`t2_cnn.py`）: 各 link の 12 スロット履歴を GRU で 1 トークンに符号化し、link 軸方向の
  相対位置バイアス付き multi-head self-attention を重ねる。
- **`STID`**（`occ_lab/train_stid.py`）: 履歴を link ごとに平坦化して線形射影 `d=96`、そこに
  **link identity embedding**（`ds=32`）、time-of-day embedding、day-of-week embedding を連結し、
  残差 MLP ブロック 3 段 → `head` で `(B, F, S)`。link identity を明示的に持つのがこの member の役割。

損失は binary cross entropy に Lovász hinge（`T2CNN_LOVASZ=0.5`）を加えたもの。Lovász は IoU の凸代理なので、
評価指標が IoU であることに合わせている。

---

## 4. onset 側

`scripts/t2_onset_blend_predict.py`。`T+30`（horizon の最終ステップ）だけを予測し、それ以外のステップは 0 にする。
onset window は履歴に queue が無く、`requires_queue_in_horizon` により horizon のどこかに queue がある window で、
実際の出現は最終ステップに集中するため。

### 4 member の重み付き平均

```
p(T+30) = 0.35 * H + 0.175 * ad9b + 0.175 * pseudo + 0.30 * STID_onset
```

| member | 実体 | 説明 |
|---|---|---|
| `H` | `work/sub_t2robust_bag5/t2_onset*.txt`（LightGBM 5 bag） | train 期間で学習した onset 分類器。特徴は `scripts/t2_m1_predict.py:onset_X` — 履歴の速度・流量統計に加え、train から作った **table prior**（時間帯 × 曜日グループごとの link 別 queue 確率 `P_q`、onset 確率 `P_on`）、その `±1, ±2` link シフト、link 内順位、最大値比 |
| `ad9b` | `scripts/t2_adapt_official.py` | production checkpoint から出発し、**採点月の各 window 日について、その日より前の日のデータだけ**で再構成損失（Task 2 ラベル不使用）を 30 step, lr 1e-4 かけて適応させた `T2Net`。重みは日をまたいで持ち越すので因果的 |
| `pseudo` | `scripts/t2_pseudo_adapt_official.py` | 同じく採点月で適応させるが、**再構成ではなく Task 2 そのもの**で教師あり適応する。origin `τ` を、その horizon 全体が当日の最初の official origin 以前に収まるように取り、履歴と**ラベル**の両方を release から作る。30 step, lr 1e-4, `T2CNN_REC=0.3 T2CNN_LOVASZ=0.5` |
| `STID_onset` | `occ_lab/models/onset_stid_s{0..3}.pt`（4 seed 平均） | `OnsetSTID`（`occ_lab/train_onset_stid.py`）。`STID` の出力を `F` 個から 1 個（`T+30`）に変えたもの。全 panel の link を通し番号にした共通の identity embedding を持つ |

### 決定規則（ratio rule）

閾値ではなく、**window ごとに何本の link を on にするか** を期待 IoU で決める
（`analysis/t2_expected_iou.py:select` / `best_k_ratio`）。確率を降順に並べ、上位 `k` 個の累積和を `S_k`、
全体和を `P` として

```
k* = argmax_k  S_k / (k + P - S_k)
```

を取り、上位 `k*` 本を 1 にする（`c = 1.0`、つまり確率のスケーリング無し）。
これは独立 Bernoulli の下での期待 IoU の比近似で、確率質量が薄く広がっている window で過剰に link を
取ることを防ぐ。**ratio rule は onset のみに使う**（ongoing では使わない。理由は 5 節末尾）。

---

## 5. ongoing 側

`scripts/t2_residual_predict.py`。horizon 6 ステップ全体を予測する。

### 5.1 member（6 本）

| member | 実体 |
|---|---|
| `gs1` | `work/sub_t2gs`（`scripts/t2_gstate_probs.py`）。LightGBM ongoing モデル + **global network-state 特徴**（panel ごとの PCA、`work/models_gstate/pca_<panel>.npz`）。特徴は official window の履歴のみから作る |
| `tfm8` | `T2Tfm` 8 seed の平均（`t2_cnn_lov_tfm_s3..s10`） |
| `cnn3` | `T2Net` 3 seed の平均（`t2_cnn_lov`, `_s1`, `_s2`） |
| `occ_cnn` | occupancy チャネル付き `T2Net` 4 seed（`occ_cnn_s0..s3`） |
| `occ_tfm` | occupancy チャネル付き `T2Tfm` 4 seed（`occ_tfm_s0..s3`） |
| `stid` | `STID` 4 seed（`occ_lab/models/stid_s{0..3}.pt`） |

### 5.2 2 段の blend

```
old = 0.20 * gs1 + 0.60 * tfm8 + 0.20 * cnn3
p0  = 0.40 * old + 0.60 * p_PS
```

`p_PS` は onset と同じ `scripts/t2_pseudo_adapt_official.py` の出力の ongoing 部分。
**採点月そのものの release から作った教師あり適応**が blend の 0.60 を占める。
`occ_cnn` / `occ_tfm` / `stid` は `p0` には入らず、次段の corrector の特徴としてのみ使う。

### 5.3 residual corrector

`occ_lab/models/residual_lgb.txt`（LightGBM、`occ_lab/fit_corrector.py` で学習）。
`p0` を直接ではなく、**残差 `r = y - p0` を回帰**する。

15 特徴（`occ_lab/residual_diag.py:NAMES`、順序は固定）:

```
p_blend, logit_blend, margin(|p0-0.5|), member_std, member_range,
gs1, tfm8, cnn3, occ_cnn, occ_tfm, stid,
logit(stid)-logit(p0), logit(occ_cnn)-logit(p0), horizon, position
```

学習データは **2031-01/02 の window**（`occ_lab/cache_oof.py` の out-of-fold キャッシュ）。
member はそこで学習に使われておらず、かつ全ての採点 origin より前の期間なので、両採点月に対して使える。

適用は

```
r_hat = corrector(features)
p     = clip(p0 + 0.75 * r_hat * 1[|r_hat| >= 0.10], 0, 1)
```

`eta = 0.75`、`gate = 0.10`。gate により、corrector が自信を持って動かす場所以外は `p0` をそのまま残す。
実測では 1 split あたり修正されるセルは 200 前後と少ない。

### 5.4 決定規則

```
queue_pred = 1[p >= 0.50]
```

平坦な閾値。ongoing には onset の ratio rule を**使わない**：ongoing のセルは空間・時間方向に強く相関しており、
ratio rule が前提とする独立 Bernoulli が成り立たないため、ここでは閾値のほうが良い。

---

## 6. 提出の組み立て

1. onset 側 → `work/sub_t2AP_onset/onset_{validation,private}.csv`（onset window の行のみ）
2. ongoing 側 → `work/sub_t2AR/ongoing_{validation,private}.csv`（ongoing window の行のみ）
3. `scripts/t2_patch_ongoing.py` / `scripts/swap_task_rows.py` で、既存提出の queue 行に対し
   該当行だけを差し替える（触らない行はバイト単位で保存される）
4. `scripts/assemble_submission.py` が Task 1/2/4 のファイルを `submission_key.csv` にマージし、
   `sample_submission.csv` と行単位で突き合わせて最終提出を書く

再現コマンドは `work/run_AP_onset.sh` と `work/run_AR.sh` にそのまま置いてある。

```bash
# onset
.venv/Scripts/python.exe scripts/t2_onset_blend_predict.py --out-dir work/sub_t2AP_onset \
  --h-dir work/sub_t2robust_bag5 --ad-dir work/sub_t2adaptcnn --ad2-dir work/sub_t2pseudo \
  --stid occ_lab/models/onset_stid_s0.pt occ_lab/models/onset_stid_s1.pt \
         occ_lab/models/onset_stid_s2.pt occ_lab/models/onset_stid_s3.pt \
  --profiles work/models_remote/occ_profiles.npz \
  --w-h 0.35 --w-ad 0.175 --w-ad2 0.175 --w-st 0.30
```

```bash
# ongoing
M=work/models_remote
.venv/Scripts/python.exe scripts/t2_residual_predict.py --out-dir work/sub_t2AR --eta 0.75 --gate 0.10 \
  --cnn $M/t2_cnn_lov.pt $M/t2_cnn_lov_s1.pt $M/t2_cnn_lov_s2.pt \
  --tfm $M/t2_cnn_lov_tfm_s3.pt ... $M/t2_cnn_lov_tfm_s10.pt \
  --occ-cnn $M/occ_cnn_s0.pt ... $M/occ_cnn_s3.pt \
  --occ-tfm $M/occ_tfm_s0.pt ... $M/occ_tfm_s3.pt \
  --stid occ_lab/models/stid_s0.pt ... occ_lab/models/stid_s3.pt \
  --profiles $M/occ_profiles.npz --gs-dir work/sub_t2gs \
  --pseudo-dir work/sub_t2pseudo --w-pseudo 0.60
```

---

## 7. ローカル評価のやり方

Task 2 の queue ラベルは全 split で withheld なので、`src/task2/score_task2.py` は公開パッケージでは動かない
（`queue_targets.parquet` が無い）。そこで評価は 2 通り。

1. **official-like window**（`analysis/t2_official_like_eval.py`）
   train split は唯一 unmasked 観測層を持つので、そこに `config/synthetic_release_v1.json` と同じ選別規則
   （origin 07–21 時、`minimum_history_coverage` 0.7、`maximum_persistence_iou` 0.9、horizon に queue あり）
   を当てて window を作り、公式と同じ IoU と同じ集約で測る。
2. **out-of-fold キャッシュ**（`occ_lab/cache_oof.py`）
   2031-01/02 の window に対する各 member の確率を保存したもの。corrector の学習と、blend 定数の確認に使う。

どちらも公式の `is_score_eligible` 相当のマスクを使い、公式と同じ「union が空なら 1.0」の規約で計算する。

---

## 8. 設計上の要点

- **onset と ongoing は等ウェイト**なので、片方だけ強くしても `S_queue` は半分しか動かない。
- **採点月自身の release が使える**。organizer の規定では window の origin `T` 以前の released observation は
  全て使ってよく、train split では `obs == true` がビット単位で成り立つ。
  これを使って採点月上に「本物の Task 2 window」を作り、教師あり適応をかけているのが `pseudo` member で、
  ongoing blend の 0.60、onset blend の 0.175 を占める。適応は必ず**当日より前の日**、および
  **当日の最初の official origin より前に horizon 全体が収まる window** のみから行い、因果性を破らない。
- **member を増やす形と、決定規則を変える形で効き方が違う**。onset は window ごとの本数決定（ratio rule）が効き、
  ongoing は平坦な閾値のままで、代わりに member 数と corrector で効かせている。
- corrector は `p0` を置き換えるのではなく **gate 付きの残差補正**なので、ほとんどのセルを素通しする。
  壊れ方が限定されるのが利点。
