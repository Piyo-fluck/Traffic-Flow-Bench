# onset 多峰性（multimodality）仮説の検証キット

置き場所：`/data1/nagao/Kaggle/TrafficFlowBench/mm_lab/`
前提：`tta_lab/`（共通部品 `tta_lab/tta/common.py` を import する）。関連ノート：72（TTA 結果）

コードの状態：`py_compile` のみ通過、**未実行**。精査のうえ使ってください。

---

## 0. 仮説と、何が分かれば次に進むか

**仮説**：onset は多峰的。同じ60分履歴から、観測されない微小な需要差によって複数のボトルネックがあり得る。BCE で単一の確率マップを学習すると、最頻モード（常連ボトルネック）に潰れる。

これまでの結果との整合：
- 失敗窓は「自信がない」のではなく、「常連を自信満々に選んでいる」。
- アンサンブル平均でレア候補が薄まる（note 72 F：単体+0.0055 → T-blend −0.0024）。

**分岐**：

| 診断①②③の結果 | 解釈 | 次の一手 |
|---|---|---|
| 外し窓の真ボトルネックが、近傍の第2〜3モード／Δの上位／ε\*の上位に出る | 情報はあるが、単一マップに潰れている | ④ multi-hypothesis へ |
| どの診断でも出ない（近傍 share≈0、Δ rank 低、ε\* 大） | 60分履歴に識別情報がほぼない | architecture で 0.9 を追うのをやめる |

## 共通の定義（`mm/diag_123.py` 冒頭）

- **窓**：blank-aware 公式風 onset 窓（validation 169 / private 162、8パネル）。`tta_lab/tta/common.windows` を使う。
- **真ボトルネック集合**：T+30 で渋滞している eligible リンク（`on["cells"]`）。
- **候補**：候補T の onset マップ `p_T = 0.7·H + 0.3·⑨b` を、NMS（間隔 `--sep`=3 リンク）で上位 `--ncand`=10 本取った位置。
- **hit(i)**：候補 i の ±sep 内に真のリンクがある。
- **rank(score)**：score の NMS ピークを上から見て、最初に hit したピークの順位。上位10本に無ければ inf。
- **外し窓**：候補T の選択集合の IoU が 0。`IoU<0.5`（weak）と `IoU≥0.5`（hit）も併記する。
- **窓のボトルネックID**：真集合の最大連結区間の中心リンク。①の近傍集計に使う。
- ⑨b は `tta_lab` の rep9b と同じ手順で再生する：rec_s20、predict-then-adapt、厳密に過去の日から16窓、AdamW 1e-4 × 30 step、seed 3。

---

## ① 近傍多峰性診断（Bayes 限界の目安）

**目的**：予測法ではなく、条件付きエントロピーの測定。「似た履歴の未来ボトルネックが何種類に分かれるか」を見る。

**手順**：
1. train 月の同一パネルの onset 窓（`ctx.eo`、真値層）を、未適応トランクで埋め込む。
   - 埋め込み＝`collapse(blocks(inp(x)))` の (c, S) をリンク方向に 8 で平均プール → flatten → L2 正規化。
   - ラベルは各窓のボトルネックID。
   - リンク位置が意味を持つため、近傍検索は同一パネル内に限る。
2. スコア月の各窓を、その日の**適応済みトランク**（`ad`）と未適応トランク（`bs`）の両方で埋め込み、cosine 上位 k（20/50/100）の train 窓を取る。
   - パネルあたりの train onset 窓は約290なので、k=100 は全体の約1/3。
3. 近傍のボトルネックIDを ±sep でまとめてモードにし、次を記録する：
   - `top_share`：最頻モードの割合
   - `true_share`：正解ボトルネックの近傍割合（0 なら「履歴に第2モードすら無い」）
   - `true_mode_rank`：正解が何番目のモードか
   - `Tpick_share`：T が選んだ位置の割合
   - `entropy`

**読み方**：
- 外し窓で `true_share ≥ 0.1` かつ `true_mode_rank ≤ 3` が多い → 多峰で、第2モードは履歴にある → ④が有望。
- `true_share = 0` が大半 → この表現空間では識別情報なし。ただし埋め込みの限界もあり得るので、②③と合わせて判断する。
- 注意：train run とスコア月は run が違う。`ad` と `bs` の差も見ること。

## ② 適応による予測差 Δ

**定義**：
- `Δ_i = logit p_i^adapt − logit p_i^base`（T+30、リンクごと）
- `Δ~_i = Δ_i − median_j Δ_j`（eligible リンク上）。run 全体の変化分を除いた量。

**測るもの**：
- `rank_delta`（Δ~ の大きい順）、`rank_absdelta`（|Δ~| の大きい順）。
- 真ボトルネックの top-1 / top-3 率と中央値順位を、外し窓 / weak / hit / 全体で出す。
- 比較用に `rank_T`、`rank_H`、`rank_base`、`rank_adapt` も出す。
- 直接補正：`logit p_new = logit p_T + λ·Δ~`、λ ∈ {0, 0.1, 0.25, 0.5, 1, 2}。
  - 月ごとの onset family を出し、**他方の月で選んだ λ** を適用した差（cross-split）も出す。

**読み方**：
- 外し窓で `rank_delta` の top-3 が高ければ、直接補正の cross-split 差を見る。
- 両月とも正で +0.005 family 以上なら提出候補になり得る（LB では約半分に縮む）。

## ③ Counterfactual breakdown margin

**手順**（適応済み CNN、候補T が実際に使う重みで）：
1. 各候補 i（上の10本）について、候補 i とその上流 `--up`=4 リンクの flow を、直近 `--slots`=3 スロット（15分）だけ `(1+ε)` 倍にして再推論する。
   - ε ∈ {0.25, 0.5, 1, 2, 4, 8}%。v・ok は変えない（make_x の q/cap チャネルだけが変わる）。
2. `ε*_i` ＝ 候補 i が NMS 1位ピーク（±3）になる最小の ε。次の2種で測る：
   - `cnn`：CNN 単体
   - `blend`：0.7·H（固定、摂動に反応しない）+ 0.3·摂動 CNN
3. 候補を「ε\* 昇順、同値なら最大 ε でのスコア上昇量の降順」で並べ、真ボトルネックの順位 `cf_*_rank` を記録する。
4. `cf_*_eps_true`（正解候補の ε\*）と `cf_*_eps_Tpick`（T の1位候補の ε\*）も記録する。

**読み方**：
- 外し窓で `cf_cnn_rank` の top-1/3 が `rank_T` より明確に高ければ、「最も少ない追加需要で崩れる所」は有用。その順位を LGBM 特徴か直接補正に使う。
- `eps_true` が有限にならない窓が大半なら、モデルの決定境界上でも正解は遠い。

**注意**：
- 上流の定義は「リンク配列の小さい側」（tfb.data：upstream→downstream の順）。
- 真値を候補の生成に使っていない（候補は p_T の NMS）。
- 真値が上位10本に入らない窓は、`true_in_cands` で分かる。

### 実行（①②③は1本）

```bash
cd /data1/nagao/Kaggle/TrafficFlowBench/mm_lab
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python mm/diag_123.py --name diag_all
# 部分実行：--panels D7_I10_E、sep や k の変更：--sep 2 --k 20 50
```

出力：`out/<name>/per_window.csv`（1窓1行、全指標）と `summary.txt`（部分集合ごとの集計 + ②の直接補正）。
所要時間の目安：⑨b の再生（約2分）＋ counterfactual（331窓 × 10候補 × 6ε、バッチ推論）で、数分〜十数分。

---

## ④ 平均しない K 個の未来仮説（multi-hypothesis、本命）

**モデル**（`mm/train_mh.py`）：
- rec_s20 のトランク（inp→blocks→collapse）
- K 本の onset ヘッド（Conv1d 1×1、T+30 の logit）。初期値は学習済み T+30 ヘッド＋小ノイズ。
- selector（リンク平均・最大プールした特徴 → K logits）

**損失**：
- ヘッドごとの損失：窓・ヘッドごとに `L_k = masked BCE + 0.5·Lovász`（T+30 のみ）。
- 勝者総取り（relaxed WTA）：argmin_k L_k に (1−ε)、他に ε/(K−1)（`--eps`=0.05）を掛けて和を取る。
- 崩壊防止：`--div`（0.1）× K 個の確率マップの平均ペア cosine 類似度を加える。
- selector：`--sel`（0.1）× CE(selector, 勝者)。
- ⑨b を使い続けるため、`--rec-w`（0.3）× masked reconstruction を加える。

**学習条件**：
- データ：train 月の onset 窓（8パネル、約2,300窓）、64リンククロップ。
- 学習率：トランク 1e-4、ヘッド 1e-3。`--freeze-trunk` でヘッドだけ学習する。
- 6,000 step。500 step ごとに、ES 窓（2031-01/02）での **oracle 損失**（mean_w min_k L_k）が最良の重みを保存する。

**評価**（`--eval`）：
- スコア月で ⑨b を再生しながら（トランクと rec を日次適応）、K 本の確率と selector 確率を保存する。
- 最初に見るのは、外し窓での次の2指標。
  - **best-of-K oracle IoU**
  - **top-1 hit の「K本のどれか」率**（T の top-1 hit 率と比較）
  - 例えば外し窓で 0.05 → 0.60 のように大きく上がれば、selection だけ解けばよい。
- 変わらなければ、この枝は即終了。
- 併記：`n_distinct_tops`（K本の1位ピークが何種類に分かれているか。1なら崩壊）、ヘッド別 IoU、selector が最良ヘッドを選ぶ率、selector ヘッドの IoU、π 混合、T-blend 版（0.7·H + 0.3·ヘッド）。
- 対照：同じレシピの **K=1**（`--K 1`）。K を増やしたことによる oracle の上昇と、単純な fine-tune 効果を分離するため。

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python mm/train_mh.py --name mh5 --K 5
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python mm/train_mh.py --name mh1 --K 1
CUDA_VISIBLE_DEVICES=3 ../.venv/bin/python mm/train_mh.py --name mh5_frz --K 5 --freeze-trunk
../.venv/bin/python mm/train_mh.py --name mh5 --K 5 --eval     # --K は学習時と同じにする
```

**振る設定**：K ∈ {3, 5, 8}、eps ∈ {0.02, 0.05, 0.1}、div ∈ {0, 0.1, 0.5}、lov ∈ {0, 0.5}。

**注意**：
- 学習ログのヘッド使用率（`head usage`）が1本に偏るなら崩壊している。div か eps を上げる。
- oracle は K を増やすだけで機械的に上がる（K 個のランダム予測でも上がる）。**必ず K=1 と、同じ K の「ランダム初期化ヘッド・学習なし」相当の感覚で比較する**こと。外し窓での top-1 hit の上昇が本質。

---

## ⑤ 未来状態の分布 → onset（MDN、未実装）

- 目的：平均表現ではなく、未来の分布そのものを学ぶ。
- target：上位10ボトルネック候補（または全リンクを 8 リンク平均に縮約）の、T+5〜T+30 の最小 speed ratio。約60次元。
- モデル：`p(Y|X) = Σ_k π_k N(μ_k, diag σ_k²)`、K=3〜5（対角共分散から始める）。トランクは rec_s20 を流用し、ヘッドだけ MDN にする。
- 教師：train のほぼ全時刻（onset 窓に限らない）。`candidates()` の全窓を使えるので、データ量が桁違いに増える。
- 推論：各成分からサンプルし、`v ≤ 0.6·vf` で渋滞化して、T+30 の onset 集合の分布を得る。
- 評価：④と同じ（成分ごとの oracle、π 最大成分、期待 IoU 最大化）。
- 着手条件：④で「K 本のどれかに正解が入る」ことが確認できた場合のみ。

## ⑥ non-local precursor 探索（未実装、優先度低）

1. flow residual（train の曜日×時刻プロファイルを引いたもの）について、各候補ボトルネックの onset（train の `ctx.eo`）と、10〜30分前の全リンクの residual との lagged 相関 / 相互情報量を計算する。
2. **validation と private で別々に計算し、上位 precursor の順位相関（split 間の安定性）を先に確認する**。安定しなければ即棄却。
3. 安定する上位5〜10本だけで、候補ごとの balanced LGBM を作る。
- 注意：precursor の値は、その窓の履歴12スロット内のもののみ使う（同日の他時刻は禁止）。

## ⑦ cross-panel 同時刻情報の合法性監査（実装不要、確認のみ）

**確認すること**：
1. Kaggle の Rules / Data ページと、`data/kaggle_public/task2/<panel>/<split>/` のファイル構造。
   - 公式の Task 2 入力（`window_history.parquet`）は、**そのパネルの mainline 履歴のみ**を含む。
   - 他パネルの T−60〜T−5 を使うことを明示的に許す／禁じる記述があるか。
2. 他パネルの同時刻データは、Task 1 の masked layer（corridors）として公開されている。ただしそれは「そのパネルの全日」であり、Task 2 の online 契約（そのウィンドウの履歴のみ）の外。
   - 私たちの内部ルール（ユーザー決定）は「同日の他時刻データは使わない」。
   - 他パネルの **同時刻より前** は時間的には未来ではないが、ウィンドウ契約の外。
3. そもそも有用か：パネル間で run の需要モードが共有されているかを、train の日次総流量・渋滞発生時刻のパネル間相関で見る。
   - 独立 seed / 独立シミュレーションなら、情報は無い。

**判定**：公式に明記された許可がない限り使わない（ユーザー判断事項）。有用性が無ければ、その時点で終了。

---

## 推奨順序

1. `diag_123.py`（①②③を一度に、GPU 1枚）
2. 並行して `train_mh.py` の K=5 / K=1 の学習（GPU 2枚）。診断の結果を待たずに回せる。
3. 診断で「正解が第2候補として見えている」なら、④の評価 → selection の設計 → ⑤。
4. どれも見えていなければ、「60分履歴にはレア onset の識別情報がほぼない」と結論し、architecture の枝を閉じる。

## ファイル

```
mm_lab/
  EXPERIMENTS_MM.md   本書
  mm/diag_123.py      ①②③（出力 out/<name>/per_window.csv, summary.txt）
  mm/train_mh.py      ④（学習：out/<name>/mh.pt, args.json、評価：eval.txt, eval_per_window.csv, probs_*.npz）
  out/                出力
```

- 依存：`tta_lab/tta/common.py`（窓、キャッシュ、⑨b 適応、IoU）
- 使うモデル：`work/models/t2_cnn_lov_rec_s20.pt`
- キャッシュ：`work/cache/t2probs_*`、`t2cnn_t2_cnn_lov_adapt_*` など（tta_lab の CODE_AND_DATA.md を参照）

新しいデータの追加は不要。
