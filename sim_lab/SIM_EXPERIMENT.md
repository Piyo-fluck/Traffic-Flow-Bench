# 案1：合成シミュレーション run による Task 2 の事前学習（sim-to-real）

置き場所：`/data1/nagao/Kaggle/TrafficFlowBench/sim_lab/`
依存：プロジェクト本体の `analysis/`・`scripts/`・`tfb/`（読み取りのみ）、`tta_lab/tta/`（評価用）
現ベスト：Z（Public LB 0.87871）＝ T ＋ Task 1 のテスト時学習 3600 step。Task 2 部分は T と同じ。

コードの状態：
- 生成器 `ctm_gen.py` と較正チェック `check_sim.py` は、ローカルで1パネル・小規模に動作確認済み。
- 学習スクリプト `train_sim.py` は構文チェックのみで、未実行です。

---

## 0. 何を解決したいのか

### 0.1 これまでの失敗の共通原因

- **train は1つのシミュレーション run しかない**。validation / private は、需要の抽選も incident も異なる別の run（DATA.md：「five incidents in validation, seven in private」）。
- どのモデルも、train run の「いつも壊れるボトルネック」を学ぶ。onset の失敗は「T の上位3候補に正解が入っているのに、train run の常連を自信満々に選ぶ」形（外し窓の 85% で、正解は T の上位3候補に入っている）。
- 表現を変える案（Task 1 移植、対照学習）、後段で選び直す案（再順位付け、ゲート、多仮説ヘッド、ヘッジ）は、いずれも run が1本のまま。だから効かなかった。
- ⑨b（過去日の自己教師あり適応）だけが効いた。「run に合わせて重みを変える」ことには価値がある、という証拠。

### 0.2 この案の考え方

**学習データの run そのものを増やす。** 自作の交通シミュレータで、同じ回廊の run を数百本作る。需要・ランプ・容量の揺らぎ・incident をランダムに変えるので、どのボトルネックが壊れるかも run ごとに変わる。その上で Task 2 の CNN を事前学習し、最後に本物の train run で微調整する。

- 狙い：「この状態ならどのボトルネックが壊れるか」を、run に依存しない形で学ばせる。
- 直接の CTM 予測（note 21）は、需要が1%ずれるだけで結果が変わり失敗した（IoU ≤ 0.10）。ここでは CTM を**予測器としては使わず、多様な訓練データの生成器として使う**。需要への感度の高さは、むしろ run の多様性を生む材料になる。
- 使える物理情報の裏付け：
  - 渋滞の先頭から捌ける流量（discharge）は run 間で 0.5% 以内で一致する（note 21）。
  - FD パラメータ（fd_parameters）・経路・ランプ接続は公開されている。
  - ランプ流量の収支は 1.1% の誤差で再現できている。

### 0.3 ルール（2026-09-19 に Kaggle Rules ページで確認済み）

- Rules の記述："Modeling the traffic physics that generated the data is not prohibited. It is the point of the benchmark."
- 守る条件：
  1. **生成に使うのは公開データと train から推定したものだけ。** validation / private の値で分布を調整しない。
  2. **主催者の scored run（seed・incident の日時と場所）を再現しようとしない。**
  3. **スコア月は評価にだけ使う。**
- 受賞時にはコードの再現性が求められる。乱数 seed はすべて引数で固定される（`--seed`）。

---

## 1. 生成器 `sim/ctm_gen.py`

**物理モデル**（`analysis/t2_ctm_sim.py` の Net を継承）：
- リンクごとの三角形 FD。自由流速度 vf・容量 cap・渋滞密度 kjam は fd_parameters から取る。
- Godunov / CTM を 6秒刻みで回す（5分スロットあたり 50 step）。短いリンクは CFL 条件を満たすよう延長する。
- 合流は Daganzo 則。本線の優先比は 車線数/(車線数+1)。
- オフランプは train で推定した固定の分岐率で抜ける。オンランプは train の曜日×時刻プロファイルで流入する。
- 上流端とオンランプには点待ち行列（入れなかった車の溜まり）を置く。
- 上流端の需要は、Task 1 の曜日×時刻の流量表（`assets/{panel}_t1tables.npz` の tq[:, :, 0]）。

**容量**（`--cap-mode`）：
- `discharge`（既定）：train の真値で「リンク s が自由流、s−1 が渋滞」のときの s の流量の 90 パーセンタイルを使う（該当セル ≥ 50 のリンクのみ。`--discharge-pct`、`--discharge-min-n`）。それ以外のリンクは公開容量。
  - 中央値だと、渋滞解消後の低い流量まで拾って容量を過小評価し、偽のボトルネックができた（ローカルで確認）。そのため高パーセンタイルにしている。
- `fd`：公開容量のみ。

**ランダム化**（すべて train 由来のプロファイルに対する乗数）：

| 階層 | 対象 | 引数（既定値） |
|---|---|---|
| run | 全体の需要倍率（対数正規） | `--run-scale 0.04` |
| run | オンランプごとの倍率 | `--run-ramp 0.08` |
| run | ピーク時刻のずれ（±分） | `--shift 15` |
| run | 容量のジッタ | `--cap-jitter 0.005` |
| day | 全体の倍率 | `--day-scale 0.03` |
| day | オンランプごとの倍率 | `--day-ramp 0.05` |
| 時刻 | ランプと上流端の AR(1) ノイズ | `--ar-sigma 0.05`、`--ar-minutes 30` |
| incident | 1日あたりの発生率（ポアソン） | `--incidents-per-day 0.05` |
| incident | 容量の低下率、継続時間 | 低下率 `--inc-min 0.3`〜`--inc-max 0.6`、継続 15〜60分（5〜20時台に発生） |
| 観測 | 速度・流量の乗法ノイズ | `--noise-v 0.02`、`--noise-q 0.03` |
| 観測 | eligible マスク | train のランダムな1日のものを流用 |

**出力**：`<out>/<panel>/run_<k>.npz`
- キーは split と同じ形：dates、weekday、true_speed、true_flow、elig、free_speed、link_ids、lanes。それに meta（run のパラメータと incident 一覧の JSON）。
- 既存の Task 2 コード（labels、earliest_onset、candidates、make_x）がそのまま読める。

**速度の目安**：1 run（30日）あたり数秒（numpy のベクトル化）。8パネル × 40 run で数分。

```bash
cd /data1/nagao/Kaggle/TrafficFlowBench
PY=.venv/bin/python
$PY sim_lab/sim/ctm_gen.py --out sim_lab/data --runs 40 --days 30             # 全8パネル
$PY sim_lab/sim/ctm_gen.py --out sim_lab/data_smoke --runs 2 --days 5 --panels D7_I10_E
```

---

## 2. 較正チェック `sim/check_sim.py`（最初の関門）

**合成 run が train run に「似ていて、かつ run ごとに違う」ことを確認する。** ここを通らないまま学習に進んではいけない。

```bash
$PY sim_lab/sim/check_sim.py --data sim_lab/data
```

| 指標 | 意味 | 目標の目安 |
|---|---|---|
| share tr/sim | 渋滞セルの割合 | 合成が train の 0.5〜2倍 |
| map corr mean/med | リンクごとの渋滞頻度の、train との相関（run 平均の地図 / run ごとの中央値） | 平均地図で ≥ 0.5 |
| top8 | 渋滞の多い上位8リンクの重なり | ≥ 3 |
| diversity | run 間の地図の平均相関 | 0.3〜0.8（1.0 は全部同じで無意味、0 近くは run ごとにバラバラすぎ） |
| onset h tr/sim | 朝の最初の渋滞発生時刻の中央値（時） | 差 ±1時間以内 |
| onsets/day tr/sim | 1日あたりの onset 窓の数（学習信号の量） | 合成が train の 0.5〜2倍 |
| flow ratio | 平均流量の比（合成 / train） | 0.95〜1.05 |

**ローカルでの試走結果**（D7_I10_E、2 run × 5日、既定値）：

```
share 0.031/0.010  map corr 0.53/0.33  top8 0  diversity 0.25  onset h 11.1/12.0  onsets/day 1.01/0.70  flow 1.01
```

- 流量水準と地図の相関はそれなりに合っている。
- 一方で、渋滞がやや少ない、上位8リンクが合わない、run 間の多様性が大きすぎる。
- **既定値は第一版にすぎない。まず較正を詰めること。**

調整のつまみ：
- 渋滞が少ない → `--run-scale` / `--day-scale` を上げるか、`--discharge-pct` を下げる。
- 場所が合わない → `--cap-mode fd` と比べる。discharge 推定の min_n / pct を変える。上位リンクの容量を実測と照合する。
- 多様性が大きすぎる → `--run-ramp` を下げる。

**それでも合わない場合の拡張候補**（未実装）：
- 容量の自動校正：train の日ごとの需要を与えて CTM を回し、リンクごとの渋滞頻度が train と合うよう容量を反復で調整する。
- ランプ需要を、train の日ごとの実測ランプ流量から直接サンプリングする。

---

## 3. 学習 `sim/train_sim.py`

**共通の設定**：
- ネットワーク・入力・ラベル・損失は、Ada 版 `analysis/t2_cnn.py` と同じ。
- 環境変数 `T2CNN_REC=0.3`、`T2CNN_LOVASZ=0.5`。損失は BCE ＋ 0.5 Lovász ＋ 0.3 マスク再構成。
- 64 リンククロップ、バッチ 32、AdamW ＋ OneCycle。
- 合成 run の学習窓は本物と同じ作り方にする（全 earliest-onset 窓 ＋ ongoing 候補）。ただし ongoing は run あたり最大 `--syn-max-og 500`（メモリ約 13MB/run）。
- 静的なリンク特徴チャネルは、本物のパネルのものを使う（ネットワークは同じ）。
- early stopping は、**すべてのモードで本物の train の 2031-01/02 窓の BCE** で行う。train 月のラベルなので、ルール上の問題はない。
- 出力 `sim_lab/out/<name>/model.pt` は rec ヘッド付きなので、⑨b の適応評価にそのまま使える。

**モード**：

| mode | 内容 | 役割 |
|---|---|---|
| `real_only` | 本物のみ、`--steps-pre + --steps-ft` step | **対照**。同じコードで rec_s20 相当のモデルを作る |
| `synth_only` | 合成のみ | **ゼロショット判定**：合成だけで本物にどこまで効くか |
| `pretrain_finetune` | 合成で `--steps-pre`（lr 1e-3）→ 本物で `--steps-ft`（lr 3e-4、学習率スケジュールはやり直し） | 本命 |
| `mix` | 毎 step、確率 `--real-share` で本物、それ以外は合成のバッチ | 代替 |

```bash
CUDA_VISIBLE_DEVICES=0 $PY sim_lab/sim/train_sim.py --name ctrl_real   --mode real_only
CUDA_VISIBLE_DEVICES=1 $PY sim_lab/sim/train_sim.py --name zs_synth    --mode synth_only
CUDA_VISIBLE_DEVICES=2 $PY sim_lab/sim/train_sim.py --name pf_20k10k   --mode pretrain_finetune
CUDA_VISIBLE_DEVICES=3 $PY sim_lab/sim/train_sim.py --name mix_50      --mode mix --real-share 0.5
```

- 所要時間の目安：30k step で 1〜2時間程度（rec_s20 の学習と同程度）。
- メモリ：本物の窓で約 3.5GB。合成は 40 run × 8 パネルで約 4GB。

---

## 4. 評価（`tta_lab` の仕組みをそのまま使う）

```bash
cd tta_lab
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tta/run_adapt.py --name sim_pf      --ckpt ../sim_lab/out/pf_20k10k/model.pt
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python tta/run_adapt.py --name sim_pf_noad --ckpt ../sim_lab/out/pf_20k10k/model.pt --no-adapt
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python tta/run_adapt.py --name sim_ctrl    --ckpt ../sim_lab/out/ctrl_real/model.pt
../.venv/bin/python -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['sim_ctrl','sim_pf','sim_pf_noad'])"
```

- `run_adapt.py` は ⑨b（過去日適応）を再生しながら、公式に近い窓（onset：val 169 / priv 162、ongoing：3351 / 3374）で評価する。
- 候補T の枠（onset は 0.7×LGBM H ＋ 0.3×CNN、ongoing は 0.20 gs1 ＋ 0.60 tfm8 ＋ 0.20 CNN、τ 0.50）に差し込んだときの、REF（候補T）との差と、窓単位の 95% CI を出す。
- **比較は `sim_ctrl`（同じコードの本物のみ）に対して行う。** REF との比較だけだと、学習の乱数差と区別できない。

**判定基準**（事前に固定）：

| 段階 | 条件 | 判定 |
|---|---|---|
| 較正 | 第2章の目標をおおむね満たす | 満たさなければ学習に進まない |
| ゼロショット | `synth_only` の onset 単体 IoU が `ctrl_real` の 7割以上 | 未満ならシミュレータと本物の差が大きすぎる → 較正に戻るか終了 |
| 本命 | `pretrain_finetune` が `ctrl_real` に対し、onset T-blend family で**両月とも +0.01 以上** | 提出候補化へ |
| 失敗の中身 | 外し窓（T の IoU=0）での top-1 的中率（`mm_lab/mm/diag_123.py` の rank 集計を流用） | 常連バイアスが減ったかを確認 |

- onset の判定の微調整は、手元で両月プラスでも LB に転移しなかった（U、X）。この案は「モデルの重み」を変えるので ⑨b と同じ種類の変更だが、**窓数が少ない点は変わらない。** 最終判断は LB で行う。

---

## 5. ファイル

```
sim_lab/
  SIM_EXPERIMENT.md        本書
  sim/ctm_gen.py           合成 run の生成
  sim/check_sim.py         較正チェック
  sim/train_sim.py         事前学習・微調整・対照
  assets/{panel}_t1tables.npz   Task 1 の曜日×時刻の平均速度・流量表（Ada のキャッシュに無いので同梱、10パネル計 25MB）
  data/                    生成データの出力先
  out/                     学習済みモデルの出力先
```

- 既存の読み込み先（読み取りのみ）：
  - `work/cache/{panel}_train.npz`、`{panel}_train_ramps.npz`
  - `data/kaggle_public/corridors/*/network/`
  - 評価時は `tta_lab` が使うキャッシュ一式（`tta_lab/CODE_AND_DATA.md` を参照）
- 既存ファイルは一切変更しない。`analysis/t2_ctm_sim.py` は import するだけ（CACHE の参照先を `sim_lab/assets` に差し替えるのは、import したモジュール変数への代入のみ）。

## 6. 既知の限界と注意点

- **シミュレータと本物の差**：主催者の生成器（DLSim4GMNS、note 21）とは実装の細部が違う。例えば合流則、incident の表現、センサーの集計方法。
  - 較正が不十分なまま学習すると、「合成 run でよく壊れる場所」という新しい偏りを学んでしまう。
  - `diversity` と `map corr` の両方を見ること。
- **incident**：主催者の incident は1か月あたり 5〜7件と少ない。既定値の 0.05/日（30日で約1.5件/パネル）はやや多め。incident が onset の主因でないことは確認済み（note 73）なので、多様性を足す程度の扱い。
- **観測の欠け**：eligible マスクは train から流用しているが、Task 2 の履歴で見えないセル（ホライズン側）は再現していない。ラベルは合成の真値から作る。
- **メモリ**：`--runs` と `--syn-max-og` で調整する。
