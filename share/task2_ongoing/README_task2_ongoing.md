# Task 2 ongoing の手法 — 再現用の説明

対象は、現在の最高提出 BB（LB 0.88697）の **Task 2 の ongoing 窓**の予測です。これは候補 AU で作ったものを
そのまま使っています。onset の窓、Task 1、Task 4 はこの文書の範囲外です。

このフォルダには、ongoing をゼロから作り直すのに必要なコードがすべて入っています。

- 実行の手順：`run_task2_ongoing.sh`
- 中身の説明：この文書

---

## 0. 準備

### フォルダ構成

このフォルダ（`task2_ongoing/`）を作業ディレクトリにして、次の 2 つを同じ階層に置いてください。

```text
task2_ongoing/
  data/kaggle_public/          コンペの配布データ（corridors/, task2/, submission_key.csv など）
  trafficflowbench-public/     運営のリポジトリ（提出ファイルの組み立てにだけ使う）
  analysis/   t2_cnn.py（CNN / Transformer の学習）、t2_gstate.py（gs1 の学習）ほか
  occ_lab/    train_occ.py / train_stid.py（補正器の特徴を出すモデル）、cache_oof.py / fit_corrector.py（補正器）
  scripts/    t2_pseudo_adapt_official.py（採点月への適応）、t2_gstate_probs.py、t2_residual_predict.py（最終予測）
              mean_prob_dirs.py（シード平均）、t2_patch_ongoing.py / swap_task_rows.py（提出ファイルへの差し込み）
  tta_lab/tta/common.py, tfb/   共通の部品
  run_task2_ongoing.sh
```

### 必要な環境

Python 3.12、torch（CUDA）、numpy、pandas、pyarrow、lightgbm、scipy を使います。

### 実行

```bash
STAGE=all GPU=0 bash run_task2_ongoing.sh        # STAGE は train / corrector / official / build / all
```

---

## 1. 全体の流れ

```text
[train]      train 期間のデータでモデルを学習する
               cnn3 / tfm8 / w192 SSL CNN ×4 / occ_cnn ×4 / occ_tfm ×4 / STID ×4 / gs1
[corrector]  train 期間の後半（2031-01〜02）で各モデルの予測を作り、補正器（LightGBM）を学習する
[official]   採点月（validation / private）の公式窓について、
               w192 SSL CNN ×4 と tfm8 を採点月に適応させて予測 → シードごとに平均。gs1 も予測
[build]      ブレンド → 補正器 → しきい値で 0/1 → ongoing_<split>.csv
             → 既存の提出ファイルの ongoing 行だけを差し替える
```

採点月の正解ラベルは一切使いません。採点月への適応に使うのは、その月の配布データのうち、**各窓の予測開始時刻
（origin）より前に公開されている値だけ**です（運営の 2026-09-20 の説明に沿っています）。

---

## 2. モデル（すべて train 期間で学習）

| 名前 | 中身 | 学習 | 保存先 |
|---|---|---|---|
| `cnn3` | `T2Net`（幅 64、link 軸方向の dilated conv） | `analysis/t2_cnn.py`、BCE + 0.5 Lovász、30k step、seed 0〜2 | `work/models/t2_cnn_lov{,_s1,_s2}.pt` |
| `tfm8` | `T2Tfm`（各 link の 12 スロットを GRU で 1 トークンに → link 方向の Transformer） | 同上で `T2CNN_ARCH=tfm`、seed 3〜10 | `work/models/t2_cnn_lov_tfm_s{3..10}.pt` |
| `p_PS` の元モデル | `T2Net` 幅 192 + 再構成ヘッド | **SSL**（マスク再構成 30k step）の後、教師あり 30k step。seed 20〜23 | `work/models/t2_cnn_lov_w192_rec_ssl30k_s{20..23}.pt` |
| `occ_cnn` / `occ_tfm` | 占有率 3ch を足した T2Net / T2Tfm | `occ_lab/train_occ.py`、seed 0〜3 | `occ_lab/models/occ_{cnn,tfm}_s*.pt`、`occ_profiles.npz` |
| `stid` | STID（リンク ID・時刻・曜日の埋め込み + MLP） | `occ_lab/train_stid.py`、seed 0〜3 | `occ_lab/models/stid_s*.pt` |
| `gs1` | LightGBM + パネル全体の状態特徴（PCA など 12 個） | `analysis/t2_gstate.py` | `work/models_gstate/` |

**モデルの入力（ニューラル系）：** 窓の履歴 12 スロット（60 分）の速度・流量・観測の有無と、固定のチャンネル
（車線数、容量、自由流速度、ランプの有無、時刻）を (チャンネル, 12, link) の形にしたものです
（`analysis/t2_cnn.py:make_x`）。

**出力：** 6 ステップ先（+5〜+30 分）の各 link の渋滞確率です。

---

## 3. 採点月への適応（`scripts/t2_pseudo_adapt_official.py`）

| | `p_PS`（w192 SSL CNN ×4） | 適応版 `tfm8`（×8） |
|---|---|---|
| 損失 | 教師あり（BCE + Lovász）+ 再構成（重み 1.0） | 教師ありのみ（`T2CNN_REC` を外して実行） |
| 設定 | 30 step / 日、lr 1e-4、バッチ 16 | 同じ（`--arch tfm`） |

- 教師ありのラベルは、**その日の最初の公式 origin より前に収まる**疑似窓を、配布データから作ります。
- 見えているセルでは、配布値がそのまま正解です。見えていないセルは損失から外します。
- 重みは日をまたいで持ち越します（因果的）。
- 出力はシードごとに `work/sub_t2pseudo_ssl_s*`、`work/sub_t2tfmps_s*` に保存し、`scripts/mean_prob_dirs.py` で
  平均して `sub_t2pseudo_ssl4`、`sub_t2tfmps8` にします。

---

## 4. ブレンドと補正（`scripts/t2_residual_predict.py`）

```text
old = 0.20 * gs1 + 0.60 * tfm8 + 0.20 * cnn3           （tfm8 は適応版 = sub_t2tfmps8）
p0  = 0.40 * old + 0.60 * p_PS                         （p_PS = sub_t2pseudo_ssl4、--w-pseudo 0.60）

r_hat = corrector(15 特徴)                              （occ_lab/models/residual_lgb.txt）
p     = clip(p0 + 0.75 * r_hat * 1[|r_hat| >= 0.10], 0, 1)  （--eta 0.75 --gate 0.10）
queue_pred = 1[p >= 0.50]
```

**補正器の 15 特徴**（`occ_lab/residual_diag.py:NAMES`、順序固定）は次のとおりです。

- p0 とその logit、|p0 − 0.5|
- 各モデルのばらつき（標準偏差、幅）
- gs1、tfm8、cnn3、occ_cnn、occ_tfm、stid の各予測
- logit(stid) − logit(p0)、logit(occ_cnn) − logit(p0)
- horizon、位置

**tfm8 の使い分け（`--tfm-feat frozen`）：** ブレンドには適応版を使います。補正器の特徴には、補正器を学習した
ときと同じ**適応前**の tfm8 を渡します。

**補正器の学習（`occ_lab/cache_oof.py` → `fit_corrector.py`）：** train 期間の後半（2031-01〜02、
`FIT_END` 以降）の窓で、各モデルの予測と正解の残差を LightGBM で回帰します。この期間はモデルの学習に使っていないので
予測は out-of-fold になり、しかも両方の採点月より前の期間です。

---

## 5. 提出ファイルへの差し込み

`work/sub_t2ongoing/ongoing_<split>.csv` には ongoing 窓の行だけが入っています。これを既存の提出ファイル
（例：BB）に差し込むと、onset の行や他のタスクの行はバイト単位でそのまま残ります。

```bash
python scripts/t2_patch_ongoing.py --base <既存の submission.csv> \
  --ongoing work/sub_t2ongoing/ongoing_validation.csv work/sub_t2ongoing/ongoing_private.csv --out-dir work/sub_t2ongoing_q
python scripts/swap_task_rows.py --base <既存の submission.csv> --task queue \
  --files work/sub_t2ongoing_q/queue_validation.csv work/sub_t2ongoing_q/queue_private.csv --out <新しい submission.csv>
```

---

## 6. 計算時間の目安

| 段階 | 処理 | 時間（RTX 6000 Ada、GPU 1 枚） |
|---|---|---|
| train | cnn3 / tfm8（30k step） | 1 本 12〜17 分 |
| train | w192 SSL（SSL 30k + 教師あり 30k） | 1 本 約 33 分（4 本を並列で回したとき） |
| official | 採点月への適応 | 1 seed × 2 か月で数分 |

複数の GPU があれば、train と official の各ループは seed ごとに並列にできます。

---

## 7. 再現で間違えやすい点

- **環境変数の付け忘れ：** `T2CNN_*` の値でモデルの構造とファイル名が決まります（`_lov`、`_tfm`、`_w192`、
  `_rec`、`_ssl30k`、`_s<seed>`）。付け忘れると、別の名前・別の構造で保存されます。
- **tfm8 の適応は再構成ヘッドなし：** `env -u T2CNN_REC` で実行します。`p_PS` のほうは `T2CNN_REC=0.3` が必要です。
- **`analysis/t2_cnn.py` の版：** このフォルダの版は、SSL 事前学習（`T2CNN_SSL`）を含む版です。
- **未使用の選択肢：** `t2_residual_predict.py` と `t2_pseudo_adapt_official.py` には、使わなかった選択肢
  （`--select`、`--solid3`、`--stid-dir` など）も残っています。既定では無効で、上の手順では使いません。
- **乱数による差：** モデルを学習し直すと乱数の影響で予測が少し変わり、BB の ongoing 行とは完全には一致しません。
  学習済みモデルからやり直す場合は、`STAGE=official` から始めてください。
