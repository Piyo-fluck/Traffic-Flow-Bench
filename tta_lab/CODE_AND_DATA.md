# tta_lab：コードとデータの説明

置き場所：`/data1/nagao/Kaggle/TrafficFlowBench/tta_lab/`
実験内容は `EXPERIMENTS.md` を参照。

**ステータス**：コードは下書きで、動作確認はほとんどしていません。確認済みなのは、ローカルで `run_adapt.py` の full、および lora+micro+D損失を 1 パネル・数 step 回して例外が出ないところまでです。`run_F_gate.py` と `run_E_meta.py` は未実行です。A 案は `tta/run_A_t1res.py`（構文チェックのみ、未実行）。精査のうえ使ってください。

## 構成

```
tta_lab/
  EXPERIMENTS.md        実験 A〜F（G参考）の詳細説明
  CODE_AND_DATA.md      本書
  tta/
    common.py           共通部品（窓・キャッシュ読込、PEFT、損失、適応、評価、report）
    run_adapt.py        B / C / D / E評価 / F素材 を1本で回すランナー
    run_F_gate.py       F：自己教師あり損失ゲートのアンサンブル
    run_E_meta.py       E：メタ学習（FOMAML / 二次MAML）
    run_A_t1res.py      A：因果的 Task1 残差（global / 日次適応）を onset LightGBM に足す対比較
  assets/
    t1/m3v1_seed0_w64.pt      Task 1 M3 CNN（v1, seed0, width 64）   ← ローカル work/sub_m3v1/t1_m3_seed0.pt
    t1/m3v1_seed1_w64.pt      同 seed1                              ← work/sub_m3v1_s1/t1_m3_seed1.pt
    t1/m3w128_seed0_w128.pt   width 128                             ← work/sub_m3w128/t1_m3_seed0.pt
    cache/t2gs_{split}_{panel}.npz   global-state LightGBM の確率（Ada に無かったので追加）
  runs/                 出力（run ごとに probs_*.npz, args.json, report.txt）
```

## 既存プロジェクトとの関係（読み取り専用）

- `common.py` はプロジェクト root（tta_lab の親）の `analysis/`、`scripts/`、`tfb/` を import するだけで、**既存ファイルは一切書き換えない**。
- 出力は `tta_lab/runs/` にだけ書く。
- 依存モジュールがローカルと Ada で同一であることは md5（改行コード除去後）で確認済み：
  t2_eda02, t2_eda03, t2_expected_iou, t2_first_onset_eval, t2_m1_models, t2_m1_predict, t2_official_like_eval, t2_pseudo_eval, t2_prior_robust, t2_shift_remedies, t1_m3_cnn, tfb/physics, t1eval, features。
- `analysis/t2_cnn.py` は **Ada 版**（REC / PL / TEACHER などのフラグ付き）を使う。
  - `T2Net` は、import 時点で環境変数 `T2CNN_REC>0` のときだけ `rec` ヘッドを持つ。そのため common.py の冒頭で `T2CNN_REC=0.3`、`T2CNN_LOVASZ=0.5` を setdefault している。
- 実行は `tta_lab/` で `.venv` の python を使う：`cd tta_lab && ../.venv/bin/python tta/run_adapt.py ...`。tta/ 内のスクリプトは同じディレクトリの common を import する。

## 使う既存データ（Ada 上に既にあるもの）

| パス | 中身 | 用途 |
|---|---|---|
| `work/cache/{panel}_{split}.npz` | 日×時刻×リンクの dense 配列（tfb.data.load）。train は true 層も含む | 窓の構築、過去日バッチ |
| `work/cache/{panel}_train_ramps.npz` | ramp の接続情報 | 静的チャネル |
| `data/kaggle_public/corridors/*/network/` | リンク表・FD・トポロジ | Ctx（容量・vf・車線） |
| `data/kaggle_public/task2/{panel}/{split}/window_index.csv` | 公式窓の一覧 | `official_blank`（公式窓と重なる疑似窓を除外） |
| `work/cache/t2probs_{split}_{panel}.npz` | LightGBM H の確率（on_p (N,S), og_p (N,6,S)）と窓の日 | onset blend の 0.7 側、窓順の検証 |
| `work/cache/t2cnn_t2_cnn_lov{,_s1,_s2}_*` | CNN×3 | ongoing の REF |
| `work/cache/t2cnn_t2_cnn_lov_tfm_s{3..10}_*` | transformer×8 | ongoing blend の 0.60 側 |
| `work/cache/t2cnn_t2_cnn_lov_adapt_*` | ⑨b（候補Tで使った適応CNN） | REF |
| `work/models/t2_cnn_lov_rec_s{20,21,22}.pt` | 再構成ヘッド付き CNN | 全実験の出発点 |

- Ada に**無い**もの：公式窓の `window_history.parquet`（提出物の作成はローカルで行う）、`work/sub_m3*`（Task 1 の丸1日再構成。A ではリークのため使わない、EXPERIMENTS §0.5）。
- 窓数（各月）：onset 169 / 162、ongoing 3351 / 3374、8パネル。

## common.py の主な関数

| 名前 | 説明 |
|---|---|
| `windows(i, panel)` | `Ctx`、静的チャネル、`{split: {z, on, og, ref}}` を返す。on/og は `first_onsets` / `ongoing_official` の窓（d, t0, v, qf, ok, fut, efut）。ref に REF 用の確率を詰める。キャッシュと窓順が一致することを assert する |
| `make_x(ctx, stat, w, m, z)` | 窓の部分集合 → CNN 入力 (N,12,12,S) |
| `build_net(ckpt, mode, rank, train_rec)` | T2Net を読み込み、C のモード（full/last2/norm/film/lora/adapter/inp/incal）で学習対象を決める。(net, params) を返す |
| `snapshot / restore` | 学習対象パラメータの退避と復元（B, reset-daily 用） |
| `ssl_loss(net, xb, w_mask, w_next, w_delta, k_tail)` | D の損失 |
| `ssl_score(net, x)` | 固定マスク（seed 12345、4回）での窓ごとの再構成損失。F のゲート |
| `past_batch(ctx, stat, z, days_pool, bs, rng)` | 指定日集合（厳密に過去の日）からランダム時刻の履歴窓を作る。masked 層のみ使用 |
| `adapt(net, params, batches, steps, lr, opt, losses, k_tail)` | 自己教師あり適応。batches は呼ぶたびに新しいバッチを返す関数、または固定テンソル |
| `onset_iou / ongoing_iou` | 公式風 IoU（onset は ratio ルール、ongoing は τ） |
| `blends(ref, on_p, og_p)` | X 単体と「候補T の枠を X で置換」したブレンド |
| `reference(ref)` | 候補T の onset / ongoing（REF） |
| `report(runs)` | 保存済み run を REF と比べた表（IoU、family、family 差、窓単位の対差分 CI）。runs/<name>/report.txt に保存 |

## run_adapt.py の主な引数

- `--name`：出力名
- `--ckpt`：出発点（既定は rec_s20）
- `--mode`、`--rank`、`--train-rec`：C
- `--steps 30 --lr 1e-4 --bs 16 --opt adamw|sgd`：日次適応
- `--losses MASK NEXT DELTA`、`--k-tail`：D
- `--order predict-then-adapt`（⑨b と同じ）｜`adapt-then-predict`
- `--reset-daily`：毎日チェックポイントから再出発（E 評価用）
- `--window-days N`：直近 N 日だけで適応
- `--no-adapt`：対照（適応なし）
- `--micro K --micro-lr --micro-bs --micro-kinds on|og|both --micro-losses`：B
- `--kinds on|og|both`、`--panels`、`--splits`、`--seed`、`--no-report`

出力 `runs/<name>/probs_{split}_{panel}.npz` の中身：
- `on_p` (N,S)：T+30 の確率
- `og_p` (N,6,S)
- `on_ssl`、`og_ssl` (N,)：F のゲート用
- `on_d`、`og_d`：窓の日

## 実行例

```bash
cd /data1/nagao/Kaggle/TrafficFlowBench/tta_lab
PY=../.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY tta/run_adapt.py --name rep9b                            # ⑨b 再現
CUDA_VISIBLE_DEVICES=0 $PY tta/run_adapt.py --name ctrl_noadapt --no-adapt          # 適応なし
CUDA_VISIBLE_DEVICES=1 $PY tta/run_adapt.py --name C_norm_1e3 --mode norm --lr 1e-3
CUDA_VISIBLE_DEVICES=1 $PY tta/run_adapt.py --name C_incal_1e2 --mode incal --lr 1e-2
CUDA_VISIBLE_DEVICES=2 $PY tta/run_adapt.py --name D_nd_k2 --losses 1 0.5 0.5 --k-tail 2
CUDA_VISIBLE_DEVICES=3 $PY tta/run_adapt.py --name B_norm_k3 --mode norm --lr 1e-3 --micro 3 --micro-lr 1e-3 --kinds on
CUDA_VISIBLE_DEVICES=0 $PY tta/run_E_meta.py --name E_meta --steps 3000 --inner 5 --alpha 1e-3 --order 1 --aug 0.1
CUDA_VISIBLE_DEVICES=0 $PY tta/run_adapt.py --name E_eval --ckpt runs/E_meta/meta.pt --opt sgd --lr 1e-3 --steps 5 --reset-daily
$PY tta/run_F_gate.py --runs rep9b C_norm_1e3 D_nd_k2 --name F_gate
$PY -c "import sys; sys.path.insert(0,'tta'); import common as C; C.report(['rep9b','C_norm_1e3'])"
```

## 既知の注意点・未検証点

- `cnn.predict` は最後に `net.train()` を呼ぶ。T2Net は GroupNorm のみで dropout も無いので、挙動は変わらない。
- `report()` はパネルごとに Ctx を作り直すので、1回あたり数分かかる。
- `run_F_gate.py` は、全 run が同じ窓集合（`--kinds` が同じ）を持つことを前提にしている。
- `run_E_meta.py` の `--inner-scope norm` のパラメータ名抽出と、二次 MAML（`--order 2`）の autocast 下での動作は未確認。
- A 案：`t1_m3_cnn` は import 時に `M3_V2` などの環境変数を読む。未設定で v1（cin=10）になる。
