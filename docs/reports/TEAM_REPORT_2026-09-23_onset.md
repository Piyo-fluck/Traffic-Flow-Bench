# Task 2 onset：主催者アナウンスを受けた作り直しと、新ベスト 0.88153

作成日：2026-09-23。対象：Task 2 の `queue_onset` 半分のみ（`queue_ongoing` は今回の対象外）。

---

## 1. 結論

- **Public LB 0.88153（提出 AN）が新ベスト**。直前のベスト 0.87902 から **+0.00251**。
- 変更点は **onset の確率ブレンドにメンバーを 1 つ足しただけ**。提出ファイルの差分は queue 行 **22 行**のみで、Task 1・3・4 と ongoing 側は一切触っていない。
- 足したのは **onset 専用の STID（リンク識別子埋め込み＋MLP）**。既存 2 メンバーにない「どのリンクが落ちやすいか」をリンク ID として持つモデル。

  | | 変更前（従来ベスト） | 変更後（AN） |
  |---|---|---|
  | onset 確率 | 0.70 × LightGBM + 0.30 × 適応 CNN | **0.35 × LightGBM + 0.35 × 適応 CNN + 0.30 × onset STID** |
  | 予測時刻 | T+30 のみ | 同じ |
  | 決定規則 | 期待 IoU の ratio ルール（c=1.0） | 同じ |

---

## 2. 前提：主催者アナウンスの内容

2026-09-23、主催者から全参加者向けに公式アナウンスが出ました（原文は `notes/onset_discussion_2026-09-23.txt`）。

- `queue_onset` の窓は、**「30 分以内に渋滞が入る最も早い起点」**を選ぶ方式で作られている。
- したがって実際の渋滞開始時刻を t とすると、選ばれる起点は必ず **T = t − 30 分**。
- 結果として、**全 120 窓すべてで渋滞は T+30 にだけ出現**し、T+5〜T+25 は空。train / validation / private の 3 分割すべてで確認済み。
- **これを使ってよい**。データも評価指標も変更しない。`window_index.csv` の `condition` 列で条件分岐してよい。

つまり onset は「いつ渋滞が始まるか」を当てる問題ではなく、**T+30 でどのリンクが渋滞するかを当てる空間問題**です。

主催者はもう 1 点、重要な注意をしています。**6 ステップに薄く予測を広げるのは IoU 上きわめて高くつく**。正解が T+30 の 3 セルのとき、同じ 3 セルを 6 ステップ全部に置くと IoU は 3/18 ≈ 0.17 まで落ちます。

---

## 3. 我々のパイプラインの状態（アナウンス前から満たしていた点）

確認したところ、アナウンスの主要な帰結は**すでに実装済み**でした。

- `scripts/t2_cnn_blend_predict.py` の onset 処理は `pred[j, F-1]`（T+30）にしか書き込まず、他の 5 ステップはゼロのまま。
- 確率も T+30 の 1 枚しか参照していない。
- 決定規則（期待 IoU の ratio ルール）も T+30 のリンク確率に対してのみ適用。
- 学習窓も `earliest_onset`（履歴に渋滞なし・T+30 で初出）で、**主催者のセレクタ条件と一致**。

したがって「6 ステップに広げて IoU を落とす」失敗は最初から起きておらず、**アナウンスから直接得られる追加の取り分はありませんでした**。

---

## 4. 残っていた論点と、2 つの実験

### 4-1. 損失が使わない 5 ステップに向いていた（効果なし）

onset 窓に対しても損失は 6 ホライズン全部にかかっていました。構造上ゼロと分かっている 5 枚に容量の 5/6 を使っている計算です。そこで **T+30 の 1 枚だけを学習する onset 専用モデル**を作りました。

単体の onset IoU（公式相当窓、家族平均）:

| モデル | validation | private |
|---|---|---|
| **T+30 単独学習（4 seed）** | **0.7504** | **0.7347** |
| ＋過去日適応 | **0.7912** | 0.7337 |
| 従来の適応 CNN（ブレンドの CNN 枠） | 0.7412 | 0.6787 |
| 従来 CNN×3 | 0.6835 | 0.6585 |
| LightGBM | 0.7652 | 0.7981 |

**単体では大きく改善**し、過去日適応を重ねると validation 0.7912 と、従来のブレンド全体（0.7862）を単体で上回りました。

ところが**ブレンドに入れるとすべて負けました**（両月で −0.002〜−0.016）。従来の適応 CNN が 0.3 の重みに値するのは自身の精度ではなく「LightGBM とは違う間違え方をする」からで、同じ CNN から fine-tune したモデルは精度が高くても補完性がない、という結果です。

### 4-2. 異種アーキテクチャ（採用）

onset が「どのリンクが落ちるか」に還元された以上、**リンク固有の性質**を直接持てるモデルが筋が良い、という判断で STID 系（Shao et al. 2022 の Spatial-Temporal IDentity）を onset 専用に作りました。

```
入力  リンクごとの 15ch × 12 スロットの履歴（占有率を含む）を平坦化 → 96 次元へ射影
      ＋ リンク識別子埋め込み（全 8 パネル通算 1,450 リンク）
      ＋ 時刻埋め込み（288 スロット）＋ 曜日埋め込み（各 32 次元）
本体  残差 MLP 3 ブロック（リンク間の混合なし＝空間は識別子が担う）
出力  T+30 の 1 枚のみ
学習  onset 窓のみ、BCE + 0.5 Lovász、8,000 step、4 seed、パラメータ 29.6 万
```

既存 2 メンバーはどちらもリンク識別子を持ちません（LightGBM は手作り特徴のみ、CNN は回廊方向に並進不変）。ここが唯一の「持っていない表現」でした。

---

## 5. 重み探索の結果

onset 家族 IoU の改善幅（従来ブレンド 0.7862 / 0.8227 に対して）:

| STID の重み | 最良の H / 適応 CNN | validation | private |
|---|---|---|---|
| 0.00（重み変更のみ） | 0.65 / 0.35 | +0.0044 | +0.0034 |
| 0.05 | 0.55 / 0.40 | +0.0205 | +0.0199 |
| 0.15 | 0.45 / 0.40 | +0.0233 | +0.0222 |
| **0.30（採用）** | **0.35 / 0.35** | **+0.0230** | **+0.0337** |
| 0.45 | 0.15 / 0.40 | +0.0356 | +0.0260 |
| 0.60 | 0.05 / 0.35 | +0.0237 | +0.0287 |

- **195 通り中 86 通りが両月で従来を上回りました。**
- **効果は STID 由来です。** 既存 2 メンバーの重みを振り直すだけでは最良でも +0.0044 / +0.0034 にとどまり、STID を 0.05 入れた時点で +0.020 に跳ねます。
- 0.10〜0.60 の広い範囲が +0.022〜+0.034 の台地になっており、特定の 1 点に依存していません。採用したのは台地の中央です。

---

## 6. 提出と検証

- 候補 AN ＝ 従来ベストの onset 行だけを差し替えたもの。ongoing 行・Task 1・Task 3・Task 4 は同一。
- 検証として、**同じスクリプトで従来の重み（0.70 / 0.30 / 0）を再現したところ、従来ベストと 0 行差**でした。比較が公平であることを確認しています。
- 手元の見積もり +0.023 / +0.034（総合換算 +0.0035〜+0.0050）に対し、LB は **+0.00251**。実現率は約 0.6 で、過去の校正（0.44〜0.76）の範囲内です。

---

## 7. 再現手順

```bash
# 1) onset STID の学習（4 seed、GPU 1 枚あたり約 3 分）
bash occ_lab/run_onset_stid.sh

# 2) 公式窓の onset 決定を作る
.venv/Scripts/python.exe scripts/t2_onset_blend_predict.py \
  --out-dir work/sub_t2onsetstid \
  --h-dir work/sub_t2robust_bag5 --ad-dir work/sub_t2adaptcnn \
  --stid occ_lab/models/onset_stid_s0.pt occ_lab/models/onset_stid_s1.pt \
         occ_lab/models/onset_stid_s2.pt occ_lab/models/onset_stid_s3.pt \
  --profiles work/models_remote/occ_profiles.npz \
  --w-h 0.35 --w-ad 0.35 --w-st 0.30

# 3) 既存の提出物の onset 行だけを差し替える
.venv/Scripts/python.exe scripts/t2_patch_ongoing.py \
  --base <既存の submission.csv> \
  --ongoing work/sub_t2onsetstid/onset_validation.csv work/sub_t2onsetstid/onset_private.csv \
  --out-dir work/sub_t2AN
.venv/Scripts/python.exe scripts/swap_task_rows.py --base <既存の submission.csv> --task queue \
  --files work/sub_t2AN/queue_validation.csv work/sub_t2AN/queue_private.csv \
  --out work/sub_AN_onsetstid/submission.csv
```

主なコード：`occ_lab/train_onset_stid.py`（学習）、`occ_lab/predict_onset_stid.py`・`occ_lab/onset_weight_sweep.py`（評価と重み探索）、`scripts/t2_onset_blend_predict.py`（公式窓の決定）。詳細は `notes/92_onset_single_horizon.md`、`notes/93_onset_stid.md`。

---

## 8. この先の見込み

- **STID のシード追加**：現在 4 seed。ニューラル系はシード平均で安定して伸びてきた経緯があり、8 seed まで増やす余地があります。
- **重みの別の点**：台地の反対側（STID 0.45、H 0.15）は validation で +0.0356 と最大です。片方の月に寄っているため中央を採りましたが、確認の価値はあります。
- **ongoing 側への展開**：ongoing にも STID を入れてありますが、6 ホライズン出力のままで onset 版ほど作り込んでいません。onset で識別子が効いた以上、ongoing 側も同じ方向で見直す価値があります。
