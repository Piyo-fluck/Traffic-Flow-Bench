# IEEE Big Data 2026 Traffic Flow Bench — 関連文献メモ

更新日: **2026-09-14**

> 目的: Traffic Flow Bench に個人で取り組むための、短い先行研究メモ。  
> 引用数はデータベースごとに集計方法が違うため、**確認したデータベース名も併記**する。プレプリントと出版版が別集計される論文では特に差が出る。

## 0. コンペとの対応

現行の主データは **California / Los Angeles 周辺**で、I-5・I-10・I-110・I-210・I-405上の約150検知器、5分間隔、2025年通年データが中心。**Shenzhen は追加地域として prospective（予定）**とされている。

- **TrafficStateBench**: 欠損センサー復元、将来予測、Fundamental Diagram (FD) 推定
- **ODMEBench**: 動的 OD demand 推定、需要と渋滞原因の対応付け
- **ShockwaveBench**: 渋滞開始/継続、queue位置、shockwave追跡

公式: https://bigdataieee.org/BigData2026/cup/

---

## 1. 基本文献

| 文献 | 要点 | 対応 | 引用数（確認時） |
|---|---|---|---:|
| **LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting** (NeurIPS 2023) | California PeMS を用いた大規模予測ベンチマーク。8,600センサー・5年間。 | forecasting | **120** (Scopus) |
| **DCRNN: Diffusion Convolutional Recurrent Neural Network** (ICLR 2018) | 道路を有向グラフ上の拡散過程として扱い、RNNで予測。METR-LA / PEMS-BAYの古典。 | forecasting baseline | **1,705** (SciSpace) |
| **Graph WaveNet** (IJCAI 2019) | adaptive adjacency と dilated temporal convolution。固定道路グラフだけに依存しない。 | forecasting baseline | **1,599** (SciSpace) |
| **IGNNK: Inductive Graph Neural Networks for Spatiotemporal Kriging** (AAAI 2021) | 未観測地点を周辺センサーから推定する inductive GNN。未知地点への転移を想定。 | spatial imputation | **196** (Semantic Scholar) |
| **Physics-Informed Deep Learning for Traffic State Estimation** (AAAI 2021) | NNに second-order traffic model を正則化として組み込み、NGSIMで検証。 | physics / TSE | **64** (SciSpace) |
| **Dynamic OD Demand Flow Estimation under Congested Traffic Conditions** (TR-C 2013) | 渋滞下の時変OD推定。DUEとkinematic-wave系の考え方を統合。Xuesong Zhou共著。 | ODMEBench | **104** (Scopus) |
| **A Data-Driven Traffic Shockwave Speed Detection Approach Based on Vehicle Trajectories Data** (JITS 2024) | FFT、turning point、DBSCAN等から部分軌跡だけでshockwave速度を推定。 | ShockwaveBench | **10** (Crossref) |
| **Investigation of Bimodal Macroscopic Fundamental Diagrams ... Shenzhen City** (TRR 2019) | Shenzhenの車・バスGPSからMFD/3D-MFDを実証分析。 | Shenzhen / FD解析 | **36** (Scopus) |
| **Traffic Control Approach Based on Multi-Source Data Fusion** (IET ITS 2019) | Shenzhenで携帯位置情報 + taxi GPSを融合し、需要源・bottleneck・制御を接続。 | demand / simulation | **21** (ResearchGate) |
| **CityFlow** (WWW 2019) | 大規模都市ネットワーク向け交通シミュレータ。RL・信号制御向け。 | simulation | **245** (ACM) |
| **Flow: A Modular Learning Framework for Mixed Autonomy Traffic** (IEEE T-RO 2022) | SUMO等とRLを結び、stop-and-go waveや混合交通制御を扱う。 | simulation / control | **107** (SciSpace) |

### 一次資料

- LargeST: https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html
- DCRNN: https://arxiv.org/abs/1707.01926
- Graph WaveNet: https://doi.org/10.24963/ijcai.2019/264
- IGNNK: https://doi.org/10.1609/aaai.v35i5.16575
- Physics-Informed TSE (2021): https://doi.org/10.1609/aaai.v35i1.16132
- Dynamic OD estimation: https://doi.org/10.1016/j.trc.2013.05.006
- Shockwave detection: https://doi.org/10.1080/15472450.2023.2270415
- Shenzhen MFD: https://doi.org/10.1177/0361198119843472
- Shenzhen multi-source fusion: https://doi.org/10.1049/iet-its.2018.5149
- CityFlow: https://doi.org/10.1145/3308558.3314139
- Flow: https://doi.org/10.1109/TRO.2021.3087314

---

## 2. 後発の改善候補

古い高引用論文は基準として有用だが、2026年時点の最終候補とは限らない。

| 文献 | 改善点 | 引用数（確認時） |
|---|---|---:|
| **STAEformer** (CIKM 2023) | adaptive spatio-temporal embedding + vanilla Transformer。5実データで当時SOTA。DCRNN/Graph WaveNetより先に試す価値が高い。 | **377** (ACM) |
| **PDFormer** (AAAI 2023) | traffic propagation delay と短距離/長距離依存を明示。伝播現象を扱う今回の問題と相性がよい。 | **262** (OpenAlex) |
| **INCREASE** (WWW 2023) | spatial proximityだけでなく functional similarity・transition probability を使うinductive kriging。IGNNKの直接的な改善候補。 | **15** (Semantic Scholar表示) |
| **AnchorGK** (KDD 2026) | incremental / stratified graph learning により、新規地点への inductive kriging を改善する非常に新しい手法。 | **0** (ResearchGate) |
| **DeKNN** (GIScience & Remote Sensing 2026) | latent-space krigingで「地点ごと完全欠損 (MCAL)」を効率よく補間。交通専用ではないがheld-out detector問題に近い。 | **0** (Crossref) |
| **PIDL + Fundamental Diagram Learner** (IEEE T-ITS 2022) | TSEだけでなく **FDの関数形そのものを同時学習**。FD recoveryには2021版より直接的。 | **71** (SciSpace) |
| **Physics-informed TSE + computational graph** (Information Fusion 2024) | sparse loop/probe dataからFDパラメータを推定しつつTSE。 | **121** (ScienceDirect “Cited by”) |
| **SPIDL: Stochastic Physics-Informed Deep Learning** (TR-C 2025) | 決定論的FDの弱点を、stochastic FDで改善。実データの散らばりを扱う。 | **15** (ScienceDirect) |
| **SWIFT: Online Adaptive Shockwave Detection and Inpainting** (TR-B 2025) | shockwaveの検出に加え、部分観測から欠けた波をinpainting。ShockwaveBenchへの対応がかなり直接的。 | **1前後**（出版直後・集計差あり） |
| **OpenCity** (ACM TIST 2025) | 米国・中国の複数都市を使うspatio-temporal foundation model。zero/few-shot cross-city transferが狙い。 | **64** (ResearchGate) |

### 一次資料・コード

- STAEformer: https://doi.org/10.1145/3583780.3615160  
  Code: https://github.com/XDZhelheim/STAEformer
- PDFormer: https://doi.org/10.1609/aaai.v37i4.25556  
  Code: https://github.com/BUAABIGSCity/PDFormer
- INCREASE: https://doi.org/10.1145/3543507.3583525  
  Code: https://github.com/zhengchuanpan/INCREASE
- AnchorGK: https://doi.org/10.1145/3770854.3780189
- DeKNN: https://doi.org/10.1080/15481603.2026.2705614
- PIDL + FD learner: https://arxiv.org/abs/2106.03142
- Information Fusion 2024: https://doi.org/10.1016/j.inffus.2023.101971
- SPIDL: https://doi.org/10.1016/j.trc.2025.105422
- SWIFT: https://doi.org/10.1016/j.trb.2025.103225
- OpenCity: https://doi.org/10.1145/3773912  
  Code: https://github.com/HKUDS/OpenCity

---

## 3. 個人で取り組む場合の優先順位

### 予測
1. **LargeST** — データ・評価の感覚を掴む
2. **Graph WaveNet** — 古典baseline
3. **STAEformer** — 現代的で比較的素直な強い候補
4. **PDFormer** — propagationを明示的に扱う場合

### 欠損センサー復元
1. **IGNNK**
2. **INCREASE**
3. **AnchorGK / DeKNN** — 2026の新しい改善案として確認

### Physics / FD
1. **PIDL + FDL (2022)**
2. **Information Fusion 2024**
3. **SPIDL (2025)**

### Shockwave
1. **Data-driven shockwave detection (2024)**
2. **SWIFT (2025)**

### OD
1. **Lu, Zhou & Zhang (2013)** を理論baselineにする
2. コンペ条件に合わせてsimulation / differentiable assignment / learning-based推定へ拡張する

---

## 4. 実装上の判断

- **DCRNN / Graph WaveNetは再現用baseline**として価値が高いが、2026年の最終モデル候補としては古い。
- forecastingなら、まず **STAEformer**、次に **PDFormer** を比較するのが合理的。
- detectorを丸ごと隠す評価なら、通常のtime-series imputationではなく **inductive spatio-temporal kriging** として考える。
- physics lossは自動的に改善を保証しない。FDの仮定、境界条件、loss weightがずれると悪化し得るため、**data-onlyとのablationを必ず残す**。
- Shenzhenが実際に追加される場合は、単一都市モデルだけでなく **cross-city transfer / foundation model** の価値が上がる。
- citation数が少ない2025–2026論文でも、今回のタスクに直接一致するもの（SWIFT、AnchorGK等）は優先して読む価値がある。

---

## 5. コンペ直結の実装資料

- **CBI+** — TrafficFlowBench/PeMS-LAを含むqueue・FD/QVDF等の解析実装  
  https://github.com/asu-trans-ai-lab/cbi_plus
- **DLSim4GMNS** — traffic assignment / network loading / simulation  
  https://github.com/asu-trans-ai-lab/DLSim4GMNS
- **IGNNK code**  
  https://github.com/Kaimaoge/IGNNK

---

## 引用数の注意

引用数は **2026-09-14確認時点**。Scopus、Semantic Scholar、ACM、OpenAlex、Crossref、ScienceDirect、SciSpace、ResearchGateでは収録範囲と版の統合方法が異なるため、**異なるデータベース間の数値を厳密なランキングとして比較しない**こと。

引用数は「定着度」の目安であり、「今回のコンペへの適合度」ではない。特に2025–2026年の論文は、引用が少なくても問題設定が直接一致する場合がある。
