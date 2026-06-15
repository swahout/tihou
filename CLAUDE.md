# 地方競馬予想AIプロジェクト — CLAUDE.md

## ⚠️ 絶対的な制約（必ず守ること）

**予測対象年のレース結果を学習データに含めてはいけない。**

例: 2026年を予測する場合 → 2026年の全レース結果は訓練データから除外する。
`LeaveOneYearOut` のロジックがこれを保証しているが、コードを変更する際も意識すること。

---

## プロジェクト概要

NAR（地方競馬全国協会）主催の地方競馬を対象に、アンサンブルMLで着順を予測するシステム。
中央競馬（uma リポジトリ）と同じ4モデルアンサンブル基盤を採用しつつ、地方競馬の特性に合わせた設計。
初期ターゲット: **大井競馬**（最も権威ある南関東地方競馬）

---

## 中央競馬（uma）との根本的な違い

| 項目 | 地方（tihou） | 中央（uma） |
|------|-------------|------------|
| レース頻度 | 毎日複数場・10R前後 | 週末中心 |
| 対象 | 特定場の全クラス | G1レース個別対応 |
| 路面 | **ダート主体** | 芝・ダート混在 |
| 同一馬の出走 | 同じ場に繰り返し出走 → 当場実績豊富 | 出走間隔長め |
| 移籍馬 | 場間移籍直後は当場実績ゼロ | 場による差異小 |
| データソース | keiba.go.jp（NAR公式） | netkeiba.com |
| 騎手差 | **地方はリーディングと下位の差が中央より大きい** | 相対的に均一 |
| 速度指数 | タイムが取得困難 → 当面は成績率ベース | 主要特徴量 |

---

## アーキテクチャ

### モデル（4モデルアンサンブル）

| モデル | 役割 | デフォルト重み |
|--------|------|---------------|
| LightGBM | 勝率二値分類 | 40% |
| XGBoost (rank:pairwise) | 着順ランキング最適化 | 25% |
| RandomForest | 汎化・過学習抑制 | 20% |
| LogisticRegression (CalibratedCV) | 確率キャリブレーション | 15% |

重みはOptunaで自動最適化される。

### 評価指標

- `top5_coverage` — 実際の3着内馬が予測top5に何頭入るか（**メイン指標**）
- `top3_hit` — 実際の3着内馬が予測top3に何頭入るか
- `top1_acc` — 1着的中率
- `spearman` — 着順相関

バックテスト: **LeaveOneYearOut**（各年を1つずつテストセット、残り年で学習）

---

## データソース — keiba.go.jp（NAR公式）

### 主要エンドポイント

```
# レース一覧（日別・場別）
GET /KeibaWeb/TodayRaceInfo/RaceList
    ?k_raceDate=YYYY%2FMM%2FDD&k_babaCode={CODE}

# 単勝・複勝オッズ + 馬・騎手・調教師情報
GET /KeibaWeb/TodayRaceInfo/OddsTanFuku
    ?k_raceDate=...&k_babaCode={CODE}&k_raceNo={N}

# 出馬表（着別成績・過去走付き）
GET /KeibaWeb/TodayRaceInfo/DebaTable
    ?k_raceDate=...&k_babaCode={CODE}&k_raceNo={N}

# レース結果（着順・馬番・タイム）
GET /KeibaWeb/TodayRaceInfo/RaceMarkTable
    ?k_raceDate=...&k_babaCode={CODE}&k_raceNo={N}
```

### 場コード

```python
VENUE_MAP = {
    "大井": 20, "船橋": 19, "川崎": 21, "浦和": 18,
    "門別": 36, "園田": 27, "姫路": 28, "名古屋": 24,
    "金沢": 22, "笠松": 25, "高知": 42, "佐賀": 44,
}
```

### スクレイピング上の注意

- **REQUEST_DELAY は最低 1.2 秒を守ること（礼儀として）**
- `TodayRaceInfo` のパスは過去日付でも動作する（データ保持期間: 概ね3〜5年）
- **`OddsTanFuku` は3年以上前のデータでエラーページを返す** → 過去データ収集には使用不可
- **`RaceMarkTable` が唯一の安定エンドポイント** → 出走馬情報＋結果を1リクエストで取得
- `RaceMarkTable` の結果は未確定時（レース前）は空になる → 当日予測には注意
- 単勝オッズ（列[15]）は歴史データでは空になることが多い（当日のみ有効）
- PC版のみ利用可（SPサイトは別URL、keiba.go.jpはJSなしで取得可能）

### RaceMarkTable の列順

```
[0]=着順, [1]=枠, [2]=馬番, [3]=馬名, [4]=所属,
[5]=性齢, [6]=負担重量, [7]=騎手（所属）, [8]=調教師,
[9]=馬体重(増減), [10]=タイム, [11]=着差, [12]=上がり3F,
[13]=コーナー通過順, [14]=人気, [15]=単勝オッズ
```

---

## ファイル構造

```
tihou/
├── CLAUDE.md
├── scrapers/
│   ├── __init__.py
│   └── keibago.py              # NAR公式サイト スクレイパー
├── models/
│   ├── __init__.py
│   ├── ensemble.py             # 4モデルアンサンブル
│   ├── features.py             # 特徴量エンジニアリング
│   ├── backtest.py             # LeaveOneYearOut CV
│   ├── tuning.py               # Optuna共通ロジック
│   └── saved/
│       ├── oi_best_params.json
│       └── oi_optuna.db
├── data/
│   ├── historical_oi/          # 大井 過去レースデータ（年別CSV）
│   │   └── oi_YYYY.csv
│   └── races/
│       ├── oi_2026_shutuba.csv
│       └── oi_2026_prediction.csv
├── collect_historical_oi.py    # 大井 過去データ収集
├── collect_oi_shutuba.py       # 大井 出走表収集
├── oi_tune.py                  # チューニング・予測メインスクリプト
└── requirements.txt
```

---

## 特徴量一覧（kawasaki v3: 33特徴量）

```
# 川崎専用
top3_rate_venue        当場3着内率（Bayesian K_HORSE=Optuna最適化）
win_rate_venue         当場勝率
n_venue                当場出走数
top3_rate_venue_dist   当場×当距離3着内率 ← 地方で最重要 [v2追加]
n_venue_dist           当場×当距離出走数 [v2追加]

# 通算
top3_rate_total        通算3着内率
win_rate_total         通算勝率
n_total                通算出走数

# 馬場状態別 [v2追加]
top3_rate_cond         当馬場状態3着内率（重・良・稍重・不良別）
n_cond                 同馬場出走数

# 直近フォーム
recent_avg_pos         直近5走平均着順
recent_top3            直近5走3着内率
recent_venue_avg_pos   当場直近5走平均着順
form_trend             直近3走改善傾向（正=改善）[v2追加]

# 速度指数
avg_speed_idx          平均相対速度指数（>1.0=速い）
best_speed_idx         上位3走平均速度（ceiling性能）
avg_last3f_idx         上がり3F相対指数
avg_corner_ratio       最終コーナー通過順÷頭数（0=逃げ,1=追込）

# 騎手・調教師
jockey_top3_rate       騎手3着内率（当場、K_JOCKEY=Optuna最適化）
jockey_win_rate        騎手勝率（当場）
jockey_top3_rate_dist  騎手×距離帯3着内率（sprint/mile/long）[v3追加]
trainer_top3_rate      調教師3着内率（当場）[v2追加]

# H2H（直接対決）[v3追加]
h2h_score              川崎同一フィールドでの過去対戦勝率（2戦未満=0.5）

# フラグ
is_ten_nori            テン乗りフラグ
class_change           昇降級（正=昇級,負=降級,0=同クラス）[v2追加]

# レース・馬属性
race_class_enc         クラス（A1=7〜未格付=0）
distance / field_size / age / weight_carried / sex_enc / umaban
days_since_last        休養日数
data_reliability       n/(n+8)
```

### K値最適化について（v3）

v2まで固定値 `K_HORSE=8, K_JOCKEY=30` を使用していたが、feature importance分析で
「騎手率34.6% vs 当場成績率1.0%」という偏りが判明。Kが小さすぎると当場実績がほぼ prior になる。

v3からは K_HORSE (2-25) と K_JOCKEY (5-80) をOptunaで最適化。
初期試行では K_HORSE=22, K_JOCKEY=50 が高スコアを示した（より保守的な平滑化が有効）。

## 特徴量一覧（oi v4: 33特徴量）

```
# 成績系（ベイズ平滑化 k=8）
top3_rate_total        通算3着内率
top3_rate_venue        当場3着内率          ← 地方で最重要
top3_rate_dist         当距離3着内率
top3_rate_venue_dist   当場当距離3着内率
top3_rate_cond         当馬場状態3着内率
win_rate_total         通算勝率
win_rate_venue         当場勝率

# データ量
n_total / n_venue / n_dist / n_vd

# 近走系（直近5走）
recent_avg_pos         直近平均着順
recent_top3            直近3着内率
recent_win             直近勝率
recent_avg_pos_venue   当場直近平均着順

# 騎手・調教師（当場集計、k=30）
jockey_top3_rate / jockey_win_rate
trainer_top3_rate

# 物理系
sex_enc / age / weight_carried / weight_change / horse_weight
umaban / waku / field_size / distance

# 休養
days_since_last_race

# メタ
data_reliability       n / (n + 5)  0=初出走, 0.5=5戦, 0.8=20戦

# v4追加: 速度指数・脚質・上がり（98.9%カバレッジ）
avg_speed_idx          馬の平均相対速度 (1.0=レース平均, >1=速い)
best_speed_idx         馬の上位3走平均速度 (ceiling性能)
avg_corner_rate        最終コーナー通過順÷頭数 (0=逃げ, 1=追い込み)
avg_rel_last3f         上がり3F÷レース平均 (1.0未満=キックが強い)

# 廃止
# win_odds             99.7%がNaN → v3で除外
```

---

## Optunaスタディのバージョン管理

特徴量を追加・変更した場合は必ずスタディ名をインクリメントすること。

```python
# kawasaki
study_name = "kawasaki_2026_8R_top5_v4"  # v1, v2, v3(破損削除), v4...

# oi
study_name = "oi_top3_top5_coverage_v4"
```

**理由**: 特徴量構成が変わると過去の試行（異なる特徴空間）とTPEサンプラーが混在し最適化が混乱する。

スタディのDBは `data/kawasaki_optuna.db` に保存（oi: `models/saved/oi_optuna.db`）。

### kawasaki スタディ履歴

| バージョン | 変更内容 | 特徴量数 | 最高 top5_coverage（8R以降） |
|-----------|---------|---------|---------------------------|
| v1 | 初期構築（25特徴量）速度指数・脚質含む | 25 | 73.6% (233試行) |
| v2 | 当場当距離・馬場状態別・調教師実績・フォームトレンド・昇降級追加 | 31 | 73.3% (300試行) |
| v3 | actual_top3インデックスバグで全試行0.0 → 削除 | - | - |
| v4 | K値Optuna最適化・騎手×距離帯・H2H直接対決スコア追加 | 33 | チューニング中 |

### oi スタディ履歴

| バージョン | 変更内容 | 特徴量数 | 最高 top5_coverage |
|-----------|---------|---------|-------------------|
| v1 | 初期構築 | 30 | ― |
| v2 | 高速化（precompute_stats）、win_odds保持 | 30 | ― |
| v3 | win_odds除外(99.7%NaN)、2026データ追加(71テストレース) | 29 | 70.4% (214試行) |
| v4 | 速度指数(avg_speed_idx, best_speed_idx)・脚質(avg_corner_rate)・上がり3F(avg_rel_last3f)追加 | 33 | チューニング中 |

---

## データ収集 → モデル構築の手順（初回）

### Step 1: 過去データ収集（最低3年分推奨）

```bash
python collect_historical_oi.py --years 2023 2024 2025
```

1日あたり10〜12レース × 開催日数 × 年数分のリクエストが発生する。
開催日を自動判定するため全日付を試し、開催なしの日はスキップする。
リクエスト間隔 1.2秒のため、1年分 ≈ 30〜60分。

### Step 2: ベースライン確認

```bash
python oi_tune.py --backtest
```

### Step 3: Optunaチューニング

```bash
python oi_tune.py --tune --trials 200
```

### Step 4: 出走表収集 → 予測

```bash
python collect_oi_shutuba.py --date 2026/06/10
python oi_tune.py --predict --date 2026/06/10
```

---

## 新規競馬場対応の手順

1. `collect_historical_oi.py` をコピーして `VENUE` と `BABA_CODE` を変更
2. `oi_tune.py` をコピーして定数を変更
3. `models/saved/` に新しい `{venue}_best_params.json` が自動生成される
4. `study_name` を `{venue}_top3_top5_coverage_v1` に設定

---

## よく使うコマンド

```bash
# 過去データ収集
python collect_historical_oi.py --years 2023 2024 2025

# 特定日のみ
python collect_historical_oi.py --date 2025/12/31

# バックテスト（デフォルトパラメータ）
python oi_tune.py --backtest

# Optunaチューニング
python oi_tune.py --tune --trials 200

# チューニング継続（同じstudyに追加）
python oi_tune.py --tune --trials 100

# 予測
python oi_tune.py --predict --date 2026/06/10
```

---

## 過去の試行と教訓

| 日付 | 内容 | 結果 |
|------|------|------|
| 2026-06-08 | 初期構築開始（keiba.go.jpスクレイピング設計） | ― |
| 2026-06-08 | OddsTanFuku を過去データ収集に使用 → 3年以上前でエラーページ返却 | RaceMarkTable一本に変更 |
| 2026-06-08 | 2019〜2021年のデータ収集を試みた → 全件取得失敗 | keiba.go.jpの保持期間は約3〜4年。2022年以降のみ収集可能 |

---

## CLAUDE.md の更新ルール

| タイミング | 更新内容 |
|-----------|----------|
| 特徴量を変更したとき | 特徴量一覧・スタディバージョンを更新 |
| 新規競馬場対応時 | 場コード・スコアを追記 |
| スクレイピングの罠を踏んだとき | データソースセクションに追記 |
| レース結果が出たとき | 振り返りセクションを追加 |
| コードの罠を踏んだとき | 「過去の試行と教訓」に追記 |

更新判断基準: 「次のセッションの Claude がこれを読んで同じ分析を再現できるか？」
