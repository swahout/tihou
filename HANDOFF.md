# 引き継ぎ書 — 川崎予測モデル v6 / 精度探索の総括

最終更新: 2026-06-17（ローカルセッション → リモート継続用）

## 0. 結論サマリー（まずこれ）

- **本番モデルは v6**（LightGBM LambdaRank, 複数年CV）。`data/kawasaki_best_params.json` に最良パラメータ保存済み。
- **honest 精度**: top5_coverage **66.8%** / top3_hit **約49%** / top1（軸が3着内）**70〜80%**。
- **当面の目標だった「6/15・16の top3_hit 80%」は到達不能と判明**。v6(LambdaRank)・v7(top3直接最適化)・v8(分散予測+モンテカルロ)すべて大サンプルで **top3_hit ~47-50% 頭打ち**。
- これはモデルの欠陥ではなく、**現データ・特徴量での情報限界**。改善に残る唯一の実効レバーは **市場オッズ/人気**（後述）。
- **軸(top1)は70-80%と強い** → 実用上は「軸1頭 + 相手を手広く流す」運用が現実的。

## 1. このセッションでやったこと

### データ収集（完了）
- 船橋・浦和 2022-2026 を全収集（川崎・大井は既存）。全4会場 約16.6万行。
- 川崎は元々5/15までだったので **6/15・6/16 を追加収集**（評価用、各12R・結果完備）。

### 修正したバグ（重要・再発注意）
1. **utf-8-sig 追記による BOM 混入** — スクレイパーが `open(...,"a",encoding="utf-8-sig")` で追記していたため、resume のたびにファイル途中へ BOM(`﻿`) が入り、直後の行の `race_date` を壊して `to_datetime` がクラッシュ。
   - 対処: 追記を `utf-8` に変更（`collect_historical_nankan.py` / `collect_historical_kawasaki.py`）、`load_history` で race_date の BOM 除去、既存CSV全BOM除去で修復。**utf-8-sig は追記に使わない。**
2. **単年CVによる過学習** — v5まで Optuna 評価が「2026単年・8R以降のみ(~40R)」で、報告 73% は過学習による水増しだった。→ **複数年CV (2024+2025+2026, 計~1,800R) に修正 = v6**。honest 値は top5 66.8%。
3. **run_collection.sh の無限再チューニング** — 完了判定が出力されないログ文字列を見ていた。→ sentinel `/tmp/kawasaki_tune_v6.done` 方式に修正。
4. パス修正（`TIHOU=/home/user/tihou` → 環境に合わせる。リモートでは要再設定）。

### 試したモデル（すべて top3_hit ~50% で頭打ち）
| 版 | 内容 | 結果 |
|----|------|------|
| v6 | 複数年CV LambdaRank（本番） | top5 66.8% / top3 ~49% / top1 70-80% |
| v7 | top3_hit 直接最適化 (`--metric top3`) | 0.471 = v6同等、182/200で中断 |
| v8 | 分散予測+モンテカルロ (`predict_dist.py`) | 大サンプル top3 ~47-49% = v6同等 |

- **v8 の学び**: 各馬の speed_idx を分位点回帰で分布予測し1000回シミュ→P(top3) で選択。理論的には「ムラのある馬/堅実な馬」を区別できるはずで、小サンプル(6/15-16)では 51% に見えたが、**大サンプル(~1800R)では v6 と同等 = 小サンプルはノイズだった**。MC > 中央値点(+0.5-1.3pp) で分散情報は僅かに効くが LambdaRank を超えない。
- **モンテカルロの限界**: 単一モデルの点スコアからの再ランキングは P(top3) がスコアに単調なので top-k 選択を変えない。分布予測版でやっと「変えられる」が、改善幅は誤差レベルだった。

## 2. 残された唯一の有望な方向：市場オッズ/人気

- 競馬予測で最も強い予測因子は**市場（単勝オッズ・人気順）**。
- 歴史データの単勝オッズは **99.7% が NaN**（keiba.go.jp の `RaceMarkTable` は過去の確定オッズを返さない）→ **訓練特徴量には使えない**。
- ただし**当日・直近レースは `OddsTanFuku` エンドポイントで取得可能**。
- 現実的な実装案:
  1. 直近・未来レースの予測時に `OddsTanFuku` で人気順/オッズを取得。
  2. v6 のモデル順位と市場人気順を**ブレンド**（加重平均 or ランク融合）して最終順位を出す。
  3. ブレンド比率は直近の held-out で較正。
- 期待効果は「数pp改善」レベル（劇的ではない）。実装は新規。**リモートでの次タスク候補No.1**。

## 3. リモートで作業を始める人へ（セットアップ）

1. **パス**: `run_collection.sh` の `TIHOU=` を実行環境に合わせる（現在ローカル用の値）。
2. **依存**: `requirements.txt`（LightGBM 4.6, optuna, pandas 等）。
3. **データは追跡済み**（`data/historical_*/`, `data/kawasaki_optuna.db` はコミット内）。keiba.go.jp 保持期間は約3-4年=2022以降のみ収集可。
4. **直近データ更新**: `python collect_historical_kawasaki.py --years 2026`（resume対応、未収集日のみ取得。今日<の日付のみ）。

## 4. 主要コマンド

```bash
# 予測（翌日 or 指定日）
python kawasaki_predict.py --date 2026/06/18
# バックテスト（年次LeaveOneYearOut、top5/top3/top1）
python kawasaki_predict.py --backtest
# チューニング（複数年CV。--metric top3 で top3_hit 最適化=v7系, 保存先別ファイル）
python kawasaki_predict.py --tune --trials 200 [--metric top3]
# 特定日の held-out 評価（top3_hit/top5/top1）
python eval_recent.py --dates 2026/06/15 2026/06/16 [--params <json>] [--no-nankan]
# 分散予測+モンテカルロ評価（v8）
python predict_dist.py --dates 2026/06/15 2026/06/16        # 特定日
python predict_dist.py --backtest-years 2024 2025 2026      # 大サンプル
# アブレーション（南関データ寄与の確認 → ほぼ無寄与と判明）
python kawasaki_predict.py --backtest --no-nankan
```

## 5. 検証で確定した事実（再実験しないため）

- **船橋・浦和データは top3_hit にほぼ無寄与**（大サンプルで 47.6↔46.0 等、誤差範囲。top5 のみ +約1pp）。収集の労力は目標に対しては限定的だった。
- **数日規模の評価はノイズが大きい**（10-24R）。モデル比較は必ず大サンプル(年次backtest, ~数百-1800R)で。
- **CLAUDE.md は実モデルに更新済み**: 川崎=単一LightGBM LambdaRank（4モデルアンサンブルは大井(oi)の方）。

## 6. ファイル早見

| ファイル | 役割 |
|---------|------|
| `kawasaki_predict.py` | 川崎 予測/バックテスト/チューニング（本体）|
| `eval_recent.py` | 特定日の held-out 精度評価（top3_hit等）|
| `predict_dist.py` | v8 分散予測+モンテカルロ評価 |
| `run_collection.sh` | 収集→チューニングの自動ラッパー |
| `collect_historical_nankan.py` / `_kawasaki.py` | データ収集（BOM修正済み）|
| `data/kawasaki_best_params.json` | v6 本番パラメータ |
| `data/kawasaki_best_params_top3.json` | v7 用（中断のため未生成の可能性）|
| `data/kawasaki_optuna.db` | Optuna スタディ（v6, v7部分試行）|

## 7. 推奨ネクストステップ（優先順）

1. **市場オッズ/人気のブレンド**（直近・未来レース向け。唯一の有望レバー。§2）。
2. v6 を運用確定し、「軸1頭+手広く流す」馬券で実運用テスト（top1 が 70-80% と強い）。
3. （任意）分散予測×LambdaRank のハイブリッド（軸=v6, 3着内セット=v8）— ただし v8 単体は改善薄。
