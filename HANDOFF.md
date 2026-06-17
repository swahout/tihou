# 引き継ぎ書 — 川崎予測モデル v9（市場人気ブレンド）/ 精度探索の総括

最終更新: 2026-06-17（v9 市場ブレンド実装を追記）

## 0. 結論サマリー（まずこれ）

- **本番運用形は v9 = v6モデル + 市場人気ブレンド**（rank fusion, `w_model=0.2`）。モデルパラメータ自体は v6（`data/kawasaki_best_params.json`）のまま。
- **★最大の発見: 市場人気(popularity)単独がモデルを全指標で上回る**（LeaveOneYearOut 1818R）:
  | | top3_hit | top5_cov | top1_acc |
  |---|---|---|---|
  | 市場のみ | 58.3% | 78.4% | 74.9% |
  | **ブレンド(w_model=0.2)** | **58.4%** | **78.8%** | 74.2% |
  | モデルのみ v6 | 48.6% | 69.2% | 57.1% |
- 市場を入れるだけで **top3 +10pp / top5 +10pp / top1 +18pp**。モデルは薄い補正役。
- **前セッションの誤認を訂正**: 「市場は歴史欠損(win_odds 99.7%NaN)で訓練/検証不可、直近のみ」は誤り。**`popularity`(人気順)列は履歴99.0%充足** → 大サンプルでブレンドをバックテスト可能（`--backtest --blend`）。
- **当面の目標「top3_hit 80%」は市場込みでも未達**だが、ブレンドで実用域(top3 58%, top1 75%)へ。モデル単独の天井~50%は変わらず。
- **軸(top1)が75%と強い** → 「軸1頭 + 相手を手広く流す」運用が現実的。

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
| **v9** | **市場人気ブレンド (`--blend`, w_model=0.2)** | **top3 58.4% / top5 78.8% / top1 74.2%（大幅改善・本番運用形）** |

- **v8 の学び**: 各馬の speed_idx を分位点回帰で分布予測し1000回シミュ→P(top3) で選択。理論的には「ムラのある馬/堅実な馬」を区別できるはずで、小サンプル(6/15-16)では 51% に見えたが、**大サンプル(~1800R)では v6 と同等 = 小サンプルはノイズだった**。MC > 中央値点(+0.5-1.3pp) で分散情報は僅かに効くが LambdaRank を超えない。
- **モンテカルロの限界**: 単一モデルの点スコアからの再ランキングは P(top3) がスコアに単調なので top-k 選択を変えない。分布予測版でやっと「変えられる」が、改善幅は誤差レベルだった。

## 2. 市場オッズ/人気ブレンド（v9・実装済み）

**前セッションで「次タスク候補No.1」としていた市場ブレンドを実装し、効果を実証した。期待は「数pp」だったが実際は二桁pp改善。**

### 実装
- `scrapers/keibago.py::get_tanfuku_odds()` — `OddsTanFuku` から当日〜直近の確定単勝/複勝オッズを取得。
- `collect_kawasaki_shutuba.py` — 出走表収集時に `OddsTanFuku` で `win_odds` を補完（DebaTableのオッズはレース前空が多いため）。
- `kawasaki_predict.py::blend_order_score()` / `market_rank_from_odds()` — モデル順位と市場人気順を rank fusion: `combined = w_model*model_rank + (1-w_model)*market_rank`、`w_model=BLEND_W_MODEL=0.2`。
- `do_predict` — `win_odds` があれば自動ブレンド・人気列表示。**無ければモデル単独にフォールバック**（前日予測でオッズ未確定の場合）。
- 検証: `kawasaki_predict.py --backtest --blend`（歴史は `popularity` を市場順位に使用）/ `eval_recent.py --blend`。

### キモ（再実験しないため）
- **市場順位の出どころが2つある**: ①予測時=`OddsTanFuku`の確定オッズ→人気順、②バックテスト=履歴の`popularity`列（99%充足）。両者は同じ市場コンセンサス。
- **w_model スイープ結果**（LOYO 1818R）: w=0で市場のみ、w=0.2〜0.25でtop5最良(78.8-79.1%)、w=1.0でモデルのみ(69%)。**0.2が無難**。
- **運用上の注意**: ブレンド効果は「発走直前の確定に近いオッズ」が前提。前日収集だとオッズ未確定でフォールバック＝v6相当。**発走直前に `collect_kawasaki_shutuba.py` を再実行**してオッズを取り込むこと。

### 次に伸ばすなら
- win_odds から**連続的な市場勝率/複勝率**を作り、rank融合でなく確率ブレンドにする（現状は順位融合）。ただし順位融合で既に市場≒上限近くまで取れている可能性が高い。
- モデルの存在価値は薄い（市場をほぼ超えない）。モデル側で伸ばすより**市場と乖離した妙味馬の検出（期待値ベース馬券）**に方向転換する方が実益が大きいかもしれない。

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
| `kawasaki_predict.py` | 川崎 予測/バックテスト/チューニング（本体）。`--backtest --blend` で市場比較、`blend_order_score()`/`market_rank_from_odds()` |
| `scrapers/keibago.py` | `get_tanfuku_odds()` 追加（OddsTanFukuから当日オッズ取得）|
| `eval_recent.py` | 特定日の held-out 精度評価（`--blend` で市場ブレンド）|
| `predict_dist.py` | v8 分散予測+モンテカルロ評価 |
| `run_collection.sh` | 収集→チューニングの自動ラッパー |
| `collect_historical_nankan.py` / `_kawasaki.py` | データ収集（BOM修正済み）|
| `data/kawasaki_best_params.json` | v6 本番パラメータ |
| `data/kawasaki_best_params_top3.json` | v7 用（中断のため未生成の可能性）|
| `data/kawasaki_optuna.db` | Optuna スタディ（v6, v7部分試行）|

## 7. 推奨ネクストステップ（優先順）

1. ~~市場オッズ/人気のブレンド~~ → **完了(v9)。期待を超える改善（§2）。**
2. **v9 を運用確定**し、「軸1頭+手広く流す」馬券で実運用テスト（top1 75%）。**発走直前にオッズ再収集を忘れない**（前日だとフォールバックでv6相当）。
3. **期待値ベース馬券へ方向転換の検討**: モデルは市場をほぼ超えないため、的中率を上げるより「市場と乖離した妙味馬（モデルが市場より高評価）」を拾う回収率設計の方が実益が大きい可能性。`OddsTanFuku`の複勝オッズ(取得済み)と組み合わせ可能。
4. （任意）確率ブレンド版（順位融合→市場勝率/複勝率の連続ブレンド）。ただし順位融合で既に市場上限近く。
