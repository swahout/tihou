#!/usr/bin/env python3
"""大井競馬 チューニング・バックテスト・予測 メインスクリプト

Usage:
    python oi_tune.py --backtest                         # バックテスト（保存済みパラメータ）
    python oi_tune.py --tune --trials 200                # Optunaチューニング
    python oi_tune.py --tune --trials 100 --backtest     # チューニング → バックテスト
    python oi_tune.py --predict --date 2026/06/10        # 予測
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from models.features import build_features, FEATURE_COLS
from models.ensemble import train_ensemble, predict_ensemble
from models.backtest import run_backtest, _build_race_id
from models.tuning import tune, suggest_params, params_to_weights, save_params, load_params

VENUE = "大井"
HIST_DIR = Path("data/historical_oi")
RACES_DIR = Path("data/races")
SAVED_DIR = Path("models/saved")
SAVED_DIR.mkdir(parents=True, exist_ok=True)

PARAMS_PATH = SAVED_DIR / "oi_best_params.json"
DB_PATH = SAVED_DIR / "oi_optuna.db"
STUDY_NAME = "oi_top3_top5_coverage_v4"  # v4: 月次walk-forward CV + 船橋データ追加
METRIC = "top5_coverage"
DATA_RELIABILITY_K = 10  # umaの知見: K=5より保守的なK=10が過小評価を防ぐ


OTHERS_DIR = Path("data/historical_others")


def load_history() -> pd.DataFrame:
    csvs = sorted(HIST_DIR.glob("oi_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{HIST_DIR} にデータがありません。"
            " まず: python collect_historical_oi.py --years 2023 2024 2025"
        )
    dfs = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"  警告: {csv} 読み込み失敗 ({e})")

    # 他馬場データ（存在すれば追加）
    other_csvs = sorted(OTHERS_DIR.glob("*.csv")) if OTHERS_DIR.exists() else []
    for csv in other_csvs:
        try:
            df = pd.read_csv(csv, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"  警告: {csv} 読み込み失敗 ({e})")
    if other_csvs:
        print(f"  他馬場データ: {len(other_csvs)}ファイル読み込み")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["race_date"] = pd.to_datetime(df_all["race_date"])
    df_all["finish_position"] = pd.to_numeric(df_all["finish_position"], errors="coerce")
    df_all = df_all[df_all["finish_position"].notna()].copy()
    df_all["finish_position"] = df_all["finish_position"].astype(int)
    venues = df_all["venue"].unique().tolist()
    print(f"履歴データ: {len(df_all)}行 ({df_all['race_date'].dt.year.min()}〜{df_all['race_date'].dt.year.max()}) 馬場: {venues}")
    return df_all


def do_backtest(df_all: pd.DataFrame, params: dict | None) -> dict:
    print("\n=== LeaveOneYearOut バックテスト ===")
    weights = params_to_weights(params) if params else None
    result = run_backtest(df_all, params=params, weights=weights)
    if "error" in result:
        print(f"エラー: {result['error']}")
        return result
    print(f"\n平均 top5_coverage : {result['top5_coverage']:.1%}")
    print(f"平均 top3_hit      : {result['top3_hit']:.1%}")
    print(f"平均 top1_acc      : {result['top1_acc']:.1%}")
    return result


def do_tune(df_all: pd.DataFrame, n_trials: int) -> dict:
    print(f"\n=== Optuna チューニング ({n_trials}試行) ===")
    print(f"スタディ: {STUDY_NAME}")

    def objective(trial):
        params = suggest_params(trial)
        weights = params_to_weights(params)
        # 直近30日をテストセットとして高速評価
        result = run_backtest(df_all, params=params, weights=weights,
                              verbose=False, test_days=30)
        if "error" in result:
            return 0.0
        score = result["top5_coverage"]
        if trial.number % 10 == 0:
            print(f"  trial {trial.number}: score={score:.4f}", flush=True)
        return score

    study = tune(objective, STUDY_NAME, DB_PATH, n_trials=n_trials)
    best = study.best_params
    print(f"\n最良 {METRIC}: {study.best_value:.1%}")
    save_params(best, PARAMS_PATH)
    print(f"パラメータ保存: {PARAMS_PATH}")
    return best


def do_predict(df_all: pd.DataFrame, date_str: str, params: dict | None) -> None:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fname = f"oi_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    shutuba_path = RACES_DIR / fname
    if not shutuba_path.exists():
        print(f"出走表が見つかりません: {shutuba_path}")
        print(f"まず: python collect_oi_shutuba.py --date {date_str}")
        return

    df_shutuba = pd.read_csv(shutuba_path, encoding="utf-8-sig")
    df_shutuba["race_date"] = pd.to_datetime(df_shutuba["race_date"])
    print(f"\n出走表: {shutuba_path} ({len(df_shutuba)}頭)")

    # 訓練データ（予測対象日より前の全データ）
    # バックテストと違い実運用では予測日前日までの全データを使用可能
    pred_date = pd.Timestamp(dt)
    df_train = df_all[df_all["race_date"] < pred_date].copy()
    if df_train.empty:
        print(f"警告: {date_str}より前の訓練データがありません。全データで学習します。")
        df_train = df_all.copy()

    # 特徴量構築・モデル学習
    print("特徴量構築・モデル学習中...")
    df_train["race_id"] = _build_race_id(df_train)
    from models.backtest import _build_train_features
    X_tr, y_tr, rids_tr = _build_train_features(df_train, params)
    if X_tr.empty:
        print("訓練データが不足しています")
        return

    weights = params_to_weights(params) if params else None
    models = train_ensemble(X_tr, y_tr, rids_tr, params)

    # レース別予測
    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        if len(race_df) < 2:
            continue
        try:
            X_pred = build_features(race_df, df_train, params)
            scores = predict_ensemble(models, X_pred, weights)
        except Exception as e:
            print(f"  {race_no}R: エラー {e}")
            continue

        race_df = race_df.copy().reset_index(drop=True)
        race_df["score"] = scores
        race_df["n_hist"] = [
            len(df_train[df_train["horse_name"] == name])
            for name in race_df["horse_name"]
        ]
        race_df["data_reliability"] = race_df["n_hist"] / (race_df["n_hist"] + DATA_RELIABILITY_K)
        # top3_prob: スコア正規化 → 信頼度補正（データ少ない馬はベースレートに引き戻す）
        total = scores.sum()
        base_rate = 3.0 / max(len(race_df), 1)
        raw_prob = scores / total * 3.0 if total > 0 else np.ones(len(scores)) * base_rate
        rel = race_df["data_reliability"].values
        race_df["top3_prob_raw"] = raw_prob
        race_df["top3_prob"] = raw_prob * rel + base_rate * (1 - rel)
        out_rows.append(race_df)

    if not out_rows:
        print("予測失敗")
        return

    df_pred = pd.concat(out_rows, ignore_index=True)
    out_path = RACES_DIR / f"oi_{dt.strftime('%Y_%m%d')}_prediction.csv"
    df_pred.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"\n予測保存: {out_path}")

    # 表示
    print(f"\n{'='*70}")
    print(f"  大井競馬 {date_str} 予測")
    print(f"{'='*70}")
    for race_no, grp in df_pred.groupby("race_no"):
        top = grp.sort_values("top3_prob", ascending=False).head(7)
        meta = top.iloc[0]
        print(f"\n【{race_no:2d}R】 {meta.get('race_name','')} {meta.get('distance','')}m {meta.get('track_cond','')}")
        rows = []
        for rank, (_, h) in enumerate(top.iterrows(), 1):
            rel = h.get("data_reliability", 0)
            warn = " ⚠️" if rel < 0.3 else ""
            rows.append([
                rank,
                int(h.get("horse_no", 0)),
                h.get("horse_name", ""),
                f"{h['top3_prob']:.1%}",
                f"{rel:.2f}",
                int(h.get("n_hist", 0)),
                h.get("jockey", ""),
                h.get("win_odds", ""),
                warn,
            ])
        print(tabulate(rows,
                       headers=["順","馬番","馬名","3着内確率","信頼度","参照数","騎手","単勝",""],
                       tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description="大井競馬 MLチューニング・予測")
    parser.add_argument("--backtest", action="store_true", help="LeaveOneYearOutバックテスト")
    parser.add_argument("--tune", action="store_true", help="Optunaチューニング")
    parser.add_argument("--trials", type=int, default=100, help="チューニング試行数")
    parser.add_argument("--predict", action="store_true", help="予測実行")
    parser.add_argument("--date", help="予測対象日 YYYY/MM/DD（省略時=当日）")
    args = parser.parse_args()

    df_all = load_history()
    params = load_params(PARAMS_PATH)
    if params:
        print(f"チューニング済みパラメータ読み込み: {PARAMS_PATH}")
    else:
        print("チューニング済みパラメータなし。デフォルトで実行します。")

    if args.tune:
        params = do_tune(df_all, args.trials)

    if args.backtest:
        do_backtest(df_all, params)

    if args.predict:
        date_str = args.date or datetime.now().strftime("%Y/%m/%d")
        do_predict(df_all, date_str, params)

    if not any([args.tune, args.backtest, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
