#!/usr/bin/env python3
"""大井競馬 直近6ヶ月データ限定チューニング → 予測

Usage:
    python tune_oi_6mo.py --tune --trials 150
    python tune_oi_6mo.py --predict --date 2026/06/10
    python tune_oi_6mo.py --tune --trials 150 --predict --date 2026/06/10
"""
import argparse
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from models.features import build_features, FEATURE_COLS
from models.ensemble import train_ensemble, predict_ensemble
from models.backtest import run_backtest, _build_race_id, _build_train_features
from models.tuning import tune, suggest_params, params_to_weights, save_params, load_params

VENUE = "大井"
HIST_DIR = Path("data/historical_oi")
RACES_DIR = Path("data/races")
RACES_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DIR = Path("models/saved")
SAVED_DIR.mkdir(parents=True, exist_ok=True)

PARAMS_PATH = SAVED_DIR / "oi_6mo_best_params.json"
DB_PATH = SAVED_DIR / "oi_6mo_optuna.db"
STUDY_NAME = "oi_6mo_top5_coverage_v2"  # v2: avg_speed_idx/best_speed_idx/avg_corner_rate/avg_rel_last3f追加
DATA_RELIABILITY_K = 10
TRAIN_MONTHS = 6


def load_oi_6mo() -> pd.DataFrame:
    csvs = sorted(HIST_DIR.glob("oi_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"{HIST_DIR} にデータがありません")
    dfs = []
    for csv in csvs:
        try:
            dfs.append(pd.read_csv(csv, encoding="utf-8-sig"))
        except Exception as e:
            print(f"  警告: {csv} 読み込み失敗 ({e})")
    df = pd.concat(dfs, ignore_index=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[df["finish_position"].notna()].copy()
    df["finish_position"] = df["finish_position"].astype(int)

    # 直近6ヶ月のみ
    cutoff = df["race_date"].max() - pd.DateOffset(months=TRAIN_MONTHS)
    df = df[df["race_date"] >= cutoff].copy()
    print(f"大井直近{TRAIN_MONTHS}ヶ月: {len(df)}行 "
          f"({df['race_date'].min().date()} 〜 {df['race_date'].max().date()})")
    return df


def do_tune(df: pd.DataFrame, n_trials: int) -> dict:
    print(f"\n=== Optuna チューニング ({n_trials}試行) ===")
    print(f"スタディ: {STUDY_NAME}")

    def objective(trial):
        params = suggest_params(trial)
        weights = params_to_weights(params)
        result = run_backtest(df, params=params, weights=weights,
                              verbose=False, test_days=30)
        if "error" in result:
            return 0.0
        score = result["top5_coverage"]
        if trial.number % 10 == 0:
            print(f"  trial {trial.number}: score={score:.4f}", flush=True)
        return score

    study = tune(objective, STUDY_NAME, DB_PATH, n_trials=n_trials)
    best = study.best_params
    print(f"\n最良 top5_coverage: {study.best_value:.1%}")
    save_params(best, PARAMS_PATH)
    print(f"パラメータ保存: {PARAMS_PATH}")
    return best


def do_predict(df_train: pd.DataFrame, date_str: str, params: dict | None) -> None:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fname = f"oi_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    shutuba_path = RACES_DIR / fname
    if not shutuba_path.exists():
        print(f"出走表が見つかりません: {shutuba_path}")
        return

    df_shutuba = pd.read_csv(shutuba_path, encoding="utf-8-sig")
    df_shutuba["race_date"] = pd.to_datetime(df_shutuba["race_date"])
    print(f"\n出走表: {shutuba_path} ({len(df_shutuba)}頭)")

    pred_date = pd.Timestamp(dt)
    # 予測日より前のデータのみ使用
    df_t = df_train[df_train["race_date"] < pred_date].copy()
    if df_t.empty:
        print("警告: 訓練データなし。全データで学習します。")
        df_t = df_train.copy()

    print(f"訓練データ: {len(df_t)}行")
    print("特徴量構築・モデル学習中...")
    df_t["race_id"] = _build_race_id(df_t)
    X_tr, y_tr, rids_tr = _build_train_features(df_t, params)
    if X_tr.empty:
        print("訓練データが不足しています")
        return

    weights = params_to_weights(params) if params else None
    models = train_ensemble(X_tr, y_tr, rids_tr, params)

    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        if len(race_df) < 2:
            continue
        try:
            X_pred = build_features(race_df, df_t, params)
            scores = predict_ensemble(models, X_pred, weights)
        except Exception as e:
            print(f"  {race_no}R: エラー {e}")
            continue

        race_df = race_df.copy().reset_index(drop=True)
        race_df["score"] = scores
        race_df["n_hist"] = [
            len(df_t[df_t["horse_name"] == name])
            for name in race_df["horse_name"]
        ]
        race_df["data_reliability"] = race_df["n_hist"] / (race_df["n_hist"] + DATA_RELIABILITY_K)
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
    out_path = RACES_DIR / f"oi_{dt.strftime('%Y_%m%d')}_6mo_prediction.csv"
    df_pred.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"\n予測保存: {out_path}")

    print(f"\n{'='*70}")
    print(f"  大井競馬 {date_str} 予測（大井直近{TRAIN_MONTHS}ヶ月モデル）")
    print(f"{'='*70}")
    for race_no, grp in df_pred.groupby("race_no"):
        top = grp.sort_values("top3_prob", ascending=False).head(5)
        meta = top.iloc[0]
        print(f"\n【{race_no:2d}R】 {meta.get('race_name','')} {meta.get('distance','')}m")
        rows = []
        for rank, (_, h) in enumerate(top.iterrows(), 1):
            rel = h.get("data_reliability", 0)
            warn = " ⚠" if rel < 0.3 else ""
            rows.append([
                rank,
                int(h.get("horse_no", 0)),
                h.get("horse_name", ""),
                f"{h['top3_prob']:.1%}",
                f"{rel:.2f}",
                int(h.get("n_hist", 0)),
                h.get("jockey", ""),
                warn,
            ])
        print(tabulate(rows,
                       headers=["順","馬番","馬名","3着内確率","信頼度","参照数","騎手",""],
                       tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--date", default="2026/06/10")
    args = parser.parse_args()

    df = load_oi_6mo()
    params = load_params(PARAMS_PATH)
    if params:
        print(f"既存パラメータ読み込み: {PARAMS_PATH}")
    else:
        print("パラメータなし。デフォルトで実行します。")

    if args.tune:
        params = do_tune(df, args.trials)

    if args.predict:
        do_predict(df, args.date, params)

    if not any([args.tune, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
