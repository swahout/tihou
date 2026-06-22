#!/usr/bin/env python3
"""任意の地方競馬場のモデルを 構築・評価・チューニング する汎用オーケストレータ。

kawasaki_predict.py の実証済みロジック(v10特徴量 + LambdaRank + 市場ブレンド)を
VENUE 差し替えで各場に適用し、場ごとに:
  --backtest : LeaveOneYearOut で model/market/blend を比較し、最適な w_model を探索
  --tune     : Optuna で LightGBM ハイパラ + K値を場別に最適化
を行う。パラメータ/スタディは場別ファイルに保存。

Usage:
  python venue_model.py --venue 名古屋 --backtest
  python venue_model.py --venue 名古屋 --tune --trials 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import kawasaki_predict as kp

VENUE_EN = {
    "川崎": "kawasaki", "大井": "oi", "船橋": "funabashi", "浦和": "urawa",
    "門別": "monbetsu", "園田": "sonoda", "姫路": "himeji", "名古屋": "nagoya",
    "金沢": "kanazawa", "笠松": "kasamatsu", "高知": "kochi", "佐賀": "saga",
}


def set_venue(venue: str):
    """kp のグローバル(当場・保存先)を場別に差し替える。"""
    en = VENUE_EN[venue]
    kp.VENUE = venue
    kp.PARAMS_PATH = Path(f"data/{en}_best_params.json")
    kp.OPTUNA_DB = Path(f"data/{en}_optuna.db")
    return en


def backtest_sweep(df_hist: pd.DataFrame, venue: str):
    """LeaveOneYearOut で各レースの model_score と popularity をキャッシュし、
    w_model をスイープして model/market/blend の top3/top5/top1 を比較する。"""
    _, _, k_horse, k_jockey = kp._load_params()
    df_v = df_hist[df_hist["venue"] == venue].copy()
    years = sorted(df_v["race_date"].dt.year.unique())
    if len(years) < 2:
        print(f"{venue}: バックテストに2年以上必要（保有 {years}）"); return

    races = []  # (actual_top3 set, model_rank, market_rank)
    for ty in years:
        df_tr = df_hist[df_hist["race_date"].dt.year < ty]
        df_te = df_v[df_v["race_date"].dt.year == ty]
        if df_tr.empty or df_te.empty:
            continue
        stats = kp.compute_stats(df_tr)
        X, y, g = kp.build_train_data(df_tr, k_horse=k_horse, k_jockey=k_jockey)
        if X.empty:
            continue
        model = kp.train_model(X, y, g)
        for _, rdf in df_te.groupby(df_te["race_date"].dt.strftime("%Y%m%d") + "_" + df_te["race_no"].astype(str)):
            rdf = rdf.reset_index(drop=True)
            if len(rdf) < 4:
                continue
            Xr = kp.build_features(rdf, stats, rdf["race_date"].iloc[0], k_horse=k_horse, k_jockey=k_jockey)
            scores, _ = kp.predict_race(model, Xr)
            pop = pd.to_numeric(rdf["popularity"], errors="coerce")
            if not pop.notna().any():
                continue
            mr = kp.market_rank_from_odds(pop)
            model_rank = pd.Series(-scores).rank(method="first").values - 1
            actual = set(np.where(rdf["finish_position"].values <= 3)[0])
            if len(actual) == 0:
                continue
            races.append((actual, model_rank, mr))
        print(f"  scored {ty} ({len(races)} races cum)", flush=True)

    if not races:
        print(f"{venue}: 評価可能レースなし"); return

    def ev(w):
        t3 = t5 = t1 = c = 0
        for actual, mrk, kr in races:
            d = min(3, len(actual))
            order = np.argsort(w * mrk + (1 - w) * kr, kind="stable")
            t3 += len(actual & set(order[:3])) / d
            t5 += len(actual & set(order[:5])) / d
            t1 += int(order[0] in actual); c += 1
        return t3 / c, t5 / c, t1 / c

    print(f"\n=== {venue} LeaveOneYearOut ブレンド比率スイープ ({len(races)}R) ===")
    print(f"{'w_model':>8}{'top3':>8}{'top5':>8}{'top1':>8}")
    best = (None, -1)
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        t3, t5, t1 = ev(w)
        tag = " 市場のみ" if w == 0 else (" モデルのみ" if w == 1 else "")
        print(f"{w:>8.1f}{t3:>8.1%}{t5:>8.1%}{t1:>8.1%}{tag}")
        if t5 > best[1]:
            best = (w, t5)
    print(f"\n→ {venue} top5最良: w_model={best[0]} (top5={best[1]:.1%})")
    print(f"  ※w=0(市場)とw=0.2(ブレンド)を比較し、モデルが上乗せできているか確認")


def main():
    p = argparse.ArgumentParser(description="地方競馬 場別モデル 構築/評価")
    p.add_argument("--venue", required=True)
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--metric", choices=["top5", "top3"], default="top5")
    args = p.parse_args()

    if args.venue not in VENUE_EN:
        raise SystemExit(f"未知の場: {args.venue}")
    en = set_venue(args.venue)
    print(f"場: {args.venue} ({en})  params={kp.PARAMS_PATH}  db={kp.OPTUNA_DB}")

    df = kp.load_history()
    n = int((df["venue"] == args.venue).sum())
    print(f"当場履歴: {n:,}行")
    if n < 1000:
        print(f"⚠ データが少ない（{n}行）。精度は限定的。")

    if args.tune:
        study = f"{en}_top5_v1" if args.metric == "top5" else f"{en}_top3_v1"
        kp.do_tune(df, n_trials=args.trials, metric=args.metric, study_name=study)
    elif args.backtest:
        backtest_sweep(df, args.venue)
    else:
        print("--backtest か --tune を指定してください")


if __name__ == "__main__":
    main()
