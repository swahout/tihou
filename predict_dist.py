#!/usr/bin/env python
"""
v8: 分散予測（分位点回帰）+ モンテカルロ・シミュレーション。

各馬の speed_idx（レース相対速度, 高い=速い）を点ではなく分布で予測する。
LightGBM の quantile 回帰を複数αで学習 → 各馬の逆CDF → N回サンプルして
レース内順位を出し、各馬の P(top3) を集計 → P(top3) 上位で選択する。

「ムラのある馬／堅実な馬」を区別できるため、点予測(LambdaRank)と top-k 選択が
変わりうる = top3_hit を動かせる可能性がある唯一の筋。

評価は eval_recent と同じ held-out 方式（対象日より前で学習→当日予測）。
比較対象として「分布の中央値で点ランキングした場合」も併記する。

使い方:
  python predict_dist.py --dates 2026/06/15 2026/06/16
  python predict_dist.py --dates 2026/06/15 2026/06/16 --nsims 2000
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

import kawasaki_predict as kp

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
_NUM_ROUNDS = 300


def build_dist_train(df_hist: pd.DataFrame, k_horse: int, k_jockey: int):
    """build_train_data と同じ年次ローリングで特徴量を作り、
    ターゲットを finish_position ではなく speed_idx にする（リークなし）。"""
    df = df_hist.copy()
    df["_year"] = df["race_date"].dt.year
    years = sorted(df["_year"].unique())
    all_X, all_t = [], []
    for i, year in enumerate(years):
        if i < 1:
            continue
        df_test = df[df["_year"] == year]
        df_before = df[df["_year"] < year]
        if df_before.empty:
            continue
        stats = kp.compute_stats(df_before)
        pred_date = pd.Timestamp(f"{year}-01-01")
        for _, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            rdf = race_df.reset_index(drop=True)
            Xr = kp.build_features(rdf, stats, pred_date, k_horse=k_horse, k_jockey=k_jockey)
            t = pd.to_numeric(rdf["speed_idx"], errors="coerce").values
            all_X.append(Xr)
            all_t.append(t)
    if not all_X:
        return pd.DataFrame(), np.array([])
    return pd.concat(all_X, ignore_index=True), np.concatenate(all_t)


def train_quantile_models(X: pd.DataFrame, t: np.ndarray) -> dict:
    Xf = X[kp.FEATURE_COLS].fillna(X[kp.FEATURE_COLS].median())
    mask = ~np.isnan(t)
    Xf, t = Xf[mask], t[mask]
    base = {
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": -1,
    }
    models = {}
    ds = lgb.Dataset(Xf, label=t)
    for a in QUANTILES:
        params = {**base, "objective": "quantile", "alpha": a, "metric": "quantile"}
        models[a] = lgb.train(params, ds, num_boost_round=_NUM_ROUNDS)
    return models


def predict_quantiles(models: dict, X: pd.DataFrame) -> np.ndarray:
    """(n_horses, n_quantiles) の予測。行ごとに昇順を強制（交差防止）。"""
    Xf = X[kp.FEATURE_COLS].fillna(X[kp.FEATURE_COLS].median())
    alphas = sorted(models.keys())
    Q = np.column_stack([models[a].predict(Xf) for a in alphas])
    return np.sort(Q, axis=1)  # 単調化


def simulate_p_top3(Q: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """各馬の分位点 Q から逆CDFサンプル → レース内top3出現確率。"""
    alphas = np.array(sorted(QUANTILES))
    n_horses = Q.shape[0]
    U = rng.random((n_sims, n_horses))
    samples = np.empty((n_sims, n_horses))
    for h in range(n_horses):
        samples[:, h] = np.interp(U[:, h], alphas, Q[h])  # 端は最小/最大分位にクリップ
    # 各シムで speed_idx 上位3頭が「3着内」
    top3 = np.argsort(-samples, axis=1)[:, :3]
    counts = np.zeros(n_horses)
    np.add.at(counts, top3.ravel(), 1)
    return counts / n_sims


def _metrics(actual_top3: set, pred_order, denom):
    t3 = len(actual_top3 & set(pred_order[:3])) / denom
    t5 = len(actual_top3 & set(pred_order[:5])) / denom
    t1 = int(pred_order[0] in actual_top3)
    return t3, t5, t1


def eval_dates(df_hist, dates, n_sims, k_horse, k_jockey):
    overall = {"mc": {"all": [], "r8": []}, "med": {"all": [], "r8": []}}
    rng = np.random.default_rng(42)

    for date_str in dates:
        pred_date = pd.Timestamp(date_str.replace("/", "-"))
        df_train = df_hist[df_hist["race_date"] < pred_date].copy()
        df_day = df_hist[(df_hist["venue"] == kp.VENUE) & (df_hist["race_date"] == pred_date)].copy()
        if df_day.empty or df_train.empty:
            print(f"\n{date_str}: スキップ")
            continue

        X_tr, t_tr = build_dist_train(df_train, k_horse, k_jockey)
        models = train_quantile_models(X_tr, t_tr)
        stats = kp.compute_stats(df_train)

        day = {"mc": {"all": [], "r8": []}, "med": {"all": [], "r8": []}}
        for _, race_df in df_day.groupby(
            df_day["race_date"].dt.strftime("%Y%m%d") + "_" + df_day["race_no"].astype(str)
        ):
            rdf = race_df.reset_index(drop=True)
            if len(rdf) < 3:
                continue
            Xr = kp.build_features(rdf, stats, rdf["race_date"].iloc[0], k_horse=k_horse, k_jockey=k_jockey)
            Q = predict_quantiles(models, Xr)
            med_idx = sorted(QUANTILES).index(0.5)

            actual_top3 = set(rdf.index[rdf["finish_position"] <= 3])
            denom = min(3, len(actual_top3))
            race_no = int(rdf["race_no"].iloc[0])

            # MC: P(top3) 降順
            p_top3 = simulate_p_top3(Q, n_sims, rng)
            mc_order = list(np.argsort(-p_top3))
            # 点: 中央値 speed_idx 降順
            med_order = list(np.argsort(-Q[:, med_idx]))

            for key, order in (("mc", mc_order), ("med", med_order)):
                m = _metrics(actual_top3, order, denom)
                day[key]["all"].append(m)
                overall[key]["all"].append(m)
                if race_no >= 8:
                    day[key]["r8"].append(m)
                    overall[key]["r8"].append(m)

        for scope, label in (("all", "全レース"), ("r8", "8R以降")):
            if not day["mc"][scope]:
                continue
            def avg(key, j):
                return np.mean([x[j] for x in day[key][scope]])
            print(f"\n{date_str} [{label}] {len(day['mc'][scope])}R")
            print(f"   MC(分布) : top3_hit={avg('mc',0):.1%}  top5={avg('mc',1):.1%}  top1={avg('mc',2):.1%}")
            print(f"   中央値点 : top3_hit={avg('med',0):.1%}  top5={avg('med',1):.1%}  top1={avg('med',2):.1%}")

    print("\n" + "=" * 55)
    for scope, label in (("all", "全レース"), ("r8", "8R以降")):
        if not overall["mc"][scope]:
            continue
        def avg(key, j):
            return np.mean([x[j] for x in overall[key][scope]])
        print(f"合計 {dates} [{label}] {len(overall['mc'][scope])}R")
        print(f"   MC(分布) : top3_hit={avg('mc',0):.1%}  top5={avg('mc',1):.1%}  top1={avg('mc',2):.1%}")
        print(f"   中央値点 : top3_hit={avg('med',0):.1%}  top5={avg('med',1):.1%}  top1={avg('med',2):.1%}")


def eval_backtest_years(df_hist, years, n_sims, k_horse, k_jockey):
    """大サンプル検証: 各 test_year を「その年より前の全データ」で学習し、
    その年の川崎全レースで MC(分布) vs 中央値点 を比較する。"""
    rng = np.random.default_rng(42)
    for year in years:
        df_train = df_hist[df_hist["race_date"].dt.year < year].copy()
        df_test = df_hist[(df_hist["venue"] == kp.VENUE) & (df_hist["race_date"].dt.year == year)].copy()
        if df_train.empty or df_test.empty:
            print(f"\n{year}: スキップ")
            continue
        X_tr, t_tr = build_dist_train(df_train, k_horse, k_jockey)
        models = train_quantile_models(X_tr, t_tr)
        stats = kp.compute_stats(df_train)

        agg = {"mc": {"all": [], "r8": []}, "med": {"all": [], "r8": []}}
        for _, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            rdf = race_df.reset_index(drop=True)
            if len(rdf) < 3:
                continue
            Xr = kp.build_features(rdf, stats, rdf["race_date"].iloc[0], k_horse=k_horse, k_jockey=k_jockey)
            Q = predict_quantiles(models, Xr)
            med_idx = sorted(QUANTILES).index(0.5)
            actual_top3 = set(rdf.index[rdf["finish_position"] <= 3])
            denom = min(3, len(actual_top3))
            race_no = int(rdf["race_no"].iloc[0])
            p_top3 = simulate_p_top3(Q, n_sims, rng)
            for key, order in (("mc", list(np.argsort(-p_top3))),
                               ("med", list(np.argsort(-Q[:, med_idx])))):
                m = _metrics(actual_top3, order, denom)
                agg[key]["all"].append(m)
                if race_no >= 8:
                    agg[key]["r8"].append(m)

        for scope, label in (("all", "全レース"), ("r8", "8R以降")):
            if not agg["mc"][scope]:
                continue
            def avg(key, j):
                return np.mean([x[j] for x in agg[key][scope]])
            print(f"\n{year} [{label}] {len(agg['mc'][scope])}R")
            print(f"   MC(分布) : top3_hit={avg('mc',0):.1%}  top5={avg('mc',1):.1%}  top1={avg('mc',2):.1%}")
            print(f"   中央値点 : top3_hit={avg('med',0):.1%}  top5={avg('med',1):.1%}  top1={avg('med',2):.1%}")


def main():
    parser = argparse.ArgumentParser(description="v8 分散予測+モンテカルロ評価")
    parser.add_argument("--dates", nargs="+", help="特定日の held-out 評価")
    parser.add_argument("--backtest-years", nargs="+", type=int, help="年次大サンプル検証 e.g. 2024 2025 2026")
    parser.add_argument("--nsims", type=int, default=1000)
    parser.add_argument("--no-nankan", action="store_true")
    args = parser.parse_args()

    _, _, k_horse, k_jockey = kp._load_params()
    df_hist = kp.load_history(use_nankan_supplement=not args.no_nankan)
    print(f"  QUANTILES={QUANTILES}  nsims={args.nsims}  K_HORSE={k_horse} K_JOCKEY={k_jockey}")
    if args.backtest_years:
        eval_backtest_years(df_hist, args.backtest_years, args.nsims, k_horse, k_jockey)
    elif args.dates:
        eval_dates(df_hist, args.dates, args.nsims, k_horse, k_jockey)
    else:
        parser.error("--dates か --backtest-years のどちらかが必要")


if __name__ == "__main__":
    main()
