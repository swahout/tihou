#!/usr/bin/env python
"""過去の人気（履歴）を特徴量にすれば、当日のライブ人気を使わずに
市場のギャップを埋められるか？を検証する。

当日 popularity は変動するので使いたくない、という要望に対し、
「その馬の過去レースでの人気の平均/直近」=確立した市場評価を特徴量にする。
これは履歴に安定して存在し、当日オッズを必要としない。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import kawasaki_predict as kp

LAG_COLS = ["lag_pop_mean", "lag_pop_recent", "lag_prev_pop", "lag_poprate_mean"]


def precompute_lagpop(df_hist):
    """各 (horse, race) について、その馬の過去レースのみから人気系ラグ特徴を作る。
    当日レースは含めない（shift)。poprate = popularity / field_size（場依存を緩和）。"""
    df = df_hist.sort_values(["horse_name", "race_date", "race_no"]).copy()
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["poprate"] = df["popularity"] / df["field_size"].clip(lower=1)
    g = df.groupby("horse_name", sort=False)
    df["lag_pop_mean"] = g["popularity"].apply(lambda s: s.shift(1).expanding().mean()).values
    df["lag_pop_recent"] = g["popularity"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).values
    df["lag_prev_pop"] = g["popularity"].shift(1).values
    df["lag_poprate_mean"] = g["poprate"].apply(lambda s: s.shift(1).expanding().mean()).values
    return df[["race_date", "race_no", "venue", "horse_name"] + LAG_COLS]


def build_all(df_hist, lagdf, k_horse, k_jockey):
    df = df_hist.copy()
    df["_year"] = df["race_date"].dt.year
    years = sorted(df["_year"].unique())
    key = ["race_date", "race_no", "venue", "horse_name"]
    Xs, pops, fins, yrs, rids, lags = [], [], [], [], [], []
    for i, year in enumerate(years):
        if i < 1:
            continue
        df_before = df[df["_year"] < year]
        df_test = df[(df["_year"] == year) & (df["venue"] == kp.VENUE)]
        if df_before.empty or df_test.empty:
            continue
        stats = kp.compute_stats(df_before)
        pred_date = pd.Timestamp(f"{year}-01-01")
        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            race_df = race_df.reset_index(drop=True)
            if len(race_df) < 3:
                continue
            pop = pd.to_numeric(race_df["popularity"], errors="coerce").values
            if np.isnan(pop).all():
                continue
            Xr = kp.build_features(race_df, stats, pred_date, k_horse=k_horse, k_jockey=k_jockey)
            merged = race_df[key].merge(lagdf, on=key, how="left")
            Xs.append(Xr)
            lags.append(merged[LAG_COLS].values)
            pops.append(pop)
            fins.append(race_df["finish_position"].values)
            yrs.append(np.full(len(race_df), year))
            rids.append(np.array([race_id] * len(race_df)))
    X = pd.concat(Xs, ignore_index=True)
    lag = pd.DataFrame(np.vstack(lags), columns=LAG_COLS)
    return (X, lag, np.concatenate(pops), np.concatenate(fins),
            np.concatenate(yrs), np.concatenate(rids))


def metrics(order, actual, denom):
    return (len(actual & set(order[:3])) / denom,
            len(actual & set(order[:5])) / denom,
            int(order[0] in actual))


def train_rank(Xcols, y_lab, groups, X):
    dtr = lgb.Dataset(X[Xcols], label=y_lab, group=groups)
    return lgb.train({**kp._DEFAULT_PARAMS}, dtr, num_boost_round=kp._DEFAULT_ROUNDS)


def main():
    df_hist = kp.load_history(use_oi_supplement=True, use_nankan_supplement=True)
    _, _, k_horse, k_jockey = kp._load_params()
    feat = kp.FEATURE_COLS

    print("人気ラグ特徴を計算中...", flush=True)
    lagdf = precompute_lagpop(df_hist)
    print("全レース特徴量を構築中（1パス）...", flush=True)
    X, lag, pop, fin, yr, rid = build_all(df_hist, lagdf, k_horse, k_jockey)
    # ラグ特徴の欠損（初出走馬）はレース内中央値で埋める
    for c in LAG_COLS:
        X[c] = lag[c].values
    cov = lag["lag_pop_mean"].notna().mean()
    print(f"  {len(X):,}行 / {len(np.unique(rid)):,}レース  lag_pop カバレッジ {cov:.1%}\n")

    feat_lag = feat + LAG_COLS
    test_years = [y for y in sorted(set(yr)) if (yr < y).sum() > 0]
    R = {"model": [], "model_lag": [], "market": []}

    for ty in test_years:
        tr, te = yr < ty, yr == ty
        Xtr, Xte = X[tr].reset_index(drop=True), X[te].reset_index(drop=True)
        # ラグ欠損はレース横断の中央値で補完
        med = Xtr[LAG_COLS].median()
        Xtr2 = Xtr.copy(); Xte2 = Xte.copy()
        for c in LAG_COLS:
            Xtr2[c] = Xtr2[c].fillna(med[c]); Xte2[c] = Xte2[c].fillna(med[c])

        rid_tr = rid[tr]
        ytr = np.zeros(tr.sum()); groups_tr = []
        for r in pd.unique(rid_tr):
            idx = np.where(rid_tr == r)[0]; n = len(idx)
            ytr[idx] = n - fin[tr][idx] + 1; groups_tr.append(n)

        m_base = train_rank(feat, ytr, groups_tr, Xtr2)
        m_lag = train_rank(feat_lag, ytr, groups_tr, Xtr2)

        s_base = m_base.predict(Xte2[feat])
        s_lag = m_lag.predict(Xte2[feat_lag])
        rid_te, pop_te, fin_te = rid[te], pop[te], fin[te]
        for r in pd.unique(rid_te):
            idx = np.where(rid_te == r)[0]
            actual = set(np.where(fin_te[idx] <= 3)[0]); denom = min(3, len(actual))
            if denom == 0:
                continue
            mr = pd.Series(pop_te[idx]).rank(method="first").values - 1
            R["model"].append(metrics(np.argsort(-s_base[idx], kind="stable"), actual, denom))
            R["model_lag"].append(metrics(np.argsort(-s_lag[idx], kind="stable"), actual, denom))
            R["market"].append(metrics(np.argsort(mr, kind="stable"), actual, denom))
        print(f"  {ty}: scored", flush=True)

    print("\n=== 過去人気を特徴量に追加（当日人気は不使用）===")
    print(f"{'手法':<22}{'top3_hit':>10}{'top5_cov':>10}{'top1_acc':>10}")
    lab = {"model":"モデル(現行33特徴)","model_lag":"+過去人気ラグ(履歴のみ)","market":"市場(ライブ人気,参考)"}
    for k in ["model","model_lag","market"]:
        a = np.array(R[k])
        print(f"{lab[k]:<22}{a[:,0].mean():>10.1%}{a[:,1].mean():>10.1%}{a[:,2].mean():>10.1%}")

    # 過去人気ラグの単独 importance を見る
    fi = pd.Series(m_lag.feature_importance(importance_type="gain"), index=feat_lag)
    print("\n[最終foldの特徴量重要度 top12]")
    for n, v in fi.sort_values(ascending=False).head(12).items():
        mark = " ★ラグ" if n in LAG_COLS else ""
        print(f"  {n:<22}{v:>12,.0f}{mark}")


if __name__ == "__main__":
    main()
