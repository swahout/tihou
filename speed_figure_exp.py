#!/usr/bin/env python
"""本物のスピード指数を作り、モデルが市場に迫れるか検証する実験。

現行 speed_idx はレース内中央値比＝絶対速度を消している。
新指数 speed_fig:
  1. speed = distance / finish_time (m/s)
  2. par = (venue,distance,track_cond) ごとの中央速度
  3. raw = speed - par  (parより何 m/s 速いか)
  4. daily variant = (race_date,venue) の raw 平均 を引く（その日の馬場速度を補正）
  → speed_fig は「馬場・距離・当日条件で補正した絶対的な速さ」

各馬の過去 speed_fig（causal shift）を特徴量化し、川崎 LOYO で
 baseline(現特徴) vs +speed_fig、および blend(市場) との比較を行う。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import kawasaki_predict as kp

SF_COLS = ["sf_recent", "sf_best", "sf_mean", "sf_last"]


def add_speed_fig(df):
    df = df.copy()
    df["_t"] = df["finish_time"].apply(kp._parse_time_secs)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    ok = (df["_t"] > 0) & df["distance"].notna() & (df["distance"] > 0)
    df["_spd"] = np.where(ok, df["distance"] / df["_t"].replace(0, np.nan), np.nan)
    tc = df["track_cond"].fillna("不明").replace("", "不明")
    df["_tc"] = tc
    par = df.groupby(["venue", "distance", "_tc"])["_spd"].transform("median")
    df["_raw"] = df["_spd"] - par
    dayvar = df.groupby([df["race_date"], df["venue"]])["_raw"].transform("mean")
    df["speed_fig"] = df["_raw"] - dayvar  # 0中心・速いほど大
    # 各馬の過去 speed_fig（causal: shift(1)）
    df = df.sort_values(["horse_name", "race_date", "race_no"])
    g = df.groupby("horse_name", sort=False)["speed_fig"]
    df["sf_recent"] = g.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).values
    df["sf_best"]   = g.apply(lambda s: s.shift(1).rolling(6, min_periods=1).max()).values
    df["sf_mean"]   = g.apply(lambda s: s.shift(1).expanding().mean()).values
    df["sf_last"]   = g.apply(lambda s: s.shift(1)).values
    return df


def main():
    df = kp.load_history(use_others_supplement=False)  # 川崎中心(南関+大井)で軽く
    df = add_speed_fig(df)
    cov = df["sf_recent"].notna().mean()
    print(f"speed_fig カバレッジ(sf_recent): {cov:.1%}")
    _, _, kh, kj = kp._load_params()
    feat = kp.FEATURE_COLS
    feat_sf = feat + SF_COLS

    dfk = df[df["venue"] == kp.VENUE].copy()
    years = sorted(dfk["race_date"].dt.year.unique())
    test_years = [y for y in years if (df["race_date"].dt.year < y).sum() > 0 and y in (2024, 2025, 2026)]

    R = {"base": [], "sf": [], "base_blend": [], "sf_blend": [], "market": []}

    # 自前ループで train/test 両方を sf 込みで構築
    def build_rows(src_df, stats, pred_date):
        Xs, sfs, fins, rids = [], [], [], []
        for rid, rdf in src_df.groupby(src_df["race_date"].dt.strftime("%Y%m%d") + "_" + src_df["race_no"].astype(str)):
            rdf = rdf.reset_index(drop=True)
            if len(rdf) < 3:
                continue
            Xr = kp.build_features(rdf, stats, pred_date, k_horse=kh, k_jockey=kj)
            Xs.append(Xr)
            sfs.append(rdf[SF_COLS].values)
            fins.append(rdf["finish_position"].values)
            rids.append(np.array([rid] * len(rdf)))
        X = pd.concat(Xs, ignore_index=True)
        sf = pd.DataFrame(np.vstack(sfs), columns=SF_COLS)
        for c in SF_COLS:
            X[c] = sf[c].values
        return X, np.concatenate(fins), np.concatenate(rids)

    def metrics(order, actual, d):
        return (len(actual & set(order[:3])) / d, len(actual & set(order[:5])) / d, int(order[0] in actual))

    for ty in test_years:
        tr = df[(df["race_date"].dt.year < ty)]
        stats = kp.compute_stats(tr)
        Xtr, fintr, ridtr = build_rows(tr[tr["venue"].isin(df["venue"].unique())], stats, pd.Timestamp(f"{ty}-01-01"))
        # ラベル
        ytr = np.zeros(len(Xtr)); groups = []
        for r in pd.unique(ridtr):
            idx = np.where(ridtr == r)[0]; n = len(idx)
            ytr[idx] = n - fintr[idx] + 1; groups.append(n)
        med = Xtr[SF_COLS].median()
        Xtr_f = Xtr.copy()
        for c in SF_COLS: Xtr_f[c] = Xtr_f[c].fillna(med[c])
        m_base = lgb.train({**kp._DEFAULT_PARAMS}, lgb.Dataset(Xtr[feat], label=ytr, group=groups), num_boost_round=kp._DEFAULT_ROUNDS)
        m_sf   = lgb.train({**kp._DEFAULT_PARAMS}, lgb.Dataset(Xtr_f[feat_sf], label=ytr, group=groups), num_boost_round=kp._DEFAULT_ROUNDS)

        te = dfk[dfk["race_date"].dt.year == ty]
        for rid, rdf in te.groupby(te["race_date"].dt.strftime("%Y%m%d") + "_" + te["race_no"].astype(str)):
            rdf = rdf.reset_index(drop=True)
            if len(rdf) < 4: continue
            Xr = kp.build_features(rdf, stats, rdf["race_date"].iloc[0], k_horse=kh, k_jockey=kj)
            for c in SF_COLS: Xr[c] = rdf[c].fillna(med[c]).values
            actual = set(np.where(rdf["finish_position"].values <= 3)[0]); d = min(3, len(actual))
            if d == 0: continue
            pop = pd.to_numeric(rdf["popularity"], errors="coerce")
            if not pop.notna().any(): continue
            mr = kp.market_rank_from_odds(pop)
            sb = m_base.predict(Xr[feat]); ss = m_sf.predict(Xr[feat_sf])
            rb = pd.Series(-sb).rank(method="first").values - 1
            rs = pd.Series(-ss).rank(method="first").values - 1
            w = 0.2
            R["base"].append(metrics(np.argsort(-sb, kind="stable"), actual, d))
            R["sf"].append(metrics(np.argsort(-ss, kind="stable"), actual, d))
            R["base_blend"].append(metrics(np.argsort(w*rb+(1-w)*mr, kind="stable"), actual, d))
            R["sf_blend"].append(metrics(np.argsort(w*rs+(1-w)*mr, kind="stable"), actual, d))
            R["market"].append(metrics(np.argsort(mr, kind="stable"), actual, d))
        print(f"  scored {ty}", flush=True)

    print(f"\n=== 川崎 speed_fig 検証 ({len(R['base'])}R, LOYO) ===")
    print(f"{'手法':<22}{'top3':>8}{'top5':>8}{'top1':>8}")
    lab = {"base":"モデル(現特徴)","sf":"モデル+speed_fig","base_blend":"ブレンド(現)","sf_blend":"ブレンド+speed_fig","market":"市場のみ"}
    for k in ["base","sf","market","base_blend","sf_blend"]:
        a = np.array(R[k]); print(f"{lab[k]:<22}{a[:,0].mean():>8.1%}{a[:,1].mean():>8.1%}{a[:,2].mean():>8.1%}")


if __name__ == "__main__":
    main()
