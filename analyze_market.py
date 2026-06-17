#!/usr/bin/env python
"""市場(人気)は我々の特徴量で再現できるか？を検証する分析スクリプト。

問い:
  1. なぜ人気が高いのか → 特徴量から popularity を予測できるか（再現性）
  2. 人気の高い馬はなぜ勝つのか → 特徴量だけで市場順位を作れば、ライブ人気を
     使わずに同等の精度が出せるか
  3. 人気には特徴量に無い「上乗せ情報」がどれだけ残っているか（残差価値）

LeaveOneYearOut。1パスで全レースの特徴量を構築（year-rolling stats, リーク無し）。
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
import kawasaki_predict as kp


def build_all(df_hist, k_horse, k_jockey):
    """全川崎レースの per-race 特徴量 + popularity + finish + year + race_id を構築。
    各行の stats はその馬のレース年より前のデータのみ（build_train_data と同じ）。"""
    df = df_hist.copy()
    df["_year"] = df["race_date"].dt.year
    years = sorted(df["_year"].unique())
    Xs, pops, fins, yrs, rids = [], [], [], [], []
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
            Xr = kp.build_features(race_df, stats, pred_date, k_horse=k_horse, k_jockey=k_jockey)
            pop = pd.to_numeric(race_df["popularity"], errors="coerce").values
            if np.isnan(pop).all():
                continue
            Xs.append(Xr)
            pops.append(pop)
            fins.append(race_df["finish_position"].values)
            yrs.append(np.full(len(race_df), year))
            rids.append(np.array([race_id] * len(race_df)))
    X = pd.concat(Xs, ignore_index=True)
    return (X, np.concatenate(pops), np.concatenate(fins),
            np.concatenate(yrs), np.concatenate(rids))


def race_metrics(order, actual_top3, denom):
    return (len(actual_top3 & set(order[:3])) / denom,
            len(actual_top3 & set(order[:5])) / denom,
            int(order[0] in actual_top3))


def main():
    df_hist = kp.load_history(use_oi_supplement=True, use_nankan_supplement=True)
    _, _, k_horse, k_jockey = kp._load_params()
    feat = kp.FEATURE_COLS

    print("全レース特徴量を構築中（1パス）...", flush=True)
    X, pop, fin, yr, rid = build_all(df_hist, k_horse, k_jockey)
    print(f"  {len(X):,}行 / {len(np.unique(rid)):,}レース  年: {sorted(set(yr))}\n")

    test_years = [y for y in sorted(set(yr)) if (yr < y).sum() > 0]

    # 集計バケツ
    M = {"model": [], "market": [], "mimic": [], "blend_live": [], "blend_mimic": []}
    spearman_pop = []   # 市場再現モデルの per-race 人気順位相関
    pop_lift = []       # finishモデルに人気を足したときの top3_hit lift（残差価値）

    for ty in test_years:
        tr = yr < ty
        te = yr == ty
        Xtr, Xte = X[tr], X[te]

        # --- M_finish: 現行モデル（LambdaRank, finish由来ラベル）---
        # rid 単位で group を作る
        rid_tr = rid[tr]
        # ラベル: レース内 n - finish + 1
        ytr = np.zeros(tr.sum())
        groups_tr = []
        for r in pd.unique(rid_tr):
            idx = np.where(rid_tr == r)[0]
            f = fin[tr][idx]
            n = len(idx)
            ytr[idx] = n - f + 1
            groups_tr.append(n)
        dtr = lgb.Dataset(Xtr[feat], label=ytr, group=groups_tr)
        m_finish = lgb.train({**kp._DEFAULT_PARAMS}, dtr, num_boost_round=kp._DEFAULT_ROUNDS)

        # --- M_pop: 市場再現モデル（特徴量→popularity 回帰, 小さいほど人気）---
        m_pop = lgb.train(
            {"objective": "regression", "metric": "l2", "learning_rate": 0.05,
             "num_leaves": 31, "verbose": -1, "n_jobs": -1},
            lgb.Dataset(Xtr[feat], label=pop[tr]), num_boost_round=400)

        # --- M_finish + popularity を足したモデル（残差価値の測定）---
        Xtr2 = Xtr[feat].copy(); Xtr2["popularity"] = pop[tr]
        dtr2 = lgb.Dataset(Xtr2, label=ytr, group=groups_tr)
        m_finish_pop = lgb.train({**kp._DEFAULT_PARAMS}, dtr2, num_boost_round=kp._DEFAULT_ROUNDS)

        # --- 評価 ---
        s_model = m_finish.predict(Xte[feat])
        s_mimic = m_pop.predict(Xte[feat])  # 予測人気（小さい=人気）
        Xte2 = Xte[feat].copy(); Xte2["popularity"] = pop[te]
        s_finpop = m_finish_pop.predict(Xte2)

        rid_te = rid[te]; pop_te = pop[te]; fin_te = fin[te]
        day_lift = []
        for r in pd.unique(rid_te):
            idx = np.where(rid_te == r)[0]
            n = len(idx)
            actual = set(np.where(fin_te[idx] <= 3)[0])
            denom = min(3, len(actual))
            if denom == 0:
                continue
            sm = s_model[idx]; smi = s_mimic[idx]; pl = pop_te[idx]; sfp = s_finpop[idx]

            model_rank = pd.Series(-sm).rank(method="first").values - 1
            mimic_rank = pd.Series(smi).rank(method="first").values - 1   # 小さい予測=人気=上位
            market_rank = pd.Series(pl).rank(method="first").values - 1
            w = kp.BLEND_W_MODEL
            blend_live = pd.Series(w*model_rank + (1-w)*market_rank).values
            blend_mimic = pd.Series(w*model_rank + (1-w)*mimic_rank).values

            M["model"].append(race_metrics(np.argsort(-sm, kind="stable"), actual, denom))
            M["market"].append(race_metrics(np.argsort(market_rank, kind="stable"), actual, denom))
            M["mimic"].append(race_metrics(np.argsort(mimic_rank, kind="stable"), actual, denom))
            M["blend_live"].append(race_metrics(np.argsort(blend_live, kind="stable"), actual, denom))
            M["blend_mimic"].append(race_metrics(np.argsort(blend_mimic, kind="stable"), actual, denom))

            # 人気順位の再現相関
            if n >= 3:
                sp = spearmanr(mimic_rank, market_rank).correlation
                if not np.isnan(sp):
                    spearman_pop.append(sp)
            # 残差価値: finish+pop の top3_hit と finish単独の差
            day_lift.append(race_metrics(np.argsort(-sfp, kind="stable"), actual, denom)[0]
                            - race_metrics(np.argsort(-sm, kind="stable"), actual, denom)[0])
        pop_lift.extend(day_lift)
        print(f"  {ty}: scored", flush=True)

    print("\n=== 結果（全テストレース平均, LeaveOneYearOut）===")
    print(f"{'手法':<14}{'top3_hit':>10}{'top5_cov':>10}{'top1_acc':>10}")
    labels = {"model":"モデルのみ","market":"市場(ライブ人気)","mimic":"市場再現(特徴量)",
              "blend_live":"ブレンド(ライブ)","blend_mimic":"ブレンド(再現)"}
    for k in ["model","mimic","blend_mimic","market","blend_live"]:
        a = np.array(M[k])
        print(f"{labels[k]:<14}{a[:,0].mean():>10.1%}{a[:,1].mean():>10.1%}{a[:,2].mean():>10.1%}")

    print(f"\n市場再現モデルの人気順位相関 (Spearman, レース平均): {np.mean(spearman_pop):.3f}")
    print(f"finishモデルに popularity を加えた時の top3_hit lift（残差価値）: {np.mean(pop_lift):+.1%}")


if __name__ == "__main__":
    main()
