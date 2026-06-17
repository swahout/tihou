#!/usr/bin/env python
"""期待値ベース馬券のバックテスト。

仮説: 市場は人気馬を買いすぎ・穴を嫌う(favorite-longshot bias)ため、
「モデル確率 > 市場確率」の馬(=過小評価)を買えば、的中率が同じでも回収率で勝てる。

- 単勝(win)で検証。model P(win)=softmax(score)、market P(win)=(1/odds)正規化。
- 2026: 実オッズ(win_odds 82.9%充足)。2024-2025: 人気→中央オッズの近似。
- 払戻: 単勝的中(finish==1)で odds*100、外れ0。回収率=払戻/投資。
- NAR単勝の控除率は約25%(全買い回収率~75%)が基準線。これを超えれば妙味あり。
"""
import numpy as np
import pandas as pd
import kawasaki_predict as kp

# 人気→単勝オッズ 近似（2026実データの中央値, value_bet調査より）
POP_TO_ODDS = {1:2.0,2:3.8,3:6.3,4:9.4,5:14.7,6:21.4,7:32.0,8:45.6,
               9:69.4,10:101.3,11:141.8,12:185.8,13:183.9,14:256.3,15:300.0,16:350.0}


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def collect_bets(df_hist, test_years, use_real_odds):
    _, _, k_horse, k_jockey = kp._load_params()
    df_kw = df_hist[df_hist["venue"] == kp.VENUE].copy()
    rows = []
    for ty in test_years:
        df_train = df_hist[df_hist["race_date"].dt.year < ty].copy()
        df_test = df_kw[df_kw["race_date"].dt.year == ty].copy()
        if df_train.empty or df_test.empty:
            continue
        stats = kp.compute_stats(df_train)
        X, y, g = kp.build_train_data(df_train, k_horse=k_horse, k_jockey=k_jockey)
        model = kp.train_model(X, y, g)
        for _, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            race_df = race_df.reset_index(drop=True)
            if len(race_df) < 4:
                continue
            Xr = kp.build_features(race_df, stats, race_df["race_date"].iloc[0],
                                   k_horse=k_horse, k_jockey=k_jockey)
            scores, _ = kp.predict_race(model, Xr)
            p_model = softmax(scores)
            pop = pd.to_numeric(race_df["popularity"], errors="coerce").values
            if use_real_odds:
                odds = pd.to_numeric(race_df["win_odds"], errors="coerce").values
            else:
                odds = np.array([POP_TO_ODDS.get(int(p), np.nan) if not np.isnan(p) else np.nan for p in pop])
            fin = race_df["finish_position"].values
            valid = ~np.isnan(odds) & (odds > 0)
            if valid.sum() < 3:
                continue
            # 市場勝率(オーバーラウンド正規化)
            inv = np.where(valid, 1.0/np.where(valid, odds, 1), 0.0)
            p_mkt = inv / inv.sum()
            for i in range(len(race_df)):
                if not valid[i]:
                    continue
                rows.append({
                    "ty": ty, "p_model": p_model[i], "p_mkt": p_mkt[i],
                    "odds": odds[i], "won": int(fin[i] == 1), "pop": pop[i],
                })
    return pd.DataFrame(rows)


def report(bets, label):
    print(f"\n========== {label} ({len(bets)}頭分) ==========")
    # ベースライン: 全頭ベット
    def roi(sub):
        if len(sub) == 0: return (0,0,0.0,0.0)
        stake = len(sub)*100
        ret = (sub["odds"]*100*sub["won"]).sum()
        return len(sub), sub["won"].sum(), ret/stake*100, sub["won"].mean()*100
    n,w,r,hit = roi(bets)
    print(f"[基準] 全頭買い         : {n:5d}点 的中{hit:4.1f}% 回収率 {r:5.1f}%")
    # 妙味: model > market を倍率で段階的に
    bets = bets.copy()
    bets["ratio"] = bets["p_model"] / bets["p_mkt"].clip(lower=1e-9)
    for thr in [1.0, 1.2, 1.5, 2.0, 3.0]:
        sub = bets[bets["ratio"] >= thr]
        n,w,r,hit = roi(sub)
        print(f"[妙味] model/market>={thr:>3.1f} : {n:5d}点 的中{hit:4.1f}% 回収率 {r:5.1f}%")
    # 逆: 市場が過大評価(model < market)を買うとどうなるか(対照)
    sub = bets[bets["ratio"] <= 0.7]
    n,w,r,hit = roi(sub)
    print(f"[対照] model/market<=0.7: {n:5d}点 的中{hit:4.1f}% 回収率 {r:5.1f}%")
    # 人気帯別の妙味(model>market)回収率
    print("  人気別 [model/market>=1.5]:")
    s2 = bets[bets["ratio"]>=1.5]
    for lo,hi,lab in [(1,3,"1-3人気"),(4,6,"4-6人気"),(7,99,"7人気以下")]:
        ss=s2[(s2["pop"]>=lo)&(s2["pop"]<=hi)]
        n,w,r,hit=roi(ss)
        if n: print(f"    {lab}: {n:4d}点 的中{hit:4.1f}% 回収率 {r:5.1f}%")


def main():
    df = kp.load_history(use_oi_supplement=True, use_nankan_supplement=True)
    print("\n=== 単勝・期待値ベース回収率バックテスト ===")
    real = collect_bets(df, [2026], use_real_odds=True)
    report(real, "2026 実オッズ")
    proxy = collect_bets(df, [2024, 2025], use_real_odds=False)
    report(proxy, "2024-2025 近似オッズ(人気→中央値)")


if __name__ == "__main__":
    main()
