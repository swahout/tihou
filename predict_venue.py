#!/usr/bin/env python3
"""任意の地方競馬場の当日予想（汎用版）。

kawasaki_predict.py のモデル基盤(v10特徴量 + 市場ブレンド)をそのまま使い、
VENUE を差し替えて他場(名古屋・門別・園田 等)に適用する。

前提:
  - その場の履歴が data/ に存在し load_history で読まれること
    (名古屋/門別/園田/金沢 は data/historical_others/{venue_en}_*.csv)
  - 当日出走表: python collect_shutuba.py --venue 名古屋 --date ...

Usage:
  python predict_venue.py --venue 名古屋 --date 2026/06/18
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kawasaki_predict as kp

VENUE_EN = {
    "川崎": "kawasaki", "大井": "oi", "船橋": "funabashi", "浦和": "urawa",
    "門別": "monbetsu", "園田": "sonoda", "姫路": "himeji", "名古屋": "nagoya",
    "金沢": "kanazawa", "笠松": "kasamatsu", "高知": "kochi", "佐賀": "saga",
}


def main():
    p = argparse.ArgumentParser(description="地方競馬 当日予想（汎用）")
    p.add_argument("--venue", required=True)
    p.add_argument("--date", required=True, help="予測日 YYYY/MM/DD")
    args = p.parse_args()

    venue, date_str = args.venue, args.date
    venue_en = VENUE_EN.get(venue, venue)
    # モデル基盤の当場を差し替え（compute_stats/build_features が参照）
    kp.VENUE = venue

    dt = datetime.strptime(date_str, "%Y/%m/%d")
    shutuba_path = kp.RACES_DIR / f"{venue_en}_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    if not shutuba_path.exists():
        print(f"出走表なし: {shutuba_path}")
        print(f"まず: python collect_shutuba.py --venue {venue} --date {date_str}")
        return

    df_hist = kp.load_history()
    n_home = int((df_hist["venue"] == venue).sum())
    print(f"当場({venue})履歴: {n_home:,}行")
    if n_home < 500:
        print(f"⚠ 当場データが少なく予測の信頼度は限定的です（{n_home}行）")

    df_sh = pd.read_csv(shutuba_path, encoding="utf-8-sig")
    print(f"出走表: {shutuba_path} ({len(df_sh)}頭, {df_sh['race_no'].nunique()}R)")

    pred_date = pd.Timestamp(dt)
    df_train = df_hist[df_hist["race_date"] < pred_date].copy()
    if df_train.empty:
        print("学習データなし")
        return

    _, _, k_horse, k_jockey = kp._load_params()
    stats = kp.compute_stats(df_train)
    X, y, g = kp.build_train_data(df_train, k_horse=k_horse, k_jockey=k_jockey)
    if X.empty:
        print("学習データ不足")
        return
    print(f"学習データ: {len(X):,}行, {len(np.unique(g)):,}レース  学習中...")
    model = kp.train_model(X, y, g)

    out_rows = []
    for race_no, race_df in df_sh.groupby("race_no"):
        if len(race_df) < 2:
            continue
        race_df = race_df.copy().reset_index(drop=True)
        Xr = kp.build_features(race_df, stats, pred_date, k_horse=k_horse, k_jockey=k_jockey)
        scores, shap_vals = kp.predict_race(model, Xr)
        exp_s = np.exp(scores - scores.max())
        raw_prob = exp_s / exp_s.sum() * 3.0
        base = 3.0 / max(len(race_df), 1)
        rel = Xr["data_reliability"].values
        top3_prob = raw_prob * rel + base * (1 - rel)
        odds = race_df["win_odds"].values if "win_odds" in race_df.columns else None
        rank_score, mkt_rank = kp.blend_order_score(scores, odds)
        reasons = [kp.make_reason(shap_vals[i], kp.FEATURE_COLS, r) if shap_vals is not None else ""
                   for i, r in Xr.iterrows()]
        race_df["score"] = scores
        race_df["top3_prob"] = top3_prob
        race_df["rank_score"] = rank_score
        race_df["market_pop"] = (mkt_rank + 1).astype(int) if mkt_rank is not None else 0
        race_df["data_reliability"] = rel
        race_df["n_hist"] = Xr["n_total"].values
        race_df["n_venue"] = Xr["n_venue"].values
        race_df["reason"] = reasons
        out_rows.append(race_df)

    if not out_rows:
        print("予測失敗")
        return
    df_pred = pd.concat(out_rows, ignore_index=True)
    out_path = kp.RACES_DIR / f"{venue_en}_{dt.strftime('%Y_%m%d')}_prediction.csv"
    df_pred.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"予測保存: {out_path}")

    has_market = (df_pred["market_pop"] > 0).any()
    print(f"\n{'='*70}\n  {venue} {date_str} 予想"
          + (f"（市場ブレンド w_model={kp.BLEND_W_MODEL}）" if has_market else "（モデル単独）")
          + f"\n{'='*70}")
    for race_no, grp in df_pred.groupby("race_no"):
        top = grp.sort_values("rank_score", ascending=False).head(5)
        meta = top.iloc[0]
        picks = " → ".join(
            f"{int(h.horse_no)}{h.horse_name}({int(h.market_pop)}人)" if h.market_pop > 0
            else f"{int(h.horse_no)}{h.horse_name}"
            for _, h in top.iterrows())
        print(f"\n【{int(race_no)}R】{meta.get('race_name','')} {int(meta.get('distance',0))}m "
              f"{int(meta.get('field_size',0))}頭\n  {picks}")


if __name__ == "__main__":
    main()
