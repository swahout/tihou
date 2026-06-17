#!/usr/bin/env python
"""
特定日（例: 2026/06/15, 06/16）の予測精度を held-out で評価する。

各対象日について「その日より前の全データ」で学習し、その日の川崎レースを予測する
（本番運用と同じ条件）。top3_hit / top5_coverage / top1_acc を
全レースと 8R以降 の両方で出す。

使い方:
  python eval_recent.py --dates 2026/06/15 2026/06/16
  python eval_recent.py --dates 2026/06/15 2026/06/16 --no-nankan   # アブレーション
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import kawasaki_predict as kp


def _load_params_from(path: str | None):
    """指定JSONからパラメータを読む（kp._load_params と同じ解釈）。
    path=None なら本番 PARAMS_PATH を使う。
    Returns: (lgb_params, num_rounds, k_horse, k_jockey)
    """
    if path is None:
        return kp._load_params()
    saved = json.loads(Path(path).read_text())
    num_rounds = int(saved.pop("num_boost_round", kp._DEFAULT_ROUNDS))
    k_horse = int(saved.pop("k_horse", kp.K_HORSE))
    k_jockey = int(saved.pop("k_jockey", kp.K_JOCKEY))
    params = {**kp._DEFAULT_PARAMS, **saved}
    return params, num_rounds, k_horse, k_jockey


def _race_metrics(race_df: pd.DataFrame, scores: np.ndarray,
                  blend: bool = False) -> tuple[float, float, int]:
    """1レース分の (top3_hit, top5_coverage, top1_in_top3) を返す。
    blend=True なら市場人気(popularity)とのブレンド順位で評価する。"""
    actual_top3 = set(race_df.index[race_df["finish_position"] <= 3])
    if blend and "popularity" in race_df.columns:
        rank_score, _ = kp.blend_order_score(
            scores, pd.to_numeric(race_df["popularity"], errors="coerce").values
        )
        order = np.argsort(-rank_score)
    else:
        order = np.argsort(-scores)
    pred_sorted = race_df.index[order]
    denom = min(3, len(actual_top3))
    top3 = len(actual_top3 & set(pred_sorted[:3])) / denom
    top5 = len(actual_top3 & set(pred_sorted[:5])) / denom
    top1 = int(pred_sorted[0] in actual_top3)
    return top3, top5, top1


def eval_dates(df_hist: pd.DataFrame, dates: list[str], params_path: str | None = None,
               blend: bool = False) -> None:
    lgb_params, num_rounds, k_horse, k_jockey = _load_params_from(params_path)
    print(f"  K_HORSE={k_horse}  K_JOCKEY={k_jockey}  "
          f"(params: {params_path or kp.PARAMS_PATH})"
          + (f"  [市場ブレンド w_model={kp.BLEND_W_MODEL}]" if blend else ""))

    df_kw = df_hist[df_hist["venue"] == kp.VENUE].copy()
    overall = {"all": [], "r8": []}  # (top3, top5, top1) のリスト

    for date_str in dates:
        pred_date = pd.Timestamp(date_str.replace("/", "-"))
        df_train = df_hist[df_hist["race_date"] < pred_date].copy()
        df_day = df_kw[df_kw["race_date"] == pred_date].copy()
        if df_day.empty:
            print(f"\n{date_str}: 該当レースなし（スキップ）")
            continue
        if df_train.empty:
            print(f"\n{date_str}: 学習データなし（スキップ）")
            continue

        stats = kp.compute_stats(df_train)
        X_tr, y_tr, g_tr = kp.build_train_data(df_train, k_horse=k_horse, k_jockey=k_jockey)
        model = kp.train_model(X_tr, y_tr, g_tr, params=lgb_params, num_rounds=num_rounds)

        day = {"all": [], "r8": []}
        for _, race_df in df_day.groupby(
            df_day["race_date"].dt.strftime("%Y%m%d") + "_" + df_day["race_no"].astype(str)
        ):
            race_df = race_df.reset_index(drop=True)
            if len(race_df) < 3:
                continue
            Xr = kp.build_features(race_df, stats, race_df["race_date"].iloc[0],
                                   k_horse=k_horse, k_jockey=k_jockey)
            scores, _ = kp.predict_race(model, Xr)
            m = _race_metrics(race_df, scores, blend=blend)
            day["all"].append(m)
            overall["all"].append(m)
            if int(race_df["race_no"].iloc[0]) >= 8:
                day["r8"].append(m)
                overall["r8"].append(m)

        for scope, label in (("all", "全レース"), ("r8", "8R以降")):
            ms = day[scope]
            if not ms:
                continue
            t3 = np.mean([x[0] for x in ms]); t5 = np.mean([x[1] for x in ms]); t1 = np.mean([x[2] for x in ms])
            print(f"\n{date_str} [{label}] {len(ms)}R: "
                  f"top3_hit={t3:.1%}  top5_coverage={t5:.1%}  top1_acc={t1:.1%}")

    print("\n" + "=" * 50)
    for scope, label in (("all", "全レース"), ("r8", "8R以降")):
        ms = overall[scope]
        if not ms:
            continue
        t3 = np.mean([x[0] for x in ms]); t5 = np.mean([x[1] for x in ms]); t1 = np.mean([x[2] for x in ms])
        print(f"合計 {dates} [{label}] {len(ms)}R: "
              f"top3_hit={t3:.1%}  top5_coverage={t5:.1%}  top1_acc={t1:.1%}")


def main():
    parser = argparse.ArgumentParser(description="特定日の予測精度を held-out 評価")
    parser.add_argument("--dates", nargs="+", required=True, help="評価対象日 YYYY/MM/DD（複数可）")
    parser.add_argument("--params", default=None, help="使用するパラメータJSON（省略時は本番v6）")
    parser.add_argument("--no-oi", action="store_true")
    parser.add_argument("--no-nankan", action="store_true")
    parser.add_argument("--blend", action="store_true",
                        help="市場人気(popularity)とのブレンド順位で評価")
    args = parser.parse_args()

    df_hist = kp.load_history(use_oi_supplement=not args.no_oi,
                              use_nankan_supplement=not args.no_nankan)
    eval_dates(df_hist, args.dates, params_path=args.params, blend=args.blend)


if __name__ == "__main__":
    main()
