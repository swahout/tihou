"""LeaveOneYearOut バックテスト"""
import numpy as np
import pandas as pd
from typing import Optional

from models.features import build_features, FEATURE_COLS
from models.ensemble import train_ensemble, predict_ensemble


def _build_race_id(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
        + "_" + df["venue"].fillna("")
        + "_" + df["race_no"].astype(str)
    ).reset_index(drop=True)


def _build_train_features(
    df_train: pd.DataFrame,
    params: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """訓練データ全体から特徴量・ターゲット・レースIDを構築"""
    race_id_col = _build_race_id(df_train)
    df_train = df_train.copy()
    df_train["race_id"] = race_id_col

    Xs, ys, rids = [], [], []
    for rid, grp in df_train.groupby("race_id"):
        if len(grp) < 4:
            continue
        try:
            race_date = grp["race_date"].iloc[0]
            df_hist = df_train[df_train["race_date"] < race_date]
            if df_hist.empty:
                continue
            X_r = build_features(grp, df_hist, params)
            Xs.append(X_r)
            ys.extend(grp["finish_position"].astype(int).tolist())
            rids.extend([rid] * len(grp))
        except Exception:
            continue

    if not Xs:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)
    return pd.concat(Xs, ignore_index=True), pd.Series(ys), pd.Series(rids)


def run_backtest(
    df_all: pd.DataFrame,
    params: Optional[dict] = None,
    weights: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    df_all: 全歴史データ（race_date, venue, race_no, horse_name, finish_position, ...）
    """
    df = df_all.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["year"] = df["race_date"].dt.year
    df = df[df["finish_position"].notna()].copy()
    df["finish_position"] = df["finish_position"].astype(int)

    years = sorted(df["year"].unique())
    if len(years) < 2:
        return {"error": "LeaveOneYearOut には最低2年分のデータが必要です"}

    all_metrics = []

    for test_year in years[1:]:
        df_train = df[df["year"] < test_year].copy()
        df_test = df[df["year"] == test_year].copy()

        if len(df_train) < 200 or len(df_test) < 20:
            continue

        if verbose:
            print(f"  [{test_year}] 学習データ {len(df_train)}行 / テストデータ {len(df_test)}行")

        X_train, y_train, rids_train = _build_train_features(df_train, params)
        if X_train.empty:
            continue

        try:
            models = train_ensemble(X_train, y_train, rids_train, params)
        except Exception as e:
            if verbose:
                print(f"    学習エラー: {e}")
            continue

        df_test["race_id"] = _build_race_id(df_test)
        top5_hits, top5_poss = 0, 0
        top3_hits = 0
        top1_hits = 0
        n_races = 0

        for _, race_df in df_test.groupby("race_id"):
            if len(race_df) < 4:
                continue
            try:
                X_test = build_features(race_df, df_train, params)
                scores = predict_ensemble(models, X_test, weights)
            except Exception:
                continue

            race_df = race_df.copy()
            race_df["score"] = scores
            srt = race_df.sort_values("score", ascending=False)
            pred5 = set(srt.head(5)["horse_name"])
            pred3 = set(srt.head(3)["horse_name"])
            act3 = set(race_df[race_df["finish_position"] <= 3]["horse_name"])
            act1 = set(race_df[race_df["finish_position"] == 1]["horse_name"])

            top5_hits += len(pred5 & act3)
            top5_poss += len(act3)
            top3_hits += len(pred3 & act3)
            top1_hits += len(pred3 & act1)
            n_races += 1

        if n_races == 0:
            continue

        m = {
            "test_year": test_year,
            "n_races": n_races,
            "top5_coverage": top5_hits / top5_poss if top5_poss else 0.0,
            "top3_hit": top3_hits / (n_races * 3),
            "top1_acc": top1_hits / n_races,
        }
        all_metrics.append(m)
        if verbose:
            print(
                f"    top5_coverage={m['top5_coverage']:.1%}  "
                f"top3_hit={m['top3_hit']:.1%}  "
                f"top1_acc={m['top1_acc']:.1%}  ({n_races}R)"
            )

    if not all_metrics:
        return {"error": "評価できるデータがありません"}

    return {
        "top5_coverage": float(np.mean([m["top5_coverage"] for m in all_metrics])),
        "top3_hit": float(np.mean([m["top3_hit"] for m in all_metrics])),
        "top1_acc": float(np.mean([m["top1_acc"] for m in all_metrics])),
        "year_results": all_metrics,
    }
