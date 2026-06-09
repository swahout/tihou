"""LeaveOneYearOut バックテスト"""
import numpy as np
import pandas as pd
from typing import Optional

from models.features import build_features, precompute_stats, build_features_precomputed, FEATURE_COLS
from models.ensemble import train_ensemble, predict_ensemble


def _build_race_id(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
        + "_" + df["venue"].fillna("")
        + "_" + df["race_no"].astype(str)
    )


def _build_train_features(
    df_train: pd.DataFrame,
    params: Optional[dict] = None,
    stats: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """訓練データ全体から特徴量・ターゲット・レースIDを構築（集計を1回だけ計算）"""
    race_id_col = _build_race_id(df_train)
    df_train = df_train.copy()
    df_train["race_id"] = race_id_col

    # 集計を一度だけ計算（呼び出し元で計算済みの場合は再利用）
    if stats is None:
        stats = precompute_stats(df_train, params)

    Xs, ys, rids = [], [], []
    for rid, grp in df_train.groupby("race_id"):
        if len(grp) < 4:
            continue
        try:
            X_r = build_features_precomputed(grp, stats, params)
            Xs.append(X_r)
            ys.extend(grp["finish_position"].astype(int).tolist())
            rids.extend([rid] * len(grp))
        except Exception:
            continue

    if not Xs:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)
    return pd.concat(Xs, ignore_index=True), pd.Series(ys), pd.Series(rids)


def _eval_split(df_train, df_test, params, weights, verbose, label):
    """共通の学習→評価ロジック"""
    if len(df_train) < 200 or len(df_test) < 20:
        return None

    # 集計を一度だけ計算（訓練・テスト両方で再利用）
    train_stats = precompute_stats(df_train, params)

    X_train, y_train, rids_train = _build_train_features(df_train, params, stats=train_stats)
    if X_train.empty:
        return None

    try:
        models = train_ensemble(X_train, y_train, rids_train, params)
    except Exception as e:
        if verbose:
            print(f"    学習エラー [{label}]: {e}")
        return None

    df_test = df_test.copy()
    df_test["race_id"] = _build_race_id(df_test)
    top5_hits, top5_poss, top3_hits, top1_hits, n_races = 0, 0, 0, 0, 0

    for _, race_df in df_test.groupby("race_id"):
        if len(race_df) < 4:
            continue
        try:
            X_test = build_features_precomputed(race_df, train_stats, params)
            scores = predict_ensemble(models, X_test, weights)
        except Exception:
            continue

        race_df = race_df.copy()
        srt = race_df.assign(score=scores).sort_values("score", ascending=False)
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
        return None

    m = {
        "label": label,
        "n_races": n_races,
        "top5_coverage": top5_hits / top5_poss if top5_poss else 0.0,
        "top3_hit": top3_hits / (n_races * 3),
        "top1_acc": top1_hits / n_races,
    }
    if verbose:
        print(
            f"    top5_coverage={m['top5_coverage']:.1%}  "
            f"top3_hit={m['top3_hit']:.1%}  "
            f"top1_acc={m['top1_acc']:.1%}  ({n_races}R)"
        )
    return m


def run_backtest(
    df_all: pd.DataFrame,
    params: Optional[dict] = None,
    weights: Optional[dict] = None,
    verbose: bool = True,
    test_days: Optional[int] = None,
) -> dict:
    """
    df_all: 全歴史データ
    test_days: 指定すると直近N日をテスト、残りを学習（Optuna用の高速モード）
               None の場合は LeaveOneYearOut
    """
    df = df_all.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_position"].notna()].copy()
    df["finish_position"] = df["finish_position"].astype(int)

    if test_days is not None:
        # 直近N日テストモード（Optuna高速チューニング用）
        cutoff = df["race_date"].max() - pd.Timedelta(days=test_days)
        df_train = df[df["race_date"] < cutoff].copy()
        df_test  = df[df["race_date"] >= cutoff].copy()
        label = f"直近{test_days}日"
        if verbose:
            print(f"  [{label}] 学習データ {len(df_train)}行 / テストデータ {len(df_test)}行")
        m = _eval_split(df_train, df_test, params, weights, verbose, label)
        if m is None:
            return {"error": "評価できるデータがありません"}
        return {
            "top5_coverage": m["top5_coverage"],
            "top3_hit": m["top3_hit"],
            "top1_acc": m["top1_acc"],
            "year_results": [m],
        }

    # LeaveOneYearOut モード
    df["year"] = df["race_date"].dt.year
    years = sorted(df["year"].unique())
    if len(years) < 2:
        return {"error": "LeaveOneYearOut には最低2年分のデータが必要です"}

    all_metrics = []
    for test_year in years[1:]:
        df_train = df[df["year"] < test_year].copy()
        df_test  = df[df["year"] == test_year].copy()
        if verbose:
            print(f"  [{test_year}] 学習データ {len(df_train)}行 / テストデータ {len(df_test)}行")
        m = _eval_split(df_train, df_test, params, weights, verbose, str(test_year))
        if m is not None:
            m["test_year"] = test_year
            all_metrics.append(m)

    if not all_metrics:
        return {"error": "評価できるデータがありません"}

    return {
        "top5_coverage": float(np.mean([m["top5_coverage"] for m in all_metrics])),
        "top3_hit": float(np.mean([m["top3_hit"] for m in all_metrics])),
        "top1_acc": float(np.mean([m["top1_acc"] for m in all_metrics])),
        "year_results": all_metrics,
    }
