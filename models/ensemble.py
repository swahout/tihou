"""4モデルアンサンブル"""
import numpy as np
import pandas as pd
from itertools import groupby as _groupby
from typing import Optional

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    race_ids: pd.Series,
    params: Optional[dict] = None,
) -> dict:
    """
    X_train  : 特徴量 DataFrame
    y_train  : 着順 Series（整数）
    race_ids : レースID Series（XGBRankerのグループ用）
    """
    p = params or {}
    y_bin = (y_train == 1).astype(int)  # 1着フラグ
    X = X_train.values

    # ── LightGBM ──
    lgb_p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": p.get("lgb_n_estimators", 300),
        "learning_rate": p.get("lgb_lr", 0.05),
        "num_leaves": p.get("lgb_num_leaves", 31),
        "min_child_samples": p.get("lgb_min_child", 20),
        "subsample": p.get("lgb_subsample", 0.8),
        "colsample_bytree": p.get("lgb_colsample", 0.8),
        "reg_alpha": p.get("lgb_alpha", 0.0),
        "reg_lambda": p.get("lgb_lambda", 1.0),
    }
    lgb_model = lgb.LGBMClassifier(**lgb_p)
    lgb_model.fit(X, y_bin.values)

    # ── XGBoost rank:pairwise ──
    # データをrace_id順にソートしてグループサイズを計算
    sort_idx = np.argsort(race_ids.values, kind="stable")
    X_sorted = X[sort_idx]
    y_sorted = y_train.values[sort_idx]
    ids_sorted = race_ids.values[sort_idx]
    group_sizes = [sum(1 for _ in g) for _, g in _groupby(ids_sorted)]

    xgb_p = {
        "objective": "rank:pairwise",
        "n_estimators": p.get("xgb_n_estimators", 300),
        "learning_rate": p.get("xgb_lr", 0.05),
        "max_depth": p.get("xgb_max_depth", 6),
        "subsample": p.get("xgb_subsample", 0.8),
        "colsample_bytree": p.get("xgb_colsample", 0.8),
        "reg_alpha": p.get("xgb_alpha", 0.0),
        "reg_lambda": p.get("xgb_lambda", 1.0),
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": 42,
    }
    xgb_model = xgb.XGBRanker(**xgb_p)
    xgb_model.fit(X_sorted, -y_sorted, group=group_sizes)

    # ── RandomForest ──
    rf_p = {
        "n_estimators": p.get("rf_n_estimators", 200),
        "max_depth": p.get("rf_max_depth", None),
        "min_samples_leaf": p.get("rf_min_leaf", 5),
        "max_features": p.get("rf_max_features", "sqrt"),
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_model = RandomForestClassifier(**rf_p)
    rf_model.fit(X, y_bin.values)

    # ── LogisticRegression + キャリブレーション ──
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr_base = LogisticRegression(C=p.get("lr_C", 1.0), max_iter=1000, random_state=42)
    lr_model = CalibratedClassifierCV(lr_base, cv=3, method="isotonic")
    lr_model.fit(X_sc, y_bin.values)

    return {
        "lgb": lgb_model,
        "xgb": xgb_model,
        "rf": rf_model,
        "lr": lr_model,
        "scaler": scaler,
        "feature_cols": list(X_train.columns),
    }


def predict_ensemble(
    models: dict,
    X_pred: pd.DataFrame,
    weights: Optional[dict] = None,
) -> np.ndarray:
    """アンサンブル予測スコア（大きいほど上位予測）を返す"""
    w = weights or {"lgb": 0.40, "xgb": 0.25, "rf": 0.20, "lr": 0.15}
    X = X_pred[models["feature_cols"]].values

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_lgb = models["lgb"].predict_proba(X)[:, 1]
        p_xgb_raw = models["xgb"].predict(X)
        rng = p_xgb_raw.max() - p_xgb_raw.min()
        p_xgb = (p_xgb_raw - p_xgb_raw.min()) / (rng + 1e-9)
        p_rf = models["rf"].predict_proba(X)[:, 1]
        X_sc = models["scaler"].transform(X)
        p_lr = models["lr"].predict_proba(X_sc)[:, 1]

    return (
        w["lgb"] * p_lgb
        + w["xgb"] * p_xgb
        + w["rf"] * p_rf
        + w["lr"] * p_lr
    )
