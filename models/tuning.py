"""Optuna チューニング共通ロジック"""
import json
from pathlib import Path
from typing import Callable, Optional

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(
    objective_fn: Callable[[optuna.Trial], float],
    study_name: str,
    db_path: Path,
    n_trials: int = 200,
    direction: str = "maximize",
) -> optuna.Study:
    """Optuna スタディを作成 or 再開してチューニングを実行する"""
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    return study


def suggest_params(trial: optuna.Trial) -> dict:
    """モデルハイパーパラメータをサジェスト"""
    return {
        # LightGBM
        "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 100, 600),
        "lgb_lr": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
        "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 15, 63),
        "lgb_min_child": trial.suggest_int("lgb_min_child", 5, 50),
        "lgb_subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
        "lgb_colsample": trial.suggest_float("lgb_colsample", 0.5, 1.0),
        "lgb_alpha": trial.suggest_float("lgb_alpha", 0.0, 2.0),
        "lgb_lambda": trial.suggest_float("lgb_lambda", 0.0, 2.0),
        # XGBoost
        "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 600),
        "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
        "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "xgb_colsample": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        # RandomForest
        "rf_n_estimators": trial.suggest_int("rf_n_estimators", 100, 400),
        "rf_max_depth": trial.suggest_categorical("rf_max_depth", [None, 5, 10, 20]),
        "rf_min_leaf": trial.suggest_int("rf_min_leaf", 2, 20),
        # LR
        "lr_C": trial.suggest_float("lr_C", 0.01, 10.0, log=True),
        # アンサンブル重み
        "w_lgb": trial.suggest_float("w_lgb", 0.2, 0.6),
        "w_xgb": trial.suggest_float("w_xgb", 0.1, 0.4),
        "w_rf": trial.suggest_float("w_rf", 0.1, 0.4),
        # w_lr = 1 - w_lgb - w_xgb - w_rf
        # ベイズ平滑化強度
        "bayesian_k": trial.suggest_int("bayesian_k", 3, 20),
        "bayesian_k_jockey": trial.suggest_int("bayesian_k_jockey", 10, 60),
    }


def params_to_weights(params: dict) -> dict:
    total = params["w_lgb"] + params["w_xgb"] + params["w_rf"]
    w_lr = max(1.0 - total, 0.05)
    norm = total + w_lr
    return {
        "lgb": params["w_lgb"] / norm,
        "xgb": params["w_xgb"] / norm,
        "rf": params["w_rf"] / norm,
        "lr": w_lr / norm,
    }


def save_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
