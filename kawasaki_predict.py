#!/usr/bin/env python3
"""川崎競馬 予測スクリプト（tihouモデル不使用のクリーン実装）

Usage:
    python kawasaki_predict.py --date 2026/06/15         # 予測
    python kawasaki_predict.py --backtest                # バックテスト

特徴量:
    - 当場（川崎）成績率（Bayesian平滑化）
    - 通算成績率
    - 直近5走フォーム
    - 騎手実績
    - テン乗りフラグ（初コンビ）
    - 物理情報（年齢、負担重量、馬番、距離）
    - データ信頼度

モデル: LightGBM単体（rank:ndcg）
根拠出力: SHAPベースの主要因表示
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate
import lightgbm as lgb

# ── パス ──────────────────────────────────────────────────
KAWASAKI_DIR = Path("data/historical_kawasaki")
OI_DIR = Path("data/historical_oi")
RACES_DIR = Path("data/races")
VENUE = "川崎"

# ── Bayesian平滑化パラメータ ──────────────────────────────
K_HORSE = 8
K_JOCKEY = 30
DATA_K = 8

# ── 特徴量列 ──────────────────────────────────────────────
FEATURE_COLS = [
    "top3_rate_venue",      # 当場3着内率 (Bayes)
    "win_rate_venue",       # 当場勝率 (Bayes)
    "top3_rate_total",      # 通算3着内率 (Bayes)
    "win_rate_total",       # 通算勝率 (Bayes)
    "n_venue",              # 当場出走数
    "n_total",              # 通算出走数
    "recent_avg_pos",       # 直近5走平均着順
    "recent_top3",          # 直近5走3着内率
    "recent_venue_avg_pos", # 当場直近5走平均着順
    "jockey_top3_rate",     # 騎手の当場3着内率
    "jockey_win_rate",      # 騎手の当場勝率
    "is_ten_nori",          # テン乗りフラグ（この騎手×馬が初コンビ=1）
    "age",
    "weight_carried",
    "sex_enc",
    "field_size",
    "distance",
    "days_since_last",
    "umaban",
    "data_reliability",
]

# SHAP表示用ラベル（コンパクトに）
FEATURE_LABEL = {
    "top3_rate_venue":      "川崎3着内率",
    "win_rate_venue":       "川崎勝率",
    "top3_rate_total":      "通算3着内率",
    "win_rate_total":       "通算勝率",
    "n_venue":              "川崎出走数",
    "n_total":              "通算出走数",
    "recent_avg_pos":       "直近着順",
    "recent_top3":          "直近3着内率",
    "recent_venue_avg_pos": "川崎直近着順",
    "jockey_top3_rate":     "騎手実績",
    "jockey_win_rate":      "騎手勝率",
    "is_ten_nori":          "テン乗り",
    "age":                  "年齢",
    "weight_carried":       "斤量",
    "sex_enc":              "性別",
    "field_size":           "頭数",
    "distance":             "距離",
    "days_since_last":      "休養日数",
    "umaban":               "馬番",
    "data_reliability":     "データ量",
}

SEX_MAP = {"牡": 0, "牝": 1, "セン": 2, "": 0}


# ── データ読み込み ─────────────────────────────────────────

def load_history(use_oi_supplement: bool = True) -> pd.DataFrame:
    dfs = []
    if KAWASAKI_DIR.exists():
        for f in sorted(KAWASAKI_DIR.glob("kawasaki_*.csv")):
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
            except Exception as e:
                print(f"  警告: {f} ({e})")
    if use_oi_supplement and OI_DIR.exists():
        for f in sorted(OI_DIR.glob("oi_*.csv")):
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
            except Exception as e:
                print(f"  警告: {f} ({e})")
    if not dfs:
        raise FileNotFoundError(
            "過去データがありません。"
            " まず: python collect_historical_kawasaki.py --years 2022 2023 2024 2025"
        )
    df = pd.concat(dfs, ignore_index=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[df["finish_position"].notna() & (df["finish_position"] >= 1)].copy()
    df["finish_position"] = df["finish_position"].astype(int)
    venues = df["venue"].unique().tolist()
    print(f"履歴データ: {len(df):,}行 "
          f"({df['race_date'].dt.year.min()}〜{df['race_date'].dt.year.max()}) "
          f"馬場: {venues}")
    return df


# ── 統計計算 ───────────────────────────────────────────────

def _bayes(num: float, den: float, prior: float, k: float) -> float:
    return (num + prior * k) / (den + k)


def compute_stats(df_hist: pd.DataFrame) -> dict:
    grp_total = df_hist.groupby("horse_name")
    total_n   = grp_total.size()
    total_top3 = grp_total.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    total_win  = grp_total.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)

    global_top3_prior = (df_hist["finish_position"] <= 3).mean()
    global_win_prior  = (df_hist["finish_position"] == 1).mean()

    df_venue = df_hist[df_hist["venue"] == VENUE]
    grp_venue  = df_venue.groupby("horse_name")
    venue_n    = grp_venue.size()
    venue_top3 = grp_venue.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    venue_win  = grp_venue.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)

    df_sorted = df_hist.sort_values(["horse_name", "race_date"])
    recent = df_sorted.groupby("horse_name").tail(5)
    recent_avg      = recent.groupby("horse_name")["finish_position"].mean()
    recent_top3_rate = recent.groupby("horse_name").apply(
        lambda g: (g["finish_position"] <= 3).mean(), include_groups=False)

    df_venue_sorted  = df_venue.sort_values(["horse_name", "race_date"])
    recent_venue     = df_venue_sorted.groupby("horse_name").tail(5)
    recent_venue_avg = recent_venue.groupby("horse_name")["finish_position"].mean()

    last_date = df_sorted.groupby("horse_name")["race_date"].max()

    grp_j     = df_venue.groupby("jockey")
    j_n       = grp_j.size()
    j_top3    = grp_j.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    j_win     = grp_j.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)
    j_top3_prior = (df_venue["finish_position"] <= 3).mean() if len(df_venue) > 0 else global_top3_prior
    j_win_prior  = (df_venue["finish_position"] == 1).mean() if len(df_venue) > 0 else global_win_prior

    # テン乗り判定用: 過去に組んだことある（馬名, 騎手）ペアを記録
    horse_jockey_pairs = set(zip(df_hist["horse_name"], df_hist["jockey"]))

    all_horses = set(total_n.index) | set(venue_n.index)
    horse_stats = {}
    for name in all_horses:
        nt = total_n.get(name, 0)
        nv = venue_n.get(name, 0)
        horse_stats[name] = {
            "n_total": nt,
            "n_venue": nv,
            "top3_rate_total":      _bayes(total_top3.get(name, 0), nt, global_top3_prior, K_HORSE),
            "win_rate_total":       _bayes(total_win.get(name, 0),  nt, global_win_prior,  K_HORSE),
            "top3_rate_venue":      _bayes(venue_top3.get(name, 0), nv, j_top3_prior, K_HORSE),
            "win_rate_venue":       _bayes(venue_win.get(name, 0),  nv, j_win_prior,  K_HORSE),
            "recent_avg_pos":       recent_avg.get(name, np.nan),
            "recent_top3":          recent_top3_rate.get(name, np.nan),
            "recent_venue_avg_pos": recent_venue_avg.get(name, np.nan),
            "last_race_date":       last_date.get(name, pd.NaT),
        }

    jockey_stats = {}
    for j in j_n.index:
        jn = j_n.get(j, 0)
        jockey_stats[j] = {
            "jockey_top3_rate": _bayes(j_top3.get(j, 0), jn, j_top3_prior, K_JOCKEY),
            "jockey_win_rate":  _bayes(j_win.get(j, 0),  jn, j_win_prior,  K_JOCKEY),
        }

    return {
        "horse":             horse_stats,
        "jockey":            jockey_stats,
        "horse_jockey_pairs": horse_jockey_pairs,
        "priors": {
            "top3":   global_top3_prior,
            "win":    global_win_prior,
            "j_top3": j_top3_prior,
            "j_win":  j_win_prior,
        },
    }


# ── 特徴量構築 ─────────────────────────────────────────────

def build_features(df_race: pd.DataFrame, stats: dict, pred_date: pd.Timestamp) -> pd.DataFrame:
    priors = stats["priors"]
    pairs  = stats["horse_jockey_pairs"]
    rows = []
    for _, h in df_race.iterrows():
        name   = h["horse_name"]
        jockey = h.get("jockey", "")
        hs = stats["horse"].get(name, {})
        js = stats["jockey"].get(jockey, {})

        last_dt = hs.get("last_race_date", pd.NaT)
        days = (pred_date - last_dt).days if pd.notna(last_dt) else 999

        n_total = hs.get("n_total", 0)
        fs = h.get("field_size", 8) or 8

        # テン乗り: この騎手×馬の組み合わせが過去に存在しなければ1
        is_ten_nori = 0 if (name, jockey) in pairs else 1

        row = {
            "horse_name":           name,
            "top3_rate_venue":      hs.get("top3_rate_venue",      _bayes(0, 0, priors["j_top3"], K_HORSE)),
            "win_rate_venue":       hs.get("win_rate_venue",        _bayes(0, 0, priors["j_win"],  K_HORSE)),
            "top3_rate_total":      hs.get("top3_rate_total",       _bayes(0, 0, priors["top3"],   K_HORSE)),
            "win_rate_total":       hs.get("win_rate_total",        _bayes(0, 0, priors["win"],    K_HORSE)),
            "n_venue":              hs.get("n_venue", 0),
            "n_total":              n_total,
            "recent_avg_pos":       hs.get("recent_avg_pos",        fs * 0.6),
            "recent_top3":          hs.get("recent_top3",           priors["top3"]),
            "recent_venue_avg_pos": hs.get("recent_venue_avg_pos",  fs * 0.6),
            "jockey_top3_rate":     js.get("jockey_top3_rate",      _bayes(0, 0, priors["j_top3"], K_JOCKEY)),
            "jockey_win_rate":      js.get("jockey_win_rate",       _bayes(0, 0, priors["j_win"],  K_JOCKEY)),
            "is_ten_nori":          is_ten_nori,
            "age":                  h.get("age", 0) or 0,
            "weight_carried":       h.get("weight_carried", 55.0) or 55.0,
            "sex_enc":              SEX_MAP.get(str(h.get("sex", "")), 0),
            "field_size":           fs,
            "distance":             h.get("distance", 1400),
            "days_since_last":      min(days, 999),
            "umaban":               h.get("horse_no", 0),
            "data_reliability":     n_total / (n_total + DATA_K),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── 学習データ構築 ─────────────────────────────────────────

def build_train_data(df_hist: pd.DataFrame) -> tuple:
    df = df_hist.copy()
    df["year_month"] = df["race_date"].dt.to_period("M")
    periods = sorted(df["year_month"].unique())

    all_X, all_y, all_groups = [], [], []
    for i, period in enumerate(periods):
        if i < 2:
            continue
        df_test   = df[df["year_month"] == period]
        df_before = df[df["year_month"] < period]
        if df_before.empty:
            continue
        stats     = compute_stats(df_before)
        pred_date = pd.Timestamp(str(period.start_time))

        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            Xr = build_features(race_df.reset_index(drop=True), stats, pred_date)
            yr = race_df["finish_position"].values
            all_X.append(Xr)
            all_y.append(yr)
            all_groups.extend([race_id] * len(race_df))

    if not all_X:
        return pd.DataFrame(), np.array([]), np.array([])

    X      = pd.concat(all_X, ignore_index=True)
    y      = np.concatenate(all_y)
    groups = np.array(all_groups)
    return X, y, groups


# ── LightGBM学習 ──────────────────────────────────────────

def train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> lgb.Booster:
    X_feat = X[FEATURE_COLS].fillna(X[FEATURE_COLS].median())

    sort_idx      = np.argsort(groups)
    X_sorted      = X_feat.iloc[sort_idx]
    y_sorted      = y[sort_idx]
    groups_sorted = groups[sort_idx]
    _, gsizes     = np.unique(groups_sorted, return_counts=True)

    label = np.zeros_like(y_sorted, dtype=float)
    for g_id in np.unique(groups_sorted):
        mask = groups_sorted == g_id
        pos  = y_sorted[mask]
        n    = pos.max()
        label[mask] = np.maximum(0, n - pos + 1)

    dataset = lgb.Dataset(X_sorted, label=label, group=gsizes)
    params = {
        "objective":     "rank_xendcg",
        "metric":        "ndcg",
        "ndcg_eval_at":  [3, 5],
        "learning_rate": 0.05,
        "num_leaves":    31,
        "min_child_samples": 10,
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":     0.1,
        "reg_lambda":    0.1,
        "verbose":       -1,
        "n_jobs":        -1,
    }
    return lgb.train(params, dataset, num_boost_round=300,
                     callbacks=[lgb.log_evaluation(period=100)])


# ── 根拠生成（SHAPベース） ────────────────────────────────

def make_reason(shap_row: np.ndarray, feature_names: list[str],
                horse_row: pd.Series) -> str:
    """
    SHAP値の上位要因を日本語で返す。
    正のSHAP → ↑（有利に働いた特徴）
    負のSHAP → ↓（不利に働いた特徴）
    テン乗りは is_ten_nori == 1 で強調表示。
    """
    n = len(feature_names)
    shap_vals = shap_row[:n]  # バイアス項を除く

    # テン乗りは特別扱い（SHAPが負なら不利サインとして明示）
    is_ten = horse_row.get("is_ten_nori", 0) == 1
    ten_label = "【テン乗り】" if is_ten else ""

    # SHAP絶対値で上位を選び、↑↓を付ける（テン乗りは除外してラベルに出す）
    feat_shap = [(f, v) for f, v in zip(feature_names, shap_vals)
                 if f != "is_ten_nori"]
    # 正: 上位2件, 負: 上位1件
    pos = sorted([(f, v) for f, v in feat_shap if v > 0], key=lambda x: -x[1])[:2]
    neg = sorted([(f, v) for f, v in feat_shap if v < 0], key=lambda x:  x[1])[:1]

    parts = [f"↑{FEATURE_LABEL.get(f, f)}" for f, _ in pos]
    parts += [f"↓{FEATURE_LABEL.get(f, f)}" for f, _ in neg]
    return ten_label + " ".join(parts)


# ── 予測 ──────────────────────────────────────────────────

def predict_race(model: lgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    X_feat = X[FEATURE_COLS].fillna(X[FEATURE_COLS].median())
    scores = model.predict(X_feat)
    try:
        contrib = model.predict(X_feat, pred_contrib=True)  # shape: (n, n_feat+1)
        shap_vals = contrib[:, :-1]
    except Exception:
        shap_vals = None
    return scores, shap_vals


def do_predict(df_hist: pd.DataFrame, date_str: str) -> None:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    shutuba_path = RACES_DIR / f"kawasaki_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    if not shutuba_path.exists():
        print(f"出走表なし: {shutuba_path}")
        print(f"まず: python collect_kawasaki_shutuba.py --date {date_str}")
        return

    df_shutuba = pd.read_csv(shutuba_path, encoding="utf-8-sig")
    print(f"出走表: {shutuba_path} ({len(df_shutuba)}頭, {df_shutuba['race_no'].nunique()}R)")

    pred_date    = pd.Timestamp(dt)
    df_train_hist = df_hist[df_hist["race_date"] < pred_date].copy()
    if df_train_hist.empty:
        df_train_hist = df_hist.copy()

    print("統計計算中...")
    stats = compute_stats(df_train_hist)

    print("学習データ構築中...")
    X, y, groups = build_train_data(df_train_hist)
    if X.empty:
        print("学習データ不足 → 統計ベースのみで予測")
        _predict_stats_only(df_shutuba, stats, pred_date)
        return

    print(f"学習データ: {len(X):,}行, {len(np.unique(groups)):,}レース")
    print("モデル学習中...")
    model = train_model(X, y, groups)

    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        if len(race_df) < 2:
            continue
        race_df = race_df.copy().reset_index(drop=True)
        Xr      = build_features(race_df, stats, pred_date)
        scores, shap_vals = predict_race(model, Xr)

        # softmax正規化: スコアが負でも確率が正になる・合計=3（期待3着内馬数）
        exp_s     = np.exp(scores - scores.max())
        raw_prob  = exp_s / exp_s.sum() * 3.0
        base_rate = 3.0 / max(len(race_df), 1)
        rel       = Xr["data_reliability"].values
        top3_prob = raw_prob * rel + base_rate * (1 - rel)

        # 根拠文字列
        reasons = []
        for i, horse_row in Xr.iterrows():
            if shap_vals is not None:
                reasons.append(make_reason(shap_vals[i], FEATURE_COLS, horse_row))
            else:
                reasons.append("")

        race_df["score"]            = scores
        race_df["top3_prob"]        = top3_prob
        race_df["data_reliability"] = rel
        race_df["n_hist"]           = Xr["n_total"].values
        race_df["n_venue"]          = Xr["n_venue"].values
        race_df["is_ten_nori"]      = Xr["is_ten_nori"].values
        race_df["reason"]           = reasons
        out_rows.append(race_df)

    if not out_rows:
        print("予測失敗")
        return

    df_pred = pd.concat(out_rows, ignore_index=True)
    out_path = RACES_DIR / f"kawasaki_{dt.strftime('%Y_%m%d')}_prediction.csv"
    df_pred.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"\n予測保存: {out_path}")
    _print_predictions(df_pred, date_str)


def _predict_stats_only(df_shutuba: pd.DataFrame, stats: dict, pred_date: pd.Timestamp) -> None:
    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        race_df = race_df.copy().reset_index(drop=True)
        Xr = build_features(race_df, stats, pred_date)
        score = (
            Xr["top3_rate_venue"]  * 2.0
            + Xr["win_rate_venue"] * 1.5
            + Xr["top3_rate_total"] * 1.0
            + Xr["jockey_top3_rate"] * 0.5
        )
        exp_s     = np.exp(score - score.max())
        raw_prob  = exp_s / exp_s.sum() * 3.0
        base_rate = 3.0 / max(len(race_df), 1)
        rel       = Xr["data_reliability"].values
        race_df["score"]            = score.values
        race_df["top3_prob"]        = raw_prob * rel + base_rate * (1 - rel)
        race_df["data_reliability"] = rel
        race_df["n_hist"]           = Xr["n_total"].values
        race_df["n_venue"]          = Xr["n_venue"].values
        race_df["is_ten_nori"]      = Xr["is_ten_nori"].values
        race_df["reason"]           = [
            ("【テン乗り】" if r == 1 else "") + "統計ベース予測"
            for r in Xr["is_ten_nori"].values
        ]
        out_rows.append(race_df)
    df_pred = pd.concat(out_rows, ignore_index=True)
    _print_predictions(df_pred, pred_date.strftime("%Y/%m/%d"))


def _print_predictions(df_pred: pd.DataFrame, date_str: str) -> None:
    print(f"\n{'='*80}")
    print(f"  川崎競馬 {date_str} 予測")
    print(f"{'='*80}")
    for race_no, grp in df_pred.groupby("race_no"):
        top  = grp.sort_values("top3_prob", ascending=False)
        meta = top.iloc[0]
        print(f"\n【{race_no:2d}R】{meta.get('race_name','')}  "
              f"{int(meta.get('distance', 0))}m")
        rows = []
        for rank, (_, h) in enumerate(top.iterrows(), 1):
            rel     = h.get("data_reliability", 0)
            n_v     = int(h.get("n_venue", 0))
            n_t     = int(h.get("n_hist", 0))
            no_data = " ⚠" if rel < 0.3 else ""
            reason  = h.get("reason", "")
            rows.append([
                rank,
                int(h.get("horse_no", 0)),
                h.get("horse_name", ""),
                f"{h['top3_prob']:.1%}",
                f"{rel:.2f}{no_data}",
                f"{n_v}/{n_t}",
                h.get("jockey", ""),
                reason,
            ])
        print(tabulate(rows,
                       headers=["順", "馬番", "馬名", "3着内確率", "信頼度", "川崎/通算", "騎手", "根拠"],
                       tablefmt="simple"))


# ── バックテスト ──────────────────────────────────────────

def do_backtest(df_hist: pd.DataFrame) -> None:
    df_kawasaki = df_hist[df_hist["venue"] == VENUE].copy()
    years = sorted(df_kawasaki["race_date"].dt.year.unique())
    if len(years) < 2:
        print("バックテストには最低2年分の川崎データが必要です")
        return

    print(f"\n=== LeaveOneYearOut バックテスト ({VENUE}) ===")
    results = []
    for test_year in years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test  = df_kawasaki[df_kawasaki["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue

        stats = compute_stats(df_train)
        X_tr, y_tr, groups_tr = build_train_data(df_train)
        if X_tr.empty:
            print(f"  {test_year}: 学習データ不足 → スキップ")
            continue

        model = train_model(X_tr, y_tr, groups_tr)
        top5_total = top3_total = top1_total = race_count = 0

        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            race_pred_date = race_df["race_date"].iloc[0]
            Xr = build_features(race_df.reset_index(drop=True), stats, race_pred_date)
            scores, _ = predict_race(model, Xr)

            actual_top3 = set(race_df.index[race_df["finish_position"] <= 3])
            pred_sorted = race_df.index[np.argsort(-scores)]

            top5_total += len(actual_top3 & set(pred_sorted[:5])) / min(3, len(actual_top3))
            top3_total += len(actual_top3 & set(pred_sorted[:3])) / min(3, len(actual_top3))
            top1_total += int(pred_sorted[0] in actual_top3)
            race_count += 1

        if race_count == 0:
            continue
        r = {
            "year": test_year, "races": race_count,
            "top5_coverage": top5_total / race_count,
            "top3_hit":      top3_total / race_count,
            "top1_acc":      top1_total / race_count,
        }
        results.append(r)
        print(f"  {test_year}: {race_count}R "
              f"top5={r['top5_coverage']:.1%} "
              f"top3={r['top3_hit']:.1%} "
              f"top1={r['top1_acc']:.1%}")

    if results:
        print(f"\n平均 top5_coverage: {np.mean([r['top5_coverage'] for r in results]):.1%}")
        print(f"平均 top3_hit     : {np.mean([r['top3_hit']      for r in results]):.1%}")
        print(f"平均 top1_acc     : {np.mean([r['top1_acc']      for r in results]):.1%}")


# ── main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="川崎競馬予測 (LightGBM rank + SHAP根拠)")
    parser.add_argument("--date",      help="予測日 YYYY/MM/DD")
    parser.add_argument("--backtest",  action="store_true")
    parser.add_argument("--no-oi",     action="store_true", help="大井データを補助に使わない")
    args = parser.parse_args()

    df_hist = load_history(use_oi_supplement=not args.no_oi)

    if args.backtest:
        do_backtest(df_hist)
        return

    if args.date:
        date_str = args.date
    else:
        from datetime import date, timedelta
        date_str = (date.today() + timedelta(days=1)).strftime("%Y/%m/%d")

    do_predict(df_hist, date_str)


if __name__ == "__main__":
    main()
