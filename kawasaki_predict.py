#!/usr/bin/env python3
"""川崎競馬 予測スクリプト（tihouモデル不使用のクリーン実装）

Usage:
    python kawasaki_predict.py --date 2026/06/15         # 予測
    python kawasaki_predict.py --date 2026/06/15 --backtest  # バックテスト

特徴量:
    - 当場（川崎）成績率（Bayesian平滑化）
    - 通算成績率
    - 直近5走フォーム
    - 騎手実績
    - 物理情報（年齢、負担重量、馬番、距離）
    - データ信頼度

モデル: LightGBM単体（rank:ndcg）
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# ── パス ──────────────────────────────────────────────────
KAWASAKI_DIR = Path("data/historical_kawasaki")
OI_DIR = Path("data/historical_oi")          # 補助データ（存在すれば使う）
RACES_DIR = Path("data/races")
VENUE = "川崎"

# ── Bayesian平滑化パラメータ ──────────────────────────────
K_HORSE = 8      # 馬成績用 (n=8で prior 50%)
K_JOCKEY = 30    # 騎手用 (n=30で prior 50%)
DATA_K = 8       # 信頼度計算用

# ── 特徴量列 ──────────────────────────────────────────────
FEATURE_COLS = [
    "top3_rate_venue",     # 当場3着内率 (Bayes)
    "win_rate_venue",      # 当場勝率 (Bayes)
    "top3_rate_total",     # 通算3着内率 (Bayes)
    "win_rate_total",      # 通算勝率 (Bayes)
    "n_venue",             # 当場出走数
    "n_total",             # 通算出走数
    "recent_avg_pos",      # 直近5走平均着順
    "recent_top3",         # 直近5走3着内率
    "recent_venue_avg_pos",# 当場直近5走平均着順
    "jockey_top3_rate",    # 騎手の当場3着内率
    "jockey_win_rate",     # 騎手の当場勝率
    "age",
    "weight_carried",
    "sex_enc",             # 牡=0 牝=1 セン=2
    "field_size",
    "distance",
    "days_since_last",     # 休養日数
    "umaban",              # 馬番（枠順効果）
    "data_reliability",    # n / (n + K)
]

SEX_MAP = {"牡": 0, "牝": 1, "セン": 2, "": 0}


# ── データ読み込み ─────────────────────────────────────────

def load_history(use_oi_supplement: bool = True) -> pd.DataFrame:
    """川崎の過去データ（＋大井補助データ）を読み込む"""
    dfs = []

    # 川崎データ（メイン）
    if KAWASAKI_DIR.exists():
        for f in sorted(KAWASAKI_DIR.glob("kawasaki_*.csv")):
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
            except Exception as e:
                print(f"  警告: {f} ({e})")

    # 大井データ（補助: 馬・騎手の統計に使うが学習データとしても使う）
    if use_oi_supplement and OI_DIR.exists():
        for f in sorted(OI_DIR.glob("oi_*.csv")):
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
            except Exception as e:
                print(f"  警告: {f} ({e})")

    if not dfs:
        raise FileNotFoundError(
            "過去データが見つかりません。"
            " まず: python collect_historical_kawasaki.py --years 2022 2023 2024 2025"
        )

    df = pd.concat(dfs, ignore_index=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[df["finish_position"].notna()].copy()
    df["finish_position"] = df["finish_position"].astype(int)
    df = df[df["finish_position"] >= 1].copy()

    venues = df["venue"].unique().tolist()
    print(f"履歴データ: {len(df):,}行 "
          f"({df['race_date'].dt.year.min()}〜{df['race_date'].dt.year.max()}) "
          f"馬場: {venues}")
    return df


# ── 統計計算 ───────────────────────────────────────────────

def _bayes(num: float, den: float, prior: float, k: float) -> float:
    return (num + prior * k) / (den + k)


def compute_stats(df_hist: pd.DataFrame) -> dict:
    """
    df_hist: 学習に使う過去データ（予測対象日より前のもの）
    返り値: {
      'horse': {horse_name: stats_dict},
      'jockey': {(jockey, venue): stats_dict},
    }
    """
    # ── 馬の通算成績 ──
    grp_total = df_hist.groupby("horse_name")
    total_n = grp_total.size()
    total_top3 = grp_total.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    total_win = grp_total.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)

    global_top3_prior = (df_hist["finish_position"] <= 3).mean()
    global_win_prior = (df_hist["finish_position"] == 1).mean()

    # ── 馬の当場（川崎のみ）成績 ──
    df_venue = df_hist[df_hist["venue"] == VENUE]
    grp_venue = df_venue.groupby("horse_name")
    venue_n = grp_venue.size()
    venue_top3 = grp_venue.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    venue_win = grp_venue.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)

    # ── 直近5走フォーム ──
    df_sorted = df_hist.sort_values(["horse_name", "race_date"])
    recent = df_sorted.groupby("horse_name").tail(5)
    recent_avg = recent.groupby("horse_name")["finish_position"].mean()
    recent_top3_rate = recent.groupby("horse_name").apply(
        lambda g: (g["finish_position"] <= 3).mean(), include_groups=False)
    # 当場直近5走
    df_venue_sorted = df_venue.sort_values(["horse_name", "race_date"])
    recent_venue = df_venue_sorted.groupby("horse_name").tail(5)
    recent_venue_avg = recent_venue.groupby("horse_name")["finish_position"].mean()

    # ── 直前レース日 ──
    last_date = df_sorted.groupby("horse_name")["race_date"].max()

    # ── 騎手の当場成績 ──
    grp_j = df_venue.groupby("jockey")
    j_n = grp_j.size()
    j_top3 = grp_j.apply(lambda g: (g["finish_position"] <= 3).sum(), include_groups=False)
    j_win = grp_j.apply(lambda g: (g["finish_position"] == 1).sum(), include_groups=False)

    jockey_top3_prior = (df_venue["finish_position"] <= 3).mean() if len(df_venue) > 0 else global_top3_prior
    jockey_win_prior = (df_venue["finish_position"] == 1).mean() if len(df_venue) > 0 else global_win_prior

    # ── 馬ごとに辞書化 ──
    all_horses = set(total_n.index) | set(venue_n.index)
    horse_stats = {}
    for name in all_horses:
        nt = total_n.get(name, 0)
        nv = venue_n.get(name, 0)
        horse_stats[name] = {
            "n_total": nt,
            "n_venue": nv,
            "top3_rate_total": _bayes(total_top3.get(name, 0), nt, global_top3_prior, K_HORSE),
            "win_rate_total": _bayes(total_win.get(name, 0), nt, global_win_prior, K_HORSE),
            "top3_rate_venue": _bayes(venue_top3.get(name, 0), nv, jockey_top3_prior, K_HORSE),
            "win_rate_venue": _bayes(venue_win.get(name, 0), nv, jockey_win_prior, K_HORSE),
            "recent_avg_pos": recent_avg.get(name, np.nan),
            "recent_top3": recent_top3_rate.get(name, np.nan),
            "recent_venue_avg_pos": recent_venue_avg.get(name, np.nan),
            "last_race_date": last_date.get(name, pd.NaT),
        }

    # ── 騎手辞書 ──
    jockey_stats = {}
    for j in j_n.index:
        jn = j_n.get(j, 0)
        jockey_stats[j] = {
            "jockey_top3_rate": _bayes(j_top3.get(j, 0), jn, jockey_top3_prior, K_JOCKEY),
            "jockey_win_rate": _bayes(j_win.get(j, 0), jn, jockey_win_prior, K_JOCKEY),
        }

    return {
        "horse": horse_stats,
        "jockey": jockey_stats,
        "priors": {
            "top3": global_top3_prior,
            "win": global_win_prior,
            "j_top3": jockey_top3_prior,
            "j_win": jockey_win_prior,
        }
    }


# ── 特徴量構築 ─────────────────────────────────────────────

def build_features(df_race: pd.DataFrame, stats: dict, pred_date: pd.Timestamp) -> pd.DataFrame:
    """1レース分の特徴量をDataFrameで返す"""
    priors = stats["priors"]
    rows = []
    for _, h in df_race.iterrows():
        name = h["horse_name"]
        hs = stats["horse"].get(name, {})
        js = stats["jockey"].get(h.get("jockey", ""), {})

        # 休養日数
        last_dt = hs.get("last_race_date", pd.NaT)
        days = (pred_date - last_dt).days if pd.notna(last_dt) else 999

        n_total = hs.get("n_total", 0)
        row = {
            "horse_name": name,
            "top3_rate_venue": hs.get("top3_rate_venue", _bayes(0, 0, priors["j_top3"], K_HORSE)),
            "win_rate_venue": hs.get("win_rate_venue", _bayes(0, 0, priors["j_win"], K_HORSE)),
            "top3_rate_total": hs.get("top3_rate_total", _bayes(0, 0, priors["top3"], K_HORSE)),
            "win_rate_total": hs.get("win_rate_total", _bayes(0, 0, priors["win"], K_HORSE)),
            "n_venue": hs.get("n_venue", 0),
            "n_total": n_total,
            "recent_avg_pos": hs.get("recent_avg_pos", h.get("field_size", 8) * 0.6),
            "recent_top3": hs.get("recent_top3", priors["top3"]),
            "recent_venue_avg_pos": hs.get("recent_venue_avg_pos", h.get("field_size", 8) * 0.6),
            "jockey_top3_rate": js.get("jockey_top3_rate", _bayes(0, 0, priors["j_top3"], K_JOCKEY)),
            "jockey_win_rate": js.get("jockey_win_rate", _bayes(0, 0, priors["j_win"], K_JOCKEY)),
            "age": h.get("age", 0) or 0,
            "weight_carried": h.get("weight_carried", 55.0) or 55.0,
            "sex_enc": SEX_MAP.get(str(h.get("sex", "")), 0),
            "field_size": h.get("field_size", 8),
            "distance": h.get("distance", 1400),
            "days_since_last": min(days, 999),
            "umaban": h.get("horse_no", 0),
            "data_reliability": n_total / (n_total + DATA_K),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── 学習データ構築（履歴の各レースに特徴量を付与） ────────────

def build_train_data(df_hist: pd.DataFrame) -> tuple:
    """
    LeaveOneYearOut用の学習データ構築。
    ただし全年一括で返す（バックテスト時は呼び出し側でスライス）。
    各レースに対して「そのレース以前の統計」を使う（理想的には）が、
    大量計算になるため月単位で近似する。
    返り値: X (features), y (finish_position), groups (race_id)
    """
    df = df_hist.copy()
    df["year_month"] = df["race_date"].dt.to_period("M")
    periods = sorted(df["year_month"].unique())

    all_X, all_y, all_groups = [], [], []

    for i, period in enumerate(periods):
        if i < 2:  # 最初の2ヶ月は統計が少なすぎるのでスキップ
            continue
        # この月のデータ
        df_test = df[df["year_month"] == period]
        # この月より前の統計（当月のデータは使わない = リーク防止）
        df_before = df[df["year_month"] < period]
        if df_before.empty:
            continue

        stats = compute_stats(df_before)
        pred_date = pd.Timestamp(str(period.start_time))

        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            Xr = build_features(race_df, stats, pred_date)
            yr = race_df["finish_position"].values
            all_X.append(Xr)
            all_y.append(yr)
            all_groups.extend([race_id] * len(race_df))

    if not all_X:
        return pd.DataFrame(), np.array([]), np.array([])

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)
    return X, y, groups


# ── LightGBM学習 ──────────────────────────────────────────

def train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> lgb.Booster:
    """LightGBM rank:ndcg モデルを学習"""
    X_feat = X[FEATURE_COLS].fillna(X[FEATURE_COLS].median())

    # グループ数を計算
    _, group_sizes = np.unique(groups, return_counts=True)

    params = {
        "objective": "rank_xendcg",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    # グループ順にソート（LightGBM rank要件）
    sort_idx = np.argsort(groups)
    X_sorted = X_feat.iloc[sort_idx]
    y_sorted = y[sort_idx]
    groups_sorted = groups[sort_idx]
    _, group_sizes_sorted = np.unique(groups_sorted, return_counts=True)

    # ラベル: 着順を降順スコアに変換 (1着=max_score, 最下位=1)
    label = np.zeros_like(y_sorted, dtype=float)
    for g_id in np.unique(groups_sorted):
        mask = groups_sorted == g_id
        pos = y_sorted[mask]
        n = pos.max()
        label[mask] = np.maximum(0, n - pos + 1)

    dataset = lgb.Dataset(X_sorted, label=label, group=group_sizes_sorted)
    model = lgb.train(
        params,
        dataset,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    return model


# ── 予測 ──────────────────────────────────────────────────

def predict_race(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """各馬のスコアを返す（高いほど強い）"""
    X_feat = X[FEATURE_COLS].fillna(X[FEATURE_COLS].median())
    return model.predict(X_feat)


def do_predict(df_hist: pd.DataFrame, date_str: str) -> None:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    shutuba_path = RACES_DIR / f"kawasaki_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    if not shutuba_path.exists():
        print(f"出走表が見つかりません: {shutuba_path}")
        print(f"まず: python collect_kawasaki_shutuba.py --date {date_str}")
        return

    df_shutuba = pd.read_csv(shutuba_path, encoding="utf-8-sig")
    print(f"出走表: {shutuba_path} ({len(df_shutuba)}頭, {df_shutuba['race_no'].nunique()}レース)")

    # 予測日より前のデータで学習
    pred_date = pd.Timestamp(dt)
    df_train_hist = df_hist[df_hist["race_date"] < pred_date].copy()
    if df_train_hist.empty:
        print("警告: 予測日より前の履歴データがありません。全データで統計計算します。")
        df_train_hist = df_hist.copy()

    print("統計計算中...")
    stats = compute_stats(df_train_hist)

    print("学習データ構築中...")
    X, y, groups = build_train_data(df_train_hist)
    if X.empty:
        print("学習データが不足。統計ベースのみで予測します。")
        _predict_stats_only(df_shutuba, stats, pred_date)
        return

    print(f"学習データ: {len(X):,}行, {groups.size and len(np.unique(groups)):,}レース")
    print("モデル学習中...")
    model = train_model(X, y, groups)

    # 各レース予測
    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        if len(race_df) < 2:
            continue
        race_df = race_df.copy().reset_index(drop=True)
        Xr = build_features(race_df, stats, pred_date)
        scores = predict_race(model, Xr)

        # スコアを確率に変換（softmax近似）
        scores = scores - scores.min()
        total = scores.sum()
        base_rate = 3.0 / max(len(race_df), 1)
        raw_prob = scores / total * 3.0 if total > 0 else np.ones(len(scores)) * base_rate
        rel = Xr["data_reliability"].values
        top3_prob = raw_prob * rel + base_rate * (1 - rel)

        race_df["score"] = scores
        race_df["top3_prob"] = top3_prob
        race_df["data_reliability"] = rel
        race_df["n_hist"] = Xr["n_total"].values
        race_df["n_venue"] = Xr["n_venue"].values
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
    """学習データ不足時: 統計値のみで順位付け"""
    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        race_df = race_df.copy().reset_index(drop=True)
        Xr = build_features(race_df, stats, pred_date)
        score = (
            Xr["top3_rate_venue"] * 2.0
            + Xr["win_rate_venue"] * 1.5
            + Xr["top3_rate_total"] * 1.0
            + Xr["jockey_top3_rate"] * 0.5
        )
        total = score.sum()
        base_rate = 3.0 / max(len(race_df), 1)
        raw_prob = score / total * 3.0 if total > 0 else np.ones(len(score)) * base_rate
        rel = Xr["data_reliability"].values
        race_df["score"] = score.values
        race_df["top3_prob"] = raw_prob * rel + base_rate * (1 - rel)
        race_df["data_reliability"] = rel
        race_df["n_hist"] = Xr["n_total"].values
        race_df["n_venue"] = Xr["n_venue"].values
        out_rows.append(race_df)

    df_pred = pd.concat(out_rows, ignore_index=True)
    _print_predictions(df_pred, pred_date.strftime("%Y/%m/%d"))


def _print_predictions(df_pred: pd.DataFrame, date_str: str) -> None:
    print(f"\n{'='*72}")
    print(f"  川崎競馬 {date_str} 予測")
    print(f"{'='*72}")
    for race_no, grp in df_pred.groupby("race_no"):
        top = grp.sort_values("top3_prob", ascending=False)
        meta = top.iloc[0]
        print(f"\n【{race_no:2d}R】{meta.get('race_name','')}  "
              f"{int(meta.get('distance',0))}m  "
              f"({meta.get('track_cond','')})")
        rows = []
        for rank, (_, h) in enumerate(top.iterrows(), 1):
            rel = h.get("data_reliability", 0)
            n_v = int(h.get("n_venue", 0))
            n_t = int(h.get("n_hist", 0))
            warn = " ⚠" if rel < 0.3 else ""
            rows.append([
                rank,
                int(h.get("horse_no", 0)),
                h.get("horse_name", ""),
                f"{h['top3_prob']:.1%}",
                f"{rel:.2f}",
                f"{n_v}/{n_t}",
                h.get("jockey", ""),
                warn,
            ])
        print(tabulate(rows,
                       headers=["順", "馬番", "馬名", "3着内確率", "信頼度", "川崎/通算", "騎手", ""],
                       tablefmt="simple"))


# ── バックテスト ──────────────────────────────────────────

def do_backtest(df_hist: pd.DataFrame) -> None:
    """LeaveOneYearOut バックテスト（川崎データのみで評価）"""
    df_kawasaki = df_hist[df_hist["venue"] == VENUE].copy()
    years = sorted(df_kawasaki["race_date"].dt.year.unique())
    if len(years) < 2:
        print("バックテストには最低2年分の川崎データが必要です")
        return

    print(f"\n=== LeaveOneYearOut バックテスト ({VENUE}) ===")
    print(f"対象年: {years}")
    results = []

    for test_year in years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test = df_kawasaki[df_kawasaki["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue

        stats = compute_stats(df_train)
        pred_date = pd.Timestamp(f"{test_year}-01-01")
        X_tr, y_tr, groups_tr = build_train_data(df_train)
        if X_tr.empty:
            print(f"  {test_year}: 学習データ不足 → スキップ")
            continue

        model = train_model(X_tr, y_tr, groups_tr)

        top5_total, top3_total, top1_total, race_count = 0, 0, 0, 0
        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            race_pred_date = race_df["race_date"].iloc[0]
            Xr = build_features(race_df.reset_index(drop=True), stats, race_pred_date)
            scores = predict_race(model, Xr)

            actual_top3 = set(race_df.index[race_df["finish_position"] <= 3])
            pred_sorted = race_df.index[np.argsort(-scores)]

            top5_hit = len(actual_top3 & set(pred_sorted[:5]))
            top5_total += top5_hit / min(3, len(actual_top3))
            top3_total += len(actual_top3 & set(pred_sorted[:3])) / min(3, len(actual_top3))
            top1_total += int(pred_sorted[0] in actual_top3)
            race_count += 1

        if race_count == 0:
            continue

        r = {
            "year": test_year,
            "races": race_count,
            "top5_coverage": top5_total / race_count,
            "top3_hit": top3_total / race_count,
            "top1_acc": top1_total / race_count,
        }
        results.append(r)
        print(f"  {test_year}: {race_count}レース "
              f"top5={r['top5_coverage']:.1%} "
              f"top3={r['top3_hit']:.1%} "
              f"top1={r['top1_acc']:.1%}")

    if results:
        avg_top5 = np.mean([r["top5_coverage"] for r in results])
        avg_top3 = np.mean([r["top3_hit"] for r in results])
        avg_top1 = np.mean([r["top1_acc"] for r in results])
        print(f"\n平均 top5_coverage: {avg_top5:.1%}")
        print(f"平均 top3_hit     : {avg_top3:.1%}")
        print(f"平均 top1_acc     : {avg_top1:.1%}")


# ── main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="川崎競馬予測 (LightGBM rank)")
    parser.add_argument("--date", help="予測日 YYYY/MM/DD（省略時=翌日）")
    parser.add_argument("--backtest", action="store_true", help="LeaveOneYearOutバックテスト")
    parser.add_argument("--no-oi", action="store_true", help="大井データを補助に使わない")
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
