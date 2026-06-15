#!/usr/bin/env python3
"""川崎競馬 予測スクリプト（tihouモデル不使用のクリーン実装）

Usage:
    python kawasaki_predict.py --date 2026/06/15
    python kawasaki_predict.py --backtest
    python kawasaki_predict.py --tune --trials 100

特徴量 (31種):
    - 当場成績率・通算成績率（Bayesian平滑化）
    - 当場当距離3着内率 / 馬場状態別3着内率 [v2追加]
    - 直近フォーム / 当場直近フォーム / フォームトレンド [v2追加]
    - 騎手実績・調教師実績（当場）[v2追加: 調教師]
    - テン乗りフラグ / 昇降級フラグ [v2追加]
    - 速度指数（相対タイム）/ 上がり3F指数 / 脚質（通過順位）
    - レースクラス
    - 物理情報（年齢・斤量・馬番・距離）
    - データ信頼度

モデル: LightGBM rank:ndcg
根拠:  SHAPベースの主要因表示
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
OI_DIR       = Path("data/historical_oi")
RACES_DIR    = Path("data/races")
PARAMS_PATH  = Path("data/kawasaki_best_params.json")
OPTUNA_DB    = Path("data/kawasaki_optuna.db")
VENUE        = "川崎"

# ── Bayesian平滑化パラメータ ──────────────────────────────
K_HORSE  = 8
K_JOCKEY = 30
DATA_K   = 8

# ── 特徴量列 ──────────────────────────────────────────────
FEATURE_COLS = [
    # 当場成績
    "top3_rate_venue",
    "win_rate_venue",
    "n_venue",
    # 当場×当距離成績 [v2]
    "top3_rate_venue_dist",
    "n_venue_dist",
    # 通算成績
    "top3_rate_total",
    "win_rate_total",
    "n_total",
    # 馬場状態別成績 [v2]
    "top3_rate_cond",
    "n_cond",
    # 直近フォーム
    "recent_avg_pos",
    "recent_top3",
    "recent_venue_avg_pos",
    # フォームトレンド [v2]: 正=改善（着順下落）、負=悪化
    "form_trend",
    # 速度指数・上がり・脚質
    "avg_speed_idx",
    "best_speed_idx",
    "avg_last3f_idx",
    "avg_corner_ratio",
    # 騎手
    "jockey_top3_rate",
    "jockey_win_rate",
    # 調教師 [v2]
    "trainer_top3_rate",
    # テン乗り
    "is_ten_nori",
    # 昇降級フラグ [v2]: 正=昇級(harder)、負=降級(easier)、0=同クラス
    "class_change",
    # レース属性
    "race_class_enc",
    "distance",
    "field_size",
    # 馬属性
    "age",
    "weight_carried",
    "sex_enc",
    "umaban",
    "days_since_last",
    "data_reliability",
]

FEATURE_LABEL = {
    "top3_rate_venue":      "川崎3着内率",
    "win_rate_venue":       "川崎勝率",
    "n_venue":              "川崎出走数",
    "top3_rate_venue_dist": "川崎同距3着内率",
    "n_venue_dist":         "川崎同距出走数",
    "top3_rate_total":      "通算3着内率",
    "win_rate_total":       "通算勝率",
    "n_total":              "通算出走数",
    "top3_rate_cond":       "馬場状態3着内率",
    "n_cond":               "同馬場出走数",
    "recent_avg_pos":       "直近着順",
    "recent_top3":          "直近3着内率",
    "recent_venue_avg_pos": "川崎直近着順",
    "form_trend":           "フォーム改善",
    "avg_speed_idx":        "速度指数",
    "best_speed_idx":       "最高速度",
    "avg_last3f_idx":       "上がり3F",
    "avg_corner_ratio":     "脚質(通過順)",
    "jockey_top3_rate":     "騎手実績",
    "jockey_win_rate":      "騎手勝率",
    "trainer_top3_rate":    "調教師実績",
    "is_ten_nori":          "テン乗り",
    "class_change":         "昇降級",
    "race_class_enc":       "レースクラス",
    "distance":             "距離",
    "field_size":           "頭数",
    "age":                  "年齢",
    "weight_carried":       "斤量",
    "sex_enc":              "性別",
    "umaban":               "馬番",
    "days_since_last":      "休養日数",
    "data_reliability":     "データ量",
}

SEX_MAP = {"牡": 0, "牝": 1, "セン": 2, "": 0}


# ── ヘルパー関数 ───────────────────────────────────────────

def _parse_time_secs(s) -> float:
    """'1:43.0' → 103.0, 空/不正 → NaN"""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    s = str(s).strip()
    try:
        if ':' in s:
            m, sec = s.split(':', 1)
            return float(m) * 60 + float(sec)
        return float(s)
    except (ValueError, AttributeError):
        return np.nan


def _extract_last_corner(passage_str) -> float:
    """'1-1-2-3' の最終コーナー位置を返す。不正 → NaN"""
    if pd.isna(passage_str) or not str(passage_str).strip():
        return np.nan
    parts = str(passage_str).strip().split('-')
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return np.nan


def _extract_class(race_name: str) -> int:
    s = str(race_name or '')
    for marker, score in [
        ('Ａ１', 7), ('A1', 7), ('Ａ２', 6), ('A2', 6),
        ('Ｂ１', 5), ('B1', 5), ('Ｂ２', 4), ('B2', 4),
        ('Ｂ３', 3), ('B3', 3), ('Ｃ１', 3), ('C1', 3),
        ('Ｃ２', 2), ('C2', 2), ('Ｃ３', 1), ('C3', 1),
    ]:
        if marker in s:
            return score
    if '特別' in s or '賞' in s:
        return 5  # 特別・重賞はB1相当以上
    if '未格付' in s or '２歳' in s or '３歳' in s:
        return 0
    return 1


def add_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    速度指数・上がり3F指数・脚質を一括で付与（全データで1回だけ呼ぶ）。
    per-race median を基準にした相対指数なので場・距離の違いに頑健。
    """
    df = df.copy()
    df['_time_sec']  = df['finish_time'].apply(_parse_time_secs)
    df['_last3f']    = pd.to_numeric(df['last_3f'], errors='coerce')
    df['_race_key']  = (df['race_date'].dt.strftime("%Y%m%d") + "_"
                        + df['race_no'].astype(str) + "_"
                        + df['venue'].astype(str))

    race_med_time  = df.groupby('_race_key')['_time_sec'].transform('median')
    race_med_last3f = df.groupby('_race_key')['_last3f'].transform('median')

    df['speed_idx'] = np.where(
        df['_time_sec'].notna() & race_med_time.notna() & (df['_time_sec'] > 0),
        race_med_time / df['_time_sec'], np.nan
    )
    df['last3f_idx'] = np.where(
        df['_last3f'].notna() & race_med_last3f.notna() & (df['_last3f'] > 0),
        race_med_last3f / df['_last3f'], np.nan
    )
    last_pos = df['passage_rate'].apply(_extract_last_corner)
    df['corner_ratio'] = last_pos / df['field_size'].clip(lower=1)

    return df.drop(columns=['_time_sec', '_last3f', '_race_key'])


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
    df["race_date"]        = pd.to_datetime(df["race_date"])
    df["finish_position"]  = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[df["finish_position"].notna() & (df["finish_position"] >= 1)].copy()
    df["finish_position"]  = df["finish_position"].astype(int)

    print(f"速度指数を計算中...")
    df = add_speed_features(df)

    venues = df["venue"].unique().tolist()
    spd_cov = df['speed_idx'].notna().mean()
    l3f_cov = df['last3f_idx'].notna().mean()
    print(f"履歴データ: {len(df):,}行 "
          f"({df['race_date'].dt.year.min()}〜{df['race_date'].dt.year.max()}) "
          f"馬場: {venues}")
    print(f"  速度指数カバレッジ: {spd_cov:.1%}  上がり指数: {l3f_cov:.1%}")
    return df


# ── 統計計算 ───────────────────────────────────────────────

def _bayes(num, den, prior, k):
    return (num + prior * k) / (den + k)


def compute_stats(df_hist: pd.DataFrame) -> dict:
    """ベクトル化による高速版（apply/lambda を排除）"""
    df = df_hist.copy()
    df["_is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["_is_win"]  = (df["finish_position"] == 1).astype(int)

    global_top3_prior = df["_is_top3"].mean()
    global_win_prior  = df["_is_win"].mean()

    total_n    = df.groupby("horse_name").size()
    total_top3 = df.groupby("horse_name")["_is_top3"].sum()
    total_win  = df.groupby("horse_name")["_is_win"].sum()

    df_venue = df[df["venue"] == VENUE].copy()
    j_top3_prior = df_venue["_is_top3"].mean() if len(df_venue) > 0 else global_top3_prior
    j_win_prior  = df_venue["_is_win"].mean()  if len(df_venue) > 0 else global_win_prior

    venue_n    = df_venue.groupby("horse_name").size()
    venue_top3 = df_venue.groupby("horse_name")["_is_top3"].sum()
    venue_win  = df_venue.groupby("horse_name")["_is_win"].sum()

    # 直近5走（cumcount降順でtail(5)を再現）
    df_s = df.sort_values(["horse_name", "race_date"])
    df_s["_rev_rank"] = df_s.groupby("horse_name").cumcount(ascending=False)
    recent = df_s[df_s["_rev_rank"] < 5].copy()
    recent_avg       = recent.groupby("horse_name")["finish_position"].mean()
    recent_top3_rate = recent.groupby("horse_name")["_is_top3"].mean()

    df_vs = df_venue.sort_values(["horse_name", "race_date"])
    df_vs["_rev_rank"] = df_vs.groupby("horse_name").cumcount(ascending=False)
    recent_v     = df_vs[df_vs["_rev_rank"] < 5]
    recent_venue_avg = recent_v.groupby("horse_name")["finish_position"].mean()

    last_date = df_s.groupby("horse_name")["race_date"].max()

    j_n     = df_venue.groupby("jockey").size()
    j_top3  = df_venue.groupby("jockey")["_is_top3"].sum()
    j_win   = df_venue.groupby("jockey")["_is_win"].sum()

    horse_jockey_pairs = set(zip(df["horse_name"], df["jockey"]))

    # 速度指数（ベクトル化）
    avg_speed_idx  = pd.Series(dtype=float)
    best_speed_idx = pd.Series(dtype=float)
    avg_last3f_idx = pd.Series(dtype=float)
    avg_corner     = pd.Series(dtype=float)

    if "speed_idx" in df.columns:
        sp_valid = df[df["speed_idx"].notna()].copy()
        avg_speed_idx = sp_valid.groupby("horse_name")["speed_idx"].mean()
        sp_valid["_sp_rank"] = sp_valid.groupby("horse_name")["speed_idx"].rank(
            method="first", ascending=False)
        best_speed_idx = (sp_valid[sp_valid["_sp_rank"] <= 3]
                          .groupby("horse_name")["speed_idx"].mean())
        avg_last3f_idx = df[df["last3f_idx"].notna()].groupby("horse_name")["last3f_idx"].mean()
        avg_corner     = df[df["corner_ratio"].notna()].groupby("horse_name")["corner_ratio"].mean()

    # ── 当場×当距離成績 [v2] ──
    vd_n    = df_venue.groupby(["horse_name", "distance"]).size()
    vd_top3 = df_venue.groupby(["horse_name", "distance"])["_is_top3"].sum()
    # per-horse dict: { distance: {"n": n, "top3_rate": r} }
    vd_dict: dict[str, dict] = {}
    for (hname, dist), cnt in vd_n.items():
        if hname not in vd_dict:
            vd_dict[hname] = {}
        top3_cnt = int(vd_top3.get((hname, dist), 0))
        vd_dict[hname][dist] = {
            "n": int(cnt),
            "top3_rate": _bayes(top3_cnt, int(cnt), j_top3_prior, K_HORSE),
        }

    # ── 馬場状態別成績 [v2] ──
    df_cond = df[df["track_cond"].notna() & (df["track_cond"] != "")].copy()
    cond_n    = df_cond.groupby(["horse_name", "track_cond"]).size()
    cond_top3 = df_cond.groupby(["horse_name", "track_cond"])["_is_top3"].sum()
    cond_dict: dict[str, dict] = {}
    for (hname, cond), cnt in cond_n.items():
        if hname not in cond_dict:
            cond_dict[hname] = {}
        top3_cnt = int(cond_top3.get((hname, cond), 0))
        cond_dict[hname][cond] = {
            "n": int(cnt),
            "top3_rate": _bayes(top3_cnt, int(cnt), global_top3_prior, K_HORSE),
        }

    # ── 調教師実績（当場）[v2] ──
    t_n     = df_venue.groupby("trainer").size()
    t_top3  = df_venue.groupby("trainer")["_is_top3"].sum()
    t_win   = df_venue.groupby("trainer")["_is_win"].sum()

    trainer_stats: dict[str, dict] = {}
    for tr in t_n.index:
        tn = int(t_n.get(tr, 0))
        trainer_stats[tr] = {
            "trainer_top3_rate": _bayes(t_top3.get(tr, 0), tn, j_top3_prior, K_JOCKEY),
        }

    # ── フォームトレンド [v2] ──
    # 直近3走の着順変化: (3走前着順 - 最新着順) → 正=改善
    df_s3 = df.sort_values(["horse_name", "race_date"])
    df_s3["_rev_idx"] = df_s3.groupby("horse_name").cumcount(ascending=False)
    pos_latest = (df_s3[df_s3["_rev_idx"] == 0]
                  .set_index("horse_name")["finish_position"])
    pos_3rd    = (df_s3[df_s3["_rev_idx"] == 2]
                  .set_index("horse_name")["finish_position"])
    form_trend_s = (pos_3rd - pos_latest)  # 正=改善、負=悪化

    # ── 前走クラス（昇降級判定用）[v2] ──
    df_s3["_class_enc"] = df_s3["race_name"].apply(_extract_class)
    last_class_s = (df_s3[df_s3["_rev_idx"] == 0]
                    .set_index("horse_name")["_class_enc"])

    all_horses = set(total_n.index) | set(venue_n.index)
    horse_stats = {}
    for name in all_horses:
        nt = int(total_n.get(name, 0))
        nv = int(venue_n.get(name, 0))
        horse_stats[name] = {
            "n_total":              nt,
            "n_venue":              nv,
            "top3_rate_total":      _bayes(total_top3.get(name, 0), nt, global_top3_prior, K_HORSE),
            "win_rate_total":       _bayes(total_win.get(name, 0),  nt, global_win_prior,  K_HORSE),
            "top3_rate_venue":      _bayes(venue_top3.get(name, 0), nv, j_top3_prior, K_HORSE),
            "win_rate_venue":       _bayes(venue_win.get(name, 0),  nv, j_win_prior,  K_HORSE),
            "recent_avg_pos":       recent_avg.get(name, np.nan),
            "recent_top3":          recent_top3_rate.get(name, np.nan),
            "recent_venue_avg_pos": recent_venue_avg.get(name, np.nan),
            "last_race_date":       last_date.get(name, pd.NaT),
            "avg_speed_idx":        avg_speed_idx.get(name, np.nan),
            "best_speed_idx":       best_speed_idx.get(name, np.nan),
            "avg_last3f_idx":       avg_last3f_idx.get(name, np.nan),
            "avg_corner_ratio":     avg_corner.get(name, np.nan),
            # v2
            "vd_stats":             vd_dict.get(name, {}),
            "cond_stats":           cond_dict.get(name, {}),
            "form_trend":           form_trend_s.get(name, np.nan),
            "last_class":           last_class_s.get(name, np.nan),
        }

    jockey_stats = {}
    for j in j_n.index:
        jn = int(j_n.get(j, 0))
        jockey_stats[j] = {
            "jockey_top3_rate": _bayes(j_top3.get(j, 0), jn, j_top3_prior, K_JOCKEY),
            "jockey_win_rate":  _bayes(j_win.get(j, 0),  jn, j_win_prior,  K_JOCKEY),
        }

    return {
        "horse":              horse_stats,
        "jockey":             jockey_stats,
        "trainer":            trainer_stats,
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
    priors          = stats["priors"]
    pairs           = stats["horse_jockey_pairs"]
    trainer_stats   = stats.get("trainer", {})
    rows = []
    for _, h in df_race.iterrows():
        name    = h["horse_name"]
        jockey  = h.get("jockey", "")
        trainer = h.get("trainer", "")
        hs = stats["horse"].get(name, {})
        js = stats["jockey"].get(jockey, {})
        ts = trainer_stats.get(trainer, {})

        last_dt = hs.get("last_race_date", pd.NaT)
        days    = (pred_date - last_dt).days if pd.notna(last_dt) else 999
        n_total = hs.get("n_total", 0)
        fs      = h.get("field_size", 8) or 8
        distance    = h.get("distance", 1400)
        track_cond  = str(h.get("track_cond", "") or "")
        race_cls    = _extract_class(h.get("race_name", ""))

        def spd(key, default=1.0):
            v = hs.get(key, np.nan)
            return v if pd.notna(v) else default

        is_ten_nori = 0 if (name, jockey) in pairs else 1

        # 当場×当距離 [v2]
        vd     = hs.get("vd_stats", {}).get(distance, {})
        n_vd   = vd.get("n", 0)
        top3_vd = vd.get("top3_rate", _bayes(0, 0, priors["j_top3"], K_HORSE))

        # 馬場状態別 [v2]
        cd     = hs.get("cond_stats", {}).get(track_cond, {})
        n_cd   = cd.get("n", 0)
        top3_cd = cd.get("top3_rate", _bayes(0, 0, priors["top3"], K_HORSE))

        # フォームトレンド [v2]
        ft_raw = hs.get("form_trend", np.nan)
        form_trend = float(ft_raw) if pd.notna(ft_raw) else 0.0

        # 昇降級 [v2]: 正=昇級(harder)、負=降級(easier)
        last_cls_raw = hs.get("last_class", np.nan)
        last_cls     = int(last_cls_raw) if pd.notna(last_cls_raw) else race_cls
        class_change = race_cls - last_cls

        # 調教師 [v2]
        trainer_top3 = ts.get("trainer_top3_rate",
                               _bayes(0, 0, priors["j_top3"], K_JOCKEY))

        row = {
            "horse_name":           name,
            "top3_rate_venue":      hs.get("top3_rate_venue",      _bayes(0, 0, priors["j_top3"], K_HORSE)),
            "win_rate_venue":       hs.get("win_rate_venue",        _bayes(0, 0, priors["j_win"],  K_HORSE)),
            "n_venue":              hs.get("n_venue", 0),
            "top3_rate_venue_dist": top3_vd,
            "n_venue_dist":         n_vd,
            "top3_rate_total":      hs.get("top3_rate_total",       _bayes(0, 0, priors["top3"],   K_HORSE)),
            "win_rate_total":       hs.get("win_rate_total",        _bayes(0, 0, priors["win"],    K_HORSE)),
            "n_total":              n_total,
            "top3_rate_cond":       top3_cd,
            "n_cond":               n_cd,
            "recent_avg_pos":       hs.get("recent_avg_pos",        fs * 0.6),
            "recent_top3":          hs.get("recent_top3",           priors["top3"]),
            "recent_venue_avg_pos": hs.get("recent_venue_avg_pos",  fs * 0.6),
            "form_trend":           form_trend,
            "avg_speed_idx":        spd("avg_speed_idx"),
            "best_speed_idx":       spd("best_speed_idx"),
            "avg_last3f_idx":       spd("avg_last3f_idx"),
            "avg_corner_ratio":     spd("avg_corner_ratio", default=0.5),
            "jockey_top3_rate":     js.get("jockey_top3_rate", _bayes(0, 0, priors["j_top3"], K_JOCKEY)),
            "jockey_win_rate":      js.get("jockey_win_rate",  _bayes(0, 0, priors["j_win"],  K_JOCKEY)),
            "trainer_top3_rate":    trainer_top3,
            "is_ten_nori":          is_ten_nori,
            "class_change":         class_change,
            "race_class_enc":       race_cls,
            "distance":             distance,
            "field_size":           fs,
            "age":                  h.get("age", 0) or 0,
            "weight_carried":       h.get("weight_carried", 55.0) or 55.0,
            "sex_enc":              SEX_MAP.get(str(h.get("sex", "")), 0),
            "umaban":               h.get("horse_no", 0),
            "days_since_last":      min(days, 999),
            "data_reliability":     n_total / (n_total + DATA_K),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── 学習データ構築 ─────────────────────────────────────────

def build_train_data(df_hist: pd.DataFrame) -> tuple:
    """年次ローリングウィンドウで学習データを構築（高速版: 月次→年次でcompute_stats呼び出しを1/12に削減）"""
    df = df_hist.copy()
    df["_year"] = df["race_date"].dt.year
    years = sorted(df["_year"].unique())

    all_X, all_y, all_groups = [], [], []
    for i, year in enumerate(years):
        if i < 1:
            continue
        df_test   = df[df["_year"] == year]
        df_before = df[df["_year"] < year]
        if df_before.empty:
            continue
        stats     = compute_stats(df_before)
        pred_date = pd.Timestamp(f"{year}-01-01")

        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_"
            + df_test["race_no"].astype(str)
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

_DEFAULT_PARAMS = {
    "objective":         "rank_xendcg",
    "metric":            "ndcg",
    "ndcg_eval_at":      [3, 5],
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "verbose":           -1,
    "n_jobs":            -1,
}
_DEFAULT_ROUNDS = 300


def _load_params() -> tuple[dict, int]:
    """保存済みパラメータを読み込む。なければデフォルト値。"""
    import json
    if PARAMS_PATH.exists():
        saved = json.loads(PARAMS_PATH.read_text())
        num_rounds = int(saved.pop("num_boost_round", _DEFAULT_ROUNDS))
        params = {**_DEFAULT_PARAMS, **saved}
        return params, num_rounds
    return dict(_DEFAULT_PARAMS), _DEFAULT_ROUNDS


def train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                params: dict | None = None, num_rounds: int | None = None) -> lgb.Booster:
    if params is None:
        params, num_rounds = _load_params()
    if num_rounds is None:
        num_rounds = _DEFAULT_ROUNDS

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
    verbose = params.get("verbose", -1)
    cbs = [lgb.log_evaluation(period=100)] if verbose >= 0 else []
    return lgb.train(params, dataset, num_boost_round=num_rounds, callbacks=cbs)


# ── SHAP根拠生成 ──────────────────────────────────────────

def make_reason(shap_row: np.ndarray, feature_names: list[str],
                horse_row: pd.Series) -> str:
    n = len(feature_names)
    shap_vals = shap_row[:n]

    is_ten = horse_row.get("is_ten_nori", 0) == 1
    ten_label = "【テン乗り】" if is_ten else ""

    feat_shap = [(f, v) for f, v in zip(feature_names, shap_vals)
                 if f != "is_ten_nori"]
    pos = sorted([(f, v) for f, v in feat_shap if v > 0], key=lambda x: -x[1])[:2]
    neg = sorted([(f, v) for f, v in feat_shap if v < 0], key=lambda x:  x[1])[:1]

    parts  = [f"↑{FEATURE_LABEL.get(f, f)}" for f, _ in pos]
    parts += [f"↓{FEATURE_LABEL.get(f, f)}" for f, _ in neg]
    return ten_label + " ".join(parts)


# ── 予測 ──────────────────────────────────────────────────

def predict_race(model: lgb.Booster, X: pd.DataFrame):
    X_feat = X[FEATURE_COLS].fillna(X[FEATURE_COLS].median())
    scores = model.predict(X_feat)
    try:
        contrib   = model.predict(X_feat, pred_contrib=True)
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

    pred_date     = pd.Timestamp(dt)
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

    # feature importance 表示（上位10件）
    fi = pd.Series(model.feature_importance(importance_type='gain'),
                   index=FEATURE_COLS).sort_values(ascending=False)
    print("\n[特徴量重要度 top10]")
    for fname, val in fi.head(10).items():
        print(f"  {FEATURE_LABEL.get(fname, fname):16s}: {val:,.0f}")

    out_rows = []
    for race_no, race_df in df_shutuba.groupby("race_no"):
        if len(race_df) < 2:
            continue
        race_df = race_df.copy().reset_index(drop=True)
        Xr      = build_features(race_df, stats, pred_date)
        scores, shap_vals = predict_race(model, Xr)

        exp_s     = np.exp(scores - scores.max())
        raw_prob  = exp_s / exp_s.sum() * 3.0
        base_rate = 3.0 / max(len(race_df), 1)
        rel       = Xr["data_reliability"].values
        top3_prob = raw_prob * rel + base_rate * (1 - rel)

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
        race_df["avg_speed_idx"]    = Xr["avg_speed_idx"].values
        race_df["avg_last3f_idx"]   = Xr["avg_last3f_idx"].values
        race_df["avg_corner_ratio"] = Xr["avg_corner_ratio"].values
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
            Xr["top3_rate_venue"]   * 2.0
            + Xr["win_rate_venue"]  * 1.5
            + Xr["avg_speed_idx"]   * 1.0
            + Xr["jockey_top3_rate"] * 0.5
        )
        exp_s     = np.exp(score - score.max())
        raw_prob  = exp_s / exp_s.sum() * 3.0
        base_rate = 3.0 / max(len(race_df), 1)
        rel = Xr["data_reliability"].values
        race_df["score"]            = score.values
        race_df["top3_prob"]        = raw_prob * rel + base_rate * (1 - rel)
        race_df["data_reliability"] = rel
        race_df["n_hist"]           = Xr["n_total"].values
        race_df["n_venue"]          = Xr["n_venue"].values
        race_df["is_ten_nori"]      = Xr["is_ten_nori"].values
        race_df["reason"]           = [
            ("【テン乗り】" if r == 1 else "") + "統計ベース"
            for r in Xr["is_ten_nori"].values
        ]
        out_rows.append(race_df)
    df_pred = pd.concat(out_rows, ignore_index=True)
    _print_predictions(df_pred, pred_date.strftime("%Y/%m/%d"))


def _print_predictions(df_pred: pd.DataFrame, date_str: str) -> None:
    print(f"\n{'='*82}")
    print(f"  川崎競馬 {date_str} 予測")
    print(f"{'='*82}")
    for race_no, grp in df_pred.groupby("race_no"):
        top  = grp.sort_values("top3_prob", ascending=False)
        meta = top.iloc[0]
        cls  = _extract_class(meta.get("race_name", ""))
        cls_label = {7:"A1",6:"A2",5:"B1",4:"B2",3:"C1/B3",2:"C2",1:"C3",0:"未格付"}.get(cls,"")
        print(f"\n【{race_no:2d}R】{meta.get('race_name','')}  "
              f"{int(meta.get('distance', 0))}m  [{cls_label}]")
        rows = []
        for rank, (_, h) in enumerate(top.iterrows(), 1):
            rel   = h.get("data_reliability", 0)
            n_v   = int(h.get("n_venue", 0))
            n_t   = int(h.get("n_hist", 0))
            spd   = h.get("avg_speed_idx", np.nan)
            l3f   = h.get("avg_last3f_idx", np.nan)
            crn   = h.get("avg_corner_ratio", np.nan)
            warn  = " ⚠" if rel < 0.3 else ""
            spd_s = f"{spd:.3f}" if pd.notna(spd) else "  -  "
            l3f_s = f"{l3f:.3f}" if pd.notna(l3f) else "  -  "
            crn_s = f"{crn:.2f}"  if pd.notna(crn) else "  - "
            reason = h.get("reason", "")
            rows.append([
                rank,
                int(h.get("horse_no", 0)),
                h.get("horse_name", ""),
                f"{h['top3_prob']:.1%}",
                f"{rel:.2f}{warn}",
                f"{n_v}/{n_t}",
                spd_s, l3f_s, crn_s,
                h.get("jockey", ""),
                reason,
            ])
        print(tabulate(rows,
                       headers=["順","馬番","馬名","3着内確率","信頼度","川崎/通算",
                                 "速度idx","上がりidx","脚質","騎手","根拠"],
                       tablefmt="simple"))


# ── Optuna チューニング ────────────────────────────────────

def do_tune(df_hist: pd.DataFrame, n_trials: int = 100) -> None:
    import json
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_kw = df_hist[df_hist["venue"] == VENUE].copy()
    all_years = sorted(df_kw["race_date"].dt.year.unique())
    # 2026年のみをCV対象（最新年の精度を直接最適化）
    test_years = [y for y in all_years if y == 2026]
    if len(test_years) < 1:
        print("チューニングに必要なデータ（2026年）がありません")
        return

    print(f"CV対象年: {test_years}")
    print("年ごとに特徴量を事前計算中（初回のみ時間がかかります）...")

    cv_splits = []
    for test_year in test_years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test  = df_kw[df_kw["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue
        stats = compute_stats(df_train)
        X_tr, y_tr, grp_tr = build_train_data(df_train)
        if X_tr.empty:
            continue

        test_races = []
        for _, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            Xr = build_features(race_df.reset_index(drop=True), stats,
                                 race_df["race_date"].iloc[0])
            actual_top3 = set(race_df.index[race_df["finish_position"] <= 3])
            race_index  = race_df.index.to_numpy()
            race_no     = int(race_df["race_no"].iloc[0])
            test_races.append((Xr, actual_top3, race_index, race_no))

        cv_splits.append((X_tr, y_tr, grp_tr, test_races))
        print(f"  {test_year}: 学習{len(X_tr):,}行, テスト{len(test_races)}R")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "rank_xendcg",
            "metric":            "ndcg",
            "ndcg_eval_at":      [3, 5],
            "verbose":           -1,
            "n_jobs":            -1,
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        num_rounds = trial.suggest_int("num_boost_round", 100, 600)

        total_top5 = total_cnt = 0
        for X_tr, y_tr, grp_tr, test_races in cv_splits:
            model = train_model(X_tr, y_tr, grp_tr, params=params, num_rounds=num_rounds)
            for Xr, actual_top3, race_index, race_no in test_races:
                if race_no < 8:  # 8R以降を評価対象
                    continue
                scores, _ = predict_race(model, Xr)
                pred_sorted = race_index[np.argsort(-scores)]
                total_top5 += len(actual_top3 & set(pred_sorted[:5])) / min(3, len(actual_top3))
                total_cnt  += 1

        return total_top5 / total_cnt if total_cnt > 0 else 0.0

    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_DB}"
    study = optuna.create_study(
        direction="maximize",
        study_name="kawasaki_2026_8R_top5_v2",
        storage=storage,
        load_if_exists=True,
    )
    print(f"\nOptuna チューニング開始 ({n_trials}試行, 2026年8R以降 top5_coverage 最大化)")
    print(f"  DB: {OPTUNA_DB}  続きから再開可能")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**study.best_params, "num_boost_round": study.best_params.get("num_boost_round", _DEFAULT_ROUNDS)}
    PARAMS_PATH.write_text(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\n最良スコア (8R以降 top5_coverage): {study.best_value:.1%}")
    print(f"最良パラメータ: {json.dumps(best, indent=2)}")
    print(f"保存先: {PARAMS_PATH}")


# ── バックテスト ──────────────────────────────────────────

def do_backtest(df_hist: pd.DataFrame) -> None:
    df_kw = df_hist[df_hist["venue"] == VENUE].copy()
    years = sorted(df_kw["race_date"].dt.year.unique())
    if len(years) < 2:
        print("バックテストには最低2年分の川崎データが必要です")
        return

    print(f"\n=== LeaveOneYearOut バックテスト ({VENUE}) ===")
    results = []
    for test_year in years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test  = df_kw[df_kw["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue

        stats = compute_stats(df_train)
        X_tr, y_tr, groups_tr = build_train_data(df_train)
        if X_tr.empty:
            print(f"  {test_year}: 学習データ不足 → スキップ")
            continue

        model = train_model(X_tr, y_tr, groups_tr)
        top5_t = top3_t = top1_t = cnt = 0

        for race_id, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3:
                continue
            Xr = build_features(race_df.reset_index(drop=True), stats,
                                 race_df["race_date"].iloc[0])
            scores, _ = predict_race(model, Xr)
            actual_top3 = set(race_df.index[race_df["finish_position"] <= 3])
            pred_sorted = race_df.index[np.argsort(-scores)]
            top5_t += len(actual_top3 & set(pred_sorted[:5])) / min(3, len(actual_top3))
            top3_t += len(actual_top3 & set(pred_sorted[:3])) / min(3, len(actual_top3))
            top1_t += int(pred_sorted[0] in actual_top3)
            cnt += 1

        if cnt == 0:
            continue
        r = {"year": test_year, "races": cnt,
             "top5_coverage": top5_t / cnt,
             "top3_hit":      top3_t / cnt,
             "top1_acc":      top1_t / cnt}
        results.append(r)
        print(f"  {test_year}: {cnt}R  "
              f"top5={r['top5_coverage']:.1%}  "
              f"top3={r['top3_hit']:.1%}  "
              f"top1={r['top1_acc']:.1%}")

    if results:
        print(f"\n平均 top5_coverage: {np.mean([r['top5_coverage'] for r in results]):.1%}")
        print(f"平均 top3_hit     : {np.mean([r['top3_hit']      for r in results]):.1%}")
        print(f"平均 top1_acc     : {np.mean([r['top1_acc']      for r in results]):.1%}")


# ── main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="川崎競馬予測 (LightGBM rank + SHAP根拠)")
    parser.add_argument("--date",     help="予測日 YYYY/MM/DD")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--tune",     action="store_true", help="Optunaチューニング実行")
    parser.add_argument("--trials",   type=int, default=100, help="Optuna試行回数 (default: 100)")
    parser.add_argument("--no-oi",   action="store_true", help="大井データを補助に使わない")
    args = parser.parse_args()

    df_hist = load_history(use_oi_supplement=not args.no_oi)

    if args.tune:
        do_tune(df_hist, n_trials=args.trials)
        return

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
