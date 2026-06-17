#!/usr/bin/env python3
"""川崎競馬 予測スクリプト（tihouモデル不使用のクリーン実装）

Usage:
    python kawasaki_predict.py --date 2026/06/15
    python kawasaki_predict.py --backtest
    python kawasaki_predict.py --tune --trials 100

特徴量 (33種):
    - 当場成績率・通算成績率（Bayesian平滑化、K値もOptunaで最適化）[v3]
    - 当場当距離3着内率 / 馬場状態別3着内率
    - 直近フォーム / 当場直近フォーム / フォームトレンド
    - 騎手実績・騎手×距離帯実績・調教師実績（当場）[v3: 騎手距離帯追加]
    - 直接対決スコア（同一フィールドでの過去勝率）[v3追加]
    - テン乗りフラグ / 昇降級フラグ
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
KAWASAKI_DIR   = Path("data/historical_kawasaki")
OI_DIR         = Path("data/historical_oi")
FUNABASHI_DIR  = Path("data/historical_funabashi")
URAWA_DIR      = Path("data/historical_urawa")
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
    # 当場×当距離成績
    "top3_rate_venue_dist",
    "n_venue_dist",
    # 通算成績
    "top3_rate_total",
    "win_rate_total",
    "n_total",
    # 馬場状態別成績
    "top3_rate_cond",
    "n_cond",
    # 直近フォーム
    "recent_avg_pos",
    "recent_top3",
    "recent_venue_avg_pos",
    # フォームトレンド: 正=改善（着順下落）、負=悪化
    "form_trend",
    # 速度指数・上がり・脚質
    "avg_speed_idx",
    "best_speed_idx",
    "avg_last3f_idx",
    "avg_corner_ratio",
    # 騎手
    "jockey_top3_rate",
    "jockey_win_rate",
    # 騎手×距離帯 [v3]
    "jockey_top3_rate_dist",
    # 調教師
    "trainer_top3_rate",
    # 直接対決スコア [v3]
    "h2h_score",
    # テン乗り
    "is_ten_nori",
    # 昇降級フラグ: 正=昇級(harder)、負=降級(easier)、0=同クラス
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
    "top3_rate_venue":       "川崎3着内率",
    "win_rate_venue":        "川崎勝率",
    "n_venue":               "川崎出走数",
    "top3_rate_venue_dist":  "川崎同距3着内率",
    "n_venue_dist":          "川崎同距出走数",
    "top3_rate_total":       "通算3着内率",
    "win_rate_total":        "通算勝率",
    "n_total":               "通算出走数",
    "top3_rate_cond":        "馬場状態3着内率",
    "n_cond":                "同馬場出走数",
    "recent_avg_pos":        "直近着順",
    "recent_top3":           "直近3着内率",
    "recent_venue_avg_pos":  "川崎直近着順",
    "form_trend":            "フォーム改善",
    "avg_speed_idx":         "速度指数",
    "best_speed_idx":        "最高速度",
    "avg_last3f_idx":        "上がり3F",
    "avg_corner_ratio":      "脚質(通過順)",
    "jockey_top3_rate":      "騎手実績",
    "jockey_win_rate":       "騎手勝率",
    "jockey_top3_rate_dist": "騎手距離帯実績",
    "trainer_top3_rate":     "調教師実績",
    "h2h_score":             "直接対決",
    "is_ten_nori":           "テン乗り",
    "class_change":          "昇降級",
    "race_class_enc":        "レースクラス",
    "distance":              "距離",
    "field_size":            "頭数",
    "age":                   "年齢",
    "weight_carried":        "斤量",
    "sex_enc":               "性別",
    "umaban":                "馬番",
    "days_since_last":       "休養日数",
    "data_reliability":      "データ量",
}

SEX_MAP = {"牡": 0, "牝": 1, "セン": 2, "": 0}

# 距離帯分類（騎手×距離帯特徴量用）
def _dist_band(dist) -> str:
    d = int(dist) if dist else 0
    if d <= 1000: return "sprint"
    if d <= 1600: return "mile"
    return "long"


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

def load_history(use_oi_supplement: bool = True,
                 use_nankan_supplement: bool = True) -> pd.DataFrame:
    dfs = []
    # 川崎(当場)は常にロード。他場は補助データとしてフラグで切替え可能
    # （アブレーション用: --no-oi / --no-nankan で寄与を検証する）
    _venue_dirs = [(KAWASAKI_DIR, "kawasaki_*.csv")]
    if use_oi_supplement:
        _venue_dirs.append((OI_DIR, "oi_*.csv"))
    if use_nankan_supplement:
        _venue_dirs.append((FUNABASHI_DIR, "funabashi_*.csv"))
        _venue_dirs.append((URAWA_DIR,     "urawa_*.csv"))
    print("補助データ: "
          f"大井={'あり' if use_oi_supplement else 'なし'} "
          f"南関(船橋・浦和)={'あり' if use_nankan_supplement else 'なし'}")
    for src_dir, pattern in _venue_dirs:
        if src_dir.exists():
            for f in sorted(src_dir.glob(pattern)):
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
    # BOM混入対策: utf-8-sigでの追記(resume)時に ﻿ が race_date 先頭へ紛れ込み
    # to_datetime が ValueError で落ちることがある。除去してから変換する。
    df["race_date"]        = df["race_date"].astype(str).str.replace("﻿", "", regex=False).str.strip()
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
    """
    ベクトル化による高速版。生カウントを返すため K値を変えても再計算不要。
    [v3] 騎手×距離帯カウント・H2H対決スコアを追加。
    """
    df = df_hist.copy()
    df["_is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["_is_win"]  = (df["finish_position"] == 1).astype(int)

    global_top3_prior = df["_is_top3"].mean()
    global_win_prior  = df["_is_win"].mean()

    # ── 通算生カウント ──
    total_n    = df.groupby("horse_name").size()
    total_top3 = df.groupby("horse_name")["_is_top3"].sum()
    total_win  = df.groupby("horse_name")["_is_win"].sum()

    # ── 当場生カウント ──
    df_venue = df[df["venue"] == VENUE].copy()
    j_top3_prior = df_venue["_is_top3"].mean() if len(df_venue) > 0 else global_top3_prior
    j_win_prior  = df_venue["_is_win"].mean()  if len(df_venue) > 0 else global_win_prior

    venue_n    = df_venue.groupby("horse_name").size()
    venue_top3 = df_venue.groupby("horse_name")["_is_top3"].sum()
    venue_win  = df_venue.groupby("horse_name")["_is_win"].sum()

    # ── 直近フォーム ──
    df_s = df.sort_values(["horse_name", "race_date"])
    df_s["_rev_rank"] = df_s.groupby("horse_name").cumcount(ascending=False)
    recent = df_s[df_s["_rev_rank"] < 5].copy()
    recent_avg       = recent.groupby("horse_name")["finish_position"].mean()
    recent_top3_rate = recent.groupby("horse_name")["_is_top3"].mean()

    df_vs = df_venue.sort_values(["horse_name", "race_date"])
    df_vs["_rev_rank"] = df_vs.groupby("horse_name").cumcount(ascending=False)
    recent_venue_avg = df_vs[df_vs["_rev_rank"] < 5].groupby("horse_name")["finish_position"].mean()

    last_date = df_s.groupby("horse_name")["race_date"].max()

    # ── フォームトレンド: (3走前着順 - 最新着順) → 正=改善 ──
    pos_latest = df_s[df_s["_rev_rank"] == 0].set_index("horse_name")["finish_position"]
    pos_3rd    = df_s[df_s["_rev_rank"] == 2].set_index("horse_name")["finish_position"]
    form_trend_s = (pos_3rd - pos_latest)

    # ── 前走クラス（昇降級用）──
    df_s["_class_enc"] = df_s["race_name"].apply(_extract_class)
    last_class_s = df_s[df_s["_rev_rank"] == 0].set_index("horse_name")["_class_enc"]

    # ── 速度指数 ──
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

    # ── 当場×当距離 生カウント ──
    vd_n    = df_venue.groupby(["horse_name", "distance"]).size()
    vd_top3 = df_venue.groupby(["horse_name", "distance"])["_is_top3"].sum()
    vd_dict: dict = {}
    for (hname, dist), cnt in vd_n.items():
        if hname not in vd_dict: vd_dict[hname] = {}
        vd_dict[hname][dist] = {"n": int(cnt), "top3": int(vd_top3.get((hname, dist), 0))}

    # ── 馬場状態別 生カウント ──
    df_cond = df[df["track_cond"].notna() & (df["track_cond"] != "")].copy()
    cond_n    = df_cond.groupby(["horse_name", "track_cond"]).size()
    cond_top3 = df_cond.groupby(["horse_name", "track_cond"])["_is_top3"].sum()
    cond_dict: dict = {}
    for (hname, cond), cnt in cond_n.items():
        if hname not in cond_dict: cond_dict[hname] = {}
        cond_dict[hname][cond] = {"n": int(cnt), "top3": int(cond_top3.get((hname, cond), 0))}

    # ── 調教師 生カウント（当場）──
    t_n    = df_venue.groupby("trainer").size()
    t_top3 = df_venue.groupby("trainer")["_is_top3"].sum()
    trainer_raw: dict = {}
    for tr in t_n.index:
        trainer_raw[tr] = {"n": int(t_n[tr]), "top3": int(t_top3.get(tr, 0))}

    # ── 騎手 生カウント（当場）──
    j_n    = df_venue.groupby("jockey").size()
    j_top3 = df_venue.groupby("jockey")["_is_top3"].sum()
    j_win  = df_venue.groupby("jockey")["_is_win"].sum()

    # ── 騎手×距離帯 生カウント [v3] ──
    df_venue["_dist_band"] = df_venue["distance"].apply(_dist_band)
    jd_n    = df_venue.groupby(["jockey", "_dist_band"]).size()
    jd_top3 = df_venue.groupby(["jockey", "_dist_band"])["_is_top3"].sum()
    jd_dict: dict = {}
    for (jname, band), cnt in jd_n.items():
        if jname not in jd_dict: jd_dict[jname] = {}
        jd_dict[jname][band] = {"n": int(cnt), "top3": int(jd_top3.get((jname, band), 0))}

    horse_jockey_pairs = set(zip(df["horse_name"], df["jockey"]))

    # ── H2H（直接対決）ベクトル化 [v3] ──
    # 川崎データのみ使用（関係性が明確・データ量削減）
    race_key_v = (df_venue["race_date"].dt.strftime("%Y%m%d") + "_"
                  + df_venue["race_no"].astype(str))
    df_h = df_venue[["horse_name", "finish_position"]].copy()
    df_h["_rk"] = race_key_v.values
    # 自己結合でペアを作る
    h2h_pairs = df_h.merge(df_h, on="_rk", suffixes=("_a", "_b"))
    h2h_pairs = h2h_pairs[h2h_pairs["horse_name_a"] != h2h_pairs["horse_name_b"]].copy()
    h2h_pairs["_ahead"] = (h2h_pairs["finish_position_a"] < h2h_pairs["finish_position_b"]).astype(int)
    h2h_agg = (h2h_pairs.groupby(["horse_name_a", "horse_name_b"])
               .agg(together=("_ahead", "count"), ahead=("_ahead", "sum"))
               .reset_index())
    # iterrowsを避けて高速構築
    h2h_dict: dict = {}
    for (a, b, tog, ahd) in zip(
        h2h_agg["horse_name_a"], h2h_agg["horse_name_b"],
        h2h_agg["together"],     h2h_agg["ahead"]
    ):
        if a not in h2h_dict: h2h_dict[a] = {}
        h2h_dict[a][b] = {
            "ahead_rate": ahd / tog if tog > 0 else 0.5,
            "together":   int(tog),
        }

    # ── horse_stats 組み立て（生カウントで保存）──
    all_horses = set(total_n.index) | set(venue_n.index)
    horse_stats: dict = {}
    for name in all_horses:
        nt = int(total_n.get(name, 0))
        nv = int(venue_n.get(name, 0))
        horse_stats[name] = {
            # 生カウント（K値はbuild_featuresで適用）
            "n_total":       nt,
            "top3_total":    int(total_top3.get(name, 0)),
            "win_total":     int(total_win.get(name, 0)),
            "n_venue":       nv,
            "top3_venue":    int(venue_top3.get(name, 0)),
            "win_venue":     int(venue_win.get(name, 0)),
            # K非依存フィールド
            "recent_avg_pos":       recent_avg.get(name, np.nan),
            "recent_top3":          recent_top3_rate.get(name, np.nan),
            "recent_venue_avg_pos": recent_venue_avg.get(name, np.nan),
            "last_race_date":       last_date.get(name, pd.NaT),
            "avg_speed_idx":        avg_speed_idx.get(name, np.nan),
            "best_speed_idx":       best_speed_idx.get(name, np.nan),
            "avg_last3f_idx":       avg_last3f_idx.get(name, np.nan),
            "avg_corner_ratio":     avg_corner.get(name, np.nan),
            "vd_stats":             vd_dict.get(name, {}),
            "cond_stats":           cond_dict.get(name, {}),
            "form_trend":           form_trend_s.get(name, np.nan),
            "last_class":           last_class_s.get(name, np.nan),
        }

    jockey_raw: dict = {}
    for j in j_n.index:
        jockey_raw[j] = {
            "n": int(j_n[j]),
            "top3": int(j_top3.get(j, 0)),
            "win":  int(j_win.get(j, 0)),
            "dist_stats": jd_dict.get(j, {}),
        }

    return {
        "horse":              horse_stats,
        "jockey":             jockey_raw,
        "trainer":            trainer_raw,
        "h2h":                h2h_dict,
        "horse_jockey_pairs": horse_jockey_pairs,
        "priors": {
            "top3":   global_top3_prior,
            "win":    global_win_prior,
            "j_top3": j_top3_prior,
            "j_win":  j_win_prior,
        },
    }


# ── 特徴量構築 ─────────────────────────────────────────────

def build_features(df_race: pd.DataFrame, stats: dict, pred_date: pd.Timestamp,
                   k_horse: int = K_HORSE, k_jockey: int = K_JOCKEY) -> pd.DataFrame:
    """
    K値をパラメータで受け取り、生カウントからBayesian平滑化レートを計算する。
    [v3] h2h_score（直接対決）・jockey_top3_rate_dist（騎手×距離帯）追加。
    """
    priors      = stats["priors"]
    pairs       = stats["horse_jockey_pairs"]
    h2h_dict    = stats.get("h2h", {})
    trainer_raw = stats.get("trainer", {})

    # 全フィールド馬名リスト（H2H計算用）
    field_names = df_race["horse_name"].tolist()

    rows = []
    for _, h in df_race.iterrows():
        name    = h["horse_name"]
        jockey  = h.get("jockey", "")
        trainer = h.get("trainer", "")
        hs = stats["horse"].get(name, {})
        jr = stats["jockey"].get(jockey, {})
        tr = trainer_raw.get(trainer, {})

        last_dt  = hs.get("last_race_date", pd.NaT)
        days     = (pred_date - last_dt).days if pd.notna(last_dt) else 999
        n_total  = hs.get("n_total", 0)
        n_venue  = hs.get("n_venue", 0)
        fs       = h.get("field_size", 8) or 8
        distance = h.get("distance", 1400)
        track_cond = str(h.get("track_cond", "") or "")
        race_cls   = _extract_class(h.get("race_name", ""))

        def spd(key, default=1.0):
            v = hs.get(key, np.nan)
            return v if pd.notna(v) else default

        is_ten_nori = 0 if (name, jockey) in pairs else 1

        # ── 馬 Bayesian レート（K値適用）──
        top3_rate_total = _bayes(hs.get("top3_total", 0), n_total, priors["top3"],   k_horse)
        win_rate_total  = _bayes(hs.get("win_total",  0), n_total, priors["win"],    k_horse)
        top3_rate_venue = _bayes(hs.get("top3_venue", 0), n_venue, priors["j_top3"], k_horse)
        win_rate_venue  = _bayes(hs.get("win_venue",  0), n_venue, priors["j_win"],  k_horse)

        # 当場×当距離
        vd      = hs.get("vd_stats", {}).get(distance, {})
        n_vd    = vd.get("n", 0)
        top3_vd = _bayes(vd.get("top3", 0), n_vd, priors["j_top3"], k_horse)

        # 馬場状態別
        cd      = hs.get("cond_stats", {}).get(track_cond, {})
        n_cd    = cd.get("n", 0)
        top3_cd = _bayes(cd.get("top3", 0), n_cd, priors["top3"], k_horse)

        # フォームトレンド
        ft_raw     = hs.get("form_trend", np.nan)
        form_trend = float(ft_raw) if pd.notna(ft_raw) else 0.0

        # 昇降級
        last_cls_raw = hs.get("last_class", np.nan)
        last_cls     = int(last_cls_raw) if pd.notna(last_cls_raw) else race_cls
        class_change = race_cls - last_cls

        # ── 騎手 Bayesian レート（K値適用）──
        j_n    = jr.get("n", 0)
        j_top3_rate = _bayes(jr.get("top3", 0), j_n, priors["j_top3"], k_jockey)
        j_win_rate  = _bayes(jr.get("win",  0), j_n, priors["j_win"],  k_jockey)

        # 騎手×距離帯 [v3]
        dist_band = _dist_band(distance)
        jd = jr.get("dist_stats", {}).get(dist_band, {})
        j_dist_top3 = _bayes(jd.get("top3", 0), jd.get("n", 0), priors["j_top3"], k_jockey)

        # ── 調教師 Bayesian レート（K値適用）──
        t_n = tr.get("n", 0)
        trainer_top3 = _bayes(tr.get("top3", 0), t_n, priors["j_top3"], k_jockey)

        # ── H2H スコア [v3] ──
        # 現フィールドの対戦相手との過去対決勝率の平均（≥2戦のみカウント）
        opponents = [x for x in field_names if x != name]
        h2h_scores = []
        for opp in opponents:
            pair = h2h_dict.get(name, {}).get(opp)
            if pair and pair["together"] >= 2:
                h2h_scores.append(pair["ahead_rate"])
        h2h_score = float(np.mean(h2h_scores)) if h2h_scores else 0.5

        row = {
            "horse_name":            name,
            "top3_rate_venue":       top3_rate_venue,
            "win_rate_venue":        win_rate_venue,
            "n_venue":               n_venue,
            "top3_rate_venue_dist":  top3_vd,
            "n_venue_dist":          n_vd,
            "top3_rate_total":       top3_rate_total,
            "win_rate_total":        win_rate_total,
            "n_total":               n_total,
            "top3_rate_cond":        top3_cd,
            "n_cond":                n_cd,
            "recent_avg_pos":        hs.get("recent_avg_pos",        fs * 0.6),
            "recent_top3":           hs.get("recent_top3",           priors["top3"]),
            "recent_venue_avg_pos":  hs.get("recent_venue_avg_pos",  fs * 0.6),
            "form_trend":            form_trend,
            "avg_speed_idx":         spd("avg_speed_idx"),
            "best_speed_idx":        spd("best_speed_idx"),
            "avg_last3f_idx":        spd("avg_last3f_idx"),
            "avg_corner_ratio":      spd("avg_corner_ratio", default=0.5),
            "jockey_top3_rate":      j_top3_rate,
            "jockey_win_rate":       j_win_rate,
            "jockey_top3_rate_dist": j_dist_top3,
            "trainer_top3_rate":     trainer_top3,
            "h2h_score":             h2h_score,
            "is_ten_nori":           is_ten_nori,
            "class_change":          class_change,
            "race_class_enc":        race_cls,
            "distance":              distance,
            "field_size":            fs,
            "age":                   h.get("age", 0) or 0,
            "weight_carried":        h.get("weight_carried", 55.0) or 55.0,
            "sex_enc":               SEX_MAP.get(str(h.get("sex", "")), 0),
            "umaban":                h.get("horse_no", 0),
            "days_since_last":       min(days, 999),
            "data_reliability":      n_total / (n_total + DATA_K),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── 学習データ構築 ─────────────────────────────────────────

def build_train_data(df_hist: pd.DataFrame,
                     k_horse: int = K_HORSE,
                     k_jockey: int = K_JOCKEY) -> tuple:
    """年次ローリングウィンドウで学習データを構築。K値を受け取りbuild_featuresに渡す。"""
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
            Xr = build_features(race_df.reset_index(drop=True), stats, pred_date,
                                 k_horse=k_horse, k_jockey=k_jockey)
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


def _load_params() -> tuple[dict, int, int, int]:
    """保存済みパラメータを読み込む。なければデフォルト値。
    Returns: (lgb_params, num_rounds, k_horse, k_jockey)
    """
    import json
    if PARAMS_PATH.exists():
        saved = json.loads(PARAMS_PATH.read_text())
        num_rounds = int(saved.pop("num_boost_round", _DEFAULT_ROUNDS))
        k_horse    = int(saved.pop("k_horse",  K_HORSE))
        k_jockey   = int(saved.pop("k_jockey", K_JOCKEY))
        params = {**_DEFAULT_PARAMS, **saved}
        return params, num_rounds, k_horse, k_jockey
    return dict(_DEFAULT_PARAMS), _DEFAULT_ROUNDS, K_HORSE, K_JOCKEY


def train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                params: dict | None = None, num_rounds: int | None = None) -> lgb.Booster:
    if params is None:
        params, num_rounds, _, _ = _load_params()
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

    _, _, k_horse, k_jockey = _load_params()
    print(f"  K_HORSE={k_horse}  K_JOCKEY={k_jockey}")

    print("統計計算中...")
    stats = compute_stats(df_train_hist)

    print("学習データ構築中...")
    X, y, groups = build_train_data(df_train_hist, k_horse=k_horse, k_jockey=k_jockey)
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
        Xr      = build_features(race_df, stats, pred_date, k_horse=k_horse, k_jockey=k_jockey)
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

def _build_race_rows_from_stats(
    df_year: pd.DataFrame, stats: dict, pred_date: pd.Timestamp,
    k_horse: int, k_jockey: int
) -> tuple[list, list, list]:
    """statsを受け取りK値適用済み特徴量を返す（チューニング高速化用）"""
    X_list, y_list, g_list = [], [], []
    for race_id, race_df in df_year.groupby(
        df_year["race_date"].dt.strftime("%Y%m%d") + "_" + df_year["race_no"].astype(str)
    ):
        if len(race_df) < 3:
            continue
        Xr = build_features(race_df.reset_index(drop=True), stats, pred_date,
                             k_horse=k_horse, k_jockey=k_jockey)
        X_list.append(Xr)
        y_list.append(race_df["finish_position"].values)
        g_list.extend([race_id] * len(race_df))
    return X_list, y_list, g_list


def do_tune(df_hist: pd.DataFrame, n_trials: int = 100,
            metric: str = "top5", study_name: str | None = None) -> None:
    import json
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # metric="top5" → 予測上位5での3着内カバレッジ最大化（v6）
    # metric="top3" → 予測上位3での3着内的中(top3_hit)最大化（v7、目標指標に直接一致）
    cutoff = 3 if metric == "top3" else 5
    if study_name is None:
        study_name = (f"kawasaki_multiyr_8R_top3_v7" if metric == "top3"
                      else "kawasaki_multiyr_8R_top5_v6")

    df_kw = df_hist[df_hist["venue"] == VENUE].copy()
    all_years = sorted(df_kw["race_date"].dt.year.unique())
    # 過学習対策: 単一年(2026)ではなく複数の held-out 年で評価する。
    # 各 test_year は「その年より前の全データ」で学習する LeaveOneYearOut 構造。
    test_years = [y for y in all_years if y in (2024, 2025, 2026)]
    if not test_years:
        print("チューニングに必要なデータ（2024〜2026年）がありません")
        return

    print(f"CV対象年: {test_years}")
    print("Raw statsを事前計算中（K値に依存しない部分のみ）...")

    # K値非依存の stats を年ごとに事前計算
    precomp: list[dict] = []
    for test_year in test_years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test  = df_kw[df_kw["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue
        # 学習データの年ごとにも stats が必要（build_train_data の内部と同じ構造）
        df_tr = df_train.copy()
        df_tr["_year"] = df_tr["race_date"].dt.year
        tr_years = sorted(df_tr["_year"].unique())
        tr_stats_by_year = {}
        for i, yr in enumerate(tr_years):
            if i < 1: continue
            df_before = df_tr[df_tr["_year"] < yr].copy()
            if df_before.empty: continue
            tr_stats_by_year[yr] = (compute_stats(df_before), df_tr[df_tr["_year"] == yr])

        test_stats = compute_stats(df_train)
        test_race_groups = []
        for _, race_df in df_test.groupby(
            df_test["race_date"].dt.strftime("%Y%m%d") + "_" + df_test["race_no"].astype(str)
        ):
            if len(race_df) < 3: continue
            rdf = race_df.reset_index(drop=True)          # index を 0,1,2... に統一
            actual_top3 = set(rdf.index[rdf["finish_position"] <= 3])
            race_no     = int(rdf["race_no"].iloc[0])
            race_date   = rdf["race_date"].iloc[0]
            test_race_groups.append((rdf, actual_top3, race_no, race_date))

        precomp.append({
            "test_year":       test_year,
            "tr_stats":        tr_stats_by_year,   # {year: (stats, df_year)}
            "df_train":        df_train,
            "test_stats":      test_stats,
            "test_races":      test_race_groups,
        })
        print(f"  {test_year}: テスト{len(test_race_groups)}R (stats計算済み)")

    if not precomp:
        print("有効なCVデータなし")
        return

    def objective(trial: optuna.Trial) -> float:
        k_horse  = trial.suggest_int("k_horse",  2, 25)
        k_jockey = trial.suggest_int("k_jockey", 8, 60)
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
        num_rounds = trial.suggest_int("num_boost_round", 50, 300)

        total_top5 = total_cnt = 0
        for pc in precomp:
            # 学習データ構築（precomputed stats + K値適用）
            all_X, all_y, all_g = [], [], []
            for yr, (stats_yr, df_yr) in pc["tr_stats"].items():
                Xl, yl, gl = _build_race_rows_from_stats(
                    df_yr, stats_yr, pd.Timestamp(f"{yr}-01-01"), k_horse, k_jockey)
                all_X.extend(Xl); all_y.extend(yl); all_g.extend(gl)
            if not all_X:
                continue
            X_tr = pd.concat(all_X, ignore_index=True)
            y_tr = np.concatenate(all_y)
            g_tr = np.array(all_g)

            model = train_model(X_tr, y_tr, g_tr, params=params, num_rounds=num_rounds)

            for race_df, actual_top3, race_no, race_date in pc["test_races"]:
                if race_no < 8:
                    continue
                Xr = build_features(race_df, pc["test_stats"], race_date,
                                    k_horse=k_horse, k_jockey=k_jockey)
                scores, _ = predict_race(model, Xr)
                pred_sorted = race_df.index[np.argsort(-scores)]
                total_top5 += len(actual_top3 & set(pred_sorted[:cutoff])) / min(3, len(actual_top3))
                total_cnt  += 1

        return total_top5 / total_cnt if total_cnt > 0 else 0.0

    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_DB}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    metric_label = "top3_hit(上位3)" if metric == "top3" else "top5_coverage(上位5)"
    print(f"\nOptuna チューニング開始 ({n_trials}試行, {test_years}の8R以降 {metric_label} 最大化)")
    print(f"  DB: {OPTUNA_DB}  スタディ: {study.study_name}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**study.best_params,
            "num_boost_round": study.best_params.get("num_boost_round", _DEFAULT_ROUNDS)}
    # v6(top5)は本番 PARAMS_PATH に保存。v7(top3)等は別ファイルに保存して本番を壊さない
    # （held-out forward で勝ったら手動/明示的に昇格する方針）
    out_path = PARAMS_PATH if metric == "top5" else PARAMS_PATH.with_name(f"kawasaki_best_params_{metric}.json")
    out_path.write_text(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\n最良スコア (8R以降 {metric_label}): {study.best_value:.1%}")
    print(f"最良パラメータ: {json.dumps(best, indent=2)}")
    print(f"保存先: {out_path}")


# ── バックテスト ──────────────────────────────────────────

def do_backtest(df_hist: pd.DataFrame) -> None:
    _, _, k_horse, k_jockey = _load_params()
    df_kw = df_hist[df_hist["venue"] == VENUE].copy()
    years = sorted(df_kw["race_date"].dt.year.unique())
    if len(years) < 2:
        print("バックテストには最低2年分の川崎データが必要です")
        return

    print(f"\n=== LeaveOneYearOut バックテスト ({VENUE}) ===")
    print(f"  K_HORSE={k_horse}  K_JOCKEY={k_jockey}")
    results = []
    for test_year in years:
        df_train = df_hist[df_hist["race_date"].dt.year < test_year].copy()
        df_test  = df_kw[df_kw["race_date"].dt.year == test_year].copy()
        if df_train.empty or df_test.empty:
            continue

        stats = compute_stats(df_train)
        X_tr, y_tr, groups_tr = build_train_data(df_train, k_horse=k_horse, k_jockey=k_jockey)
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
                                 race_df["race_date"].iloc[0],
                                 k_horse=k_horse, k_jockey=k_jockey)
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
    parser.add_argument("--metric",   choices=["top5", "top3"], default="top5",
                        help="チューニング最適化指標。top5=v6(top5_coverage), top3=v7(top3_hit直接最適化)")
    parser.add_argument("--no-oi",     action="store_true", help="大井データを補助に使わない")
    parser.add_argument("--no-nankan", action="store_true", help="船橋・浦和データを補助に使わない（アブレーション用）")
    args = parser.parse_args()

    df_hist = load_history(use_oi_supplement=not args.no_oi,
                           use_nankan_supplement=not args.no_nankan)

    if args.tune:
        do_tune(df_hist, n_trials=args.trials, metric=args.metric)
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
