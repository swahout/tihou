"""特徴量エンジニアリング（地方競馬版）"""
import numpy as np
import pandas as pd
from typing import Optional

FEATURE_COLS = [
    "n_total", "n_venue", "n_dist", "n_venue_dist",
    "top3_rate_total", "top3_rate_venue", "top3_rate_dist",
    "top3_rate_venue_dist", "top3_rate_cond",
    "win_rate_total", "win_rate_venue",
    "recent_avg_pos", "recent_top3", "recent_win",
    "recent_avg_pos_venue",
    "jockey_top3_rate", "jockey_win_rate",
    "trainer_top3_rate",
    "days_since_last",
    "sex_enc", "age", "weight_carried", "weight_change", "horse_weight",
    "umaban", "waku", "field_size", "distance_f",
    "data_reliability",
]

SEX_MAP = {"牡": 0, "牝": 1, "騸": 2, "セ": 2}


def _bayes(num, den, prior, k):
    """ベイズ平滑化レート: (num + prior*k) / (den + k)"""
    return (num + prior * k) / (den + k)


def precompute_stats(df_history: pd.DataFrame, params: Optional[dict] = None) -> dict:
    """
    歴史データから全集計を一度だけ計算。
    結果はすべて dict (key → value) 形式で返すので per-race ループ内では .map() で高速ルックアップ可能。
    """
    dh = df_history.copy()
    dh["race_date"] = pd.to_datetime(dh["race_date"])
    dh["is_top3"] = (dh["finish_position"] <= 3).astype(float)
    dh["is_win"] = (dh["finish_position"] == 1).astype(float)

    # ── 馬別 全体 ──
    ht = dh.groupby("horse_name").agg(
        n_total=("finish_position", "count"),
        top3_total=("is_top3", "sum"),
        win_total=("is_win", "sum"),
    )

    # ── 馬別 場別 (venue, horse_name) ──
    hv = (
        dh.groupby(["venue", "horse_name"]).agg(
            n_venue=("finish_position", "count"),
            top3_venue=("is_top3", "sum"),
            win_venue=("is_win", "sum"),
        )
        if "venue" in dh.columns else pd.DataFrame()
    )

    # ── 馬別 距離別 ──
    hd = (
        dh.groupby(["distance", "horse_name"]).agg(
            n_dist=("finish_position", "count"),
            top3_dist=("is_top3", "sum"),
        )
        if "distance" in dh.columns else pd.DataFrame()
    )

    # ── 馬別 場×距離 ──
    hvd = (
        dh.groupby(["venue", "distance", "horse_name"]).agg(
            n_venue_dist=("finish_position", "count"),
            top3_venue_dist=("is_top3", "sum"),
        )
        if ("venue" in dh.columns and "distance" in dh.columns) else pd.DataFrame()
    )

    # ── 馬別 馬場状態 ──
    hc = (
        dh.groupby(["track_cond", "horse_name"]).agg(
            n_cond=("finish_position", "count"),
            top3_cond=("is_top3", "sum"),
        )
        if "track_cond" in dh.columns else pd.DataFrame()
    )

    # ── 近走（直近5走） ──
    dh_sorted = dh.sort_values("race_date")
    recent5 = dh_sorted.groupby("horse_name").tail(5)
    hr = recent5.groupby("horse_name").agg(
        recent_avg_pos=("finish_position", "mean"),
        recent_top3=("is_top3", "mean"),
        recent_win=("is_win", "mean"),
    )

    # ── 近走 当場 (venue → horse_name → recent_avg_pos_venue) ──
    hrv_by_venue: dict[str, pd.Series] = {}
    if "venue" in dh.columns:
        for venue_val, vdf in dh.groupby("venue"):
            rv5 = vdf.sort_values("race_date").groupby("horse_name").tail(5)
            hrv_by_venue[venue_val] = rv5.groupby("horse_name")["finish_position"].mean()

    # ── 最終走日 ──
    last_date = dh.groupby("horse_name")["race_date"].max()

    # ── 騎手・調教師（当場） ──
    jk_by_venue: dict[str, pd.DataFrame] = {}
    tr_by_venue: dict[str, pd.DataFrame] = {}
    if "venue" in dh.columns:
        for venue_val, vdf in dh.groupby("venue"):
            if "jockey" in vdf.columns:
                jk_by_venue[venue_val] = vdf.groupby("jockey").agg(
                    jockey_n=("finish_position", "count"),
                    jockey_top3=("is_top3", "sum"),
                    jockey_win=("is_win", "sum"),
                )
            if "trainer" in vdf.columns:
                tr_by_venue[venue_val] = vdf.groupby("trainer").agg(
                    trainer_n=("finish_position", "count"),
                    trainer_top3=("is_top3", "sum"),
                )
    # 場情報なし時はフォールバック
    if not jk_by_venue and "jockey" in dh.columns:
        jk_by_venue["_all"] = dh.groupby("jockey").agg(
            jockey_n=("finish_position", "count"),
            jockey_top3=("is_top3", "sum"),
            jockey_win=("is_win", "sum"),
        )
    if not tr_by_venue and "trainer" in dh.columns:
        tr_by_venue["_all"] = dh.groupby("trainer").agg(
            trainer_n=("finish_position", "count"),
            trainer_top3=("is_top3", "sum"),
        )

    # ── dict 形式に変換（map() 高速化のため） ──
    def to_col_dicts(df_agg: pd.DataFrame) -> dict[str, dict]:
        return {col: df_agg[col].to_dict() for col in df_agg.columns}

    # horse_name → stats
    horse_total_d = to_col_dicts(ht)
    horse_recent_d = to_col_dicts(hr)
    last_date_d = last_date.to_dict()

    # (venue, horse_name) → stats
    hv_d = to_col_dicts(hv) if not hv.empty else {}
    # (distance, horse_name) → stats
    hd_d = to_col_dicts(hd) if not hd.empty else {}
    # (venue, distance, horse_name) → stats
    hvd_d = to_col_dicts(hvd) if not hvd.empty else {}
    # (track_cond, horse_name) → stats
    hc_d = to_col_dicts(hc) if not hc.empty else {}

    # venue → horse_name → value
    hrv_d = {v: s.to_dict() for v, s in hrv_by_venue.items()}

    # venue → jockey/trainer → stats cols
    jk_d = {v: to_col_dicts(df) for v, df in jk_by_venue.items()}
    tr_d = {v: to_col_dicts(df) for v, df in tr_by_venue.items()}

    return {
        "horse_total": horse_total_d,
        "horse_venue": hv_d,
        "horse_dist": hd_d,
        "horse_vd": hvd_d,
        "horse_cond": hc_d,
        "horse_recent": horse_recent_d,
        "horse_recent_venue": hrv_d,
        "last_race_date": last_date_d,
        "jockey_venue": jk_d,
        "trainer_venue": tr_d,
    }


def build_features_precomputed(
    df_race: pd.DataFrame,
    stats: dict,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """precompute_stats() の結果を使った高速特徴量構築（dict.get() ルックアップ）"""
    p = params or {}
    K = p.get("bayesian_k", 8)
    K_j = p.get("bayesian_k_jockey", 30)

    field_size = len(df_race)
    base_rate = min(3.0 / field_size, 1.0) if field_size > 0 else 0.33

    venue = df_race["venue"].iloc[0] if "venue" in df_race.columns else None
    distance = df_race["distance"].iloc[0] if "distance" in df_race.columns else None
    track_cond = df_race["track_cond"].iloc[0] if "track_cond" in df_race.columns else None
    race_date = df_race["race_date"].iloc[0] if "race_date" in df_race.columns else None

    names = df_race["horse_name"].tolist()
    n = len(names)

    def lookup(d: dict, key) -> np.ndarray:
        return np.array([d.get(k, 0.0) for k in key], dtype=float)

    def lookup2(outer: dict, outer_key, col: str, keys) -> np.ndarray:
        inner = outer.get(col, {}) if outer_key is None else outer.get(col, {})
        return np.array([inner.get(k, 0.0) for k in keys], dtype=float)

    # ── 全体 ──
    ht = stats["horse_total"]
    n_total = lookup(ht.get("n_total", {}), names)
    top3_total = lookup(ht.get("top3_total", {}), names)
    win_total = lookup(ht.get("win_total", {}), names)

    # ── 場別 ──
    hv = stats["horse_venue"]
    if venue and hv:
        n_venue = np.array([hv.get("n_venue", {}).get((venue, nm), 0.0) for nm in names], dtype=float)
        top3_venue = np.array([hv.get("top3_venue", {}).get((venue, nm), 0.0) for nm in names], dtype=float)
        win_venue = np.array([hv.get("win_venue", {}).get((venue, nm), 0.0) for nm in names], dtype=float)
    else:
        n_venue = top3_venue = win_venue = np.zeros(n)

    # ── 距離別 ──
    hd = stats["horse_dist"]
    if distance and hd:
        n_dist = np.array([hd.get("n_dist", {}).get((distance, nm), 0.0) for nm in names], dtype=float)
        top3_dist = np.array([hd.get("top3_dist", {}).get((distance, nm), 0.0) for nm in names], dtype=float)
    else:
        n_dist = top3_dist = np.zeros(n)

    # ── 場×距離 ──
    hvd = stats["horse_vd"]
    if venue and distance and hvd:
        n_vd = np.array([hvd.get("n_venue_dist", {}).get((venue, distance, nm), 0.0) for nm in names], dtype=float)
        top3_vd = np.array([hvd.get("top3_venue_dist", {}).get((venue, distance, nm), 0.0) for nm in names], dtype=float)
    else:
        n_vd = top3_vd = np.zeros(n)

    # ── 馬場状態 ──
    hc = stats["horse_cond"]
    if track_cond and hc:
        n_cond = np.array([hc.get("n_cond", {}).get((track_cond, nm), 0.0) for nm in names], dtype=float)
        top3_cond = np.array([hc.get("top3_cond", {}).get((track_cond, nm), 0.0) for nm in names], dtype=float)
    else:
        n_cond = top3_cond = np.zeros(n)

    # ── 近走 ──
    hr = stats["horse_recent"]
    recent_avg_pos = np.array([hr.get("recent_avg_pos", {}).get(nm, field_size / 2) for nm in names], dtype=float)
    recent_top3 = np.array([hr.get("recent_top3", {}).get(nm, base_rate) for nm in names], dtype=float)
    recent_win = np.array([hr.get("recent_win", {}).get(nm, base_rate / 3) for nm in names], dtype=float)

    # ── 近走 当場 ──
    hrv_map = stats["horse_recent_venue"]
    hrv = hrv_map.get(venue, {}) if venue else {}
    recent_avg_pos_venue = np.array([hrv.get(nm, field_size / 2) for nm in names], dtype=float)

    # ── 最終走日 ──
    last_d = stats["last_race_date"]
    if race_date is not None:
        ref = pd.Timestamp(race_date)
        days_since = np.array([
            min(max((ref - pd.Timestamp(last_d[nm])).days, 0), 365) if nm in last_d else 60.0
            for nm in names
        ], dtype=float)
    else:
        days_since = np.full(n, 60.0)

    # ── 騎手 ──
    jk_map = stats["jockey_venue"]
    jk_key = venue if venue in jk_map else ("_all" if "_all" in jk_map else None)
    if jk_key and "jockey" in df_race.columns:
        jk = jk_map[jk_key]
        jockeys = df_race["jockey"].tolist()
        jockey_n = np.array([jk.get("jockey_n", {}).get(j, 0.0) for j in jockeys], dtype=float)
        jockey_top3 = np.array([jk.get("jockey_top3", {}).get(j, 0.0) for j in jockeys], dtype=float)
        jockey_win = np.array([jk.get("jockey_win", {}).get(j, 0.0) for j in jockeys], dtype=float)
    else:
        jockey_n = jockey_top3 = jockey_win = np.zeros(n)

    # ── 調教師 ──
    tr_map = stats["trainer_venue"]
    tr_key = venue if venue in tr_map else ("_all" if "_all" in tr_map else None)
    if tr_key and "trainer" in df_race.columns:
        tr = tr_map[tr_key]
        trainers = df_race["trainer"].tolist()
        trainer_n = np.array([tr.get("trainer_n", {}).get(t, 0.0) for t in trainers], dtype=float)
        trainer_top3 = np.array([tr.get("trainer_top3", {}).get(t, 0.0) for t in trainers], dtype=float)
    else:
        trainer_n = trainer_top3 = np.zeros(n)

    # ── Bayesian rates ──
    top3_rate_total = _bayes(top3_total, n_total, base_rate, K)
    top3_rate_venue = _bayes(top3_venue, n_venue, base_rate, K)
    top3_rate_dist = _bayes(top3_dist, n_dist, base_rate, K)
    top3_rate_vd = _bayes(top3_vd, n_vd, base_rate, K)
    top3_rate_cond = _bayes(top3_cond, n_cond, base_rate, K)
    win_rate_total = _bayes(win_total, n_total, base_rate / 3, K)
    win_rate_venue = _bayes(win_venue, n_venue, base_rate / 3, K)

    jockey_top3_rate = _bayes(jockey_top3, jockey_n, 0.3, K_j)
    jockey_win_rate = _bayes(jockey_win, jockey_n, 0.1, K_j)
    trainer_top3_rate = _bayes(trainer_top3, trainer_n, 0.3, K_j)

    # ── 物理系 ──
    def col_arr(col, default):
        if col in df_race.columns:
            return df_race[col].fillna(default).values.astype(float)
        return np.full(n, float(default))

    sex_enc = df_race.get("sex", pd.Series([""] * n)).map(SEX_MAP).fillna(0).values.astype(float)
    age = col_arr("age", 4)
    weight_carried = col_arr("weight_carried", 55.0)
    weight_change = col_arr("weight_change", 0)
    horse_weight = col_arr("horse_weight", 480)
    umaban = col_arr("horse_no", 0)
    waku = col_arr("waku", 0)
    data_reliability = n_total / (n_total + 5)

    out = pd.DataFrame({
        "n_total": n_total,
        "n_venue": n_venue,
        "n_dist": n_dist,
        "n_venue_dist": n_vd,
        "top3_rate_total": top3_rate_total,
        "top3_rate_venue": top3_rate_venue,
        "top3_rate_dist": top3_rate_dist,
        "top3_rate_venue_dist": top3_rate_vd,
        "top3_rate_cond": top3_rate_cond,
        "win_rate_total": win_rate_total,
        "win_rate_venue": win_rate_venue,
        "recent_avg_pos": recent_avg_pos,
        "recent_top3": recent_top3,
        "recent_win": recent_win,
        "recent_avg_pos_venue": recent_avg_pos_venue,
        "jockey_top3_rate": jockey_top3_rate,
        "jockey_win_rate": jockey_win_rate,
        "trainer_top3_rate": trainer_top3_rate,
        "days_since_last": days_since,
        "sex_enc": sex_enc,
        "age": age,
        "weight_carried": weight_carried,
        "weight_change": weight_change,
        "horse_weight": horse_weight,
        "umaban": umaban,
        "waku": waku,
        "field_size": float(field_size),
        "distance_f": float(distance) if distance else 0.0,
        "data_reliability": data_reliability,
    })
    return out[FEATURE_COLS]


def build_features(
    df_race: pd.DataFrame,
    df_history: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """予測パス用（シングルレース呼び出し）。precompute → build_precomputed のラッパー。"""
    if df_history.empty:
        field_size = len(df_race)
        base_rate = min(3.0 / field_size, 1.0) if field_size > 0 else 0.33
        distance = df_race["distance"].iloc[0] if "distance" in df_race.columns else None
        return _empty_features(df_race, base_rate, field_size, distance)

    stats = precompute_stats(df_history, params)
    return build_features_precomputed(df_race, stats, params)


def _empty_features(df_race, base_rate, field_size, distance):
    n = len(df_race)
    data = {col: [0.0] * n for col in FEATURE_COLS}
    df = pd.DataFrame(data)
    for col in ["top3_rate_total", "top3_rate_venue", "top3_rate_dist",
                "top3_rate_venue_dist", "top3_rate_cond"]:
        df[col] = base_rate
    df["win_rate_total"] = base_rate / 3
    df["win_rate_venue"] = base_rate / 3
    df["recent_avg_pos"] = field_size / 2
    df["recent_avg_pos_venue"] = field_size / 2
    df["jockey_top3_rate"] = 0.3
    df["jockey_win_rate"] = 0.1
    df["trainer_top3_rate"] = 0.3
    df["days_since_last"] = 60.0
    df["distance_f"] = float(distance) if distance else 0.0
    df["field_size"] = float(field_size)
    if "age" in df_race.columns:
        df["age"] = df_race["age"].fillna(4).values
    if "weight_carried" in df_race.columns:
        df["weight_carried"] = df_race["weight_carried"].fillna(55.0).values
    if "horse_no" in df_race.columns:
        df["umaban"] = df_race["horse_no"].fillna(0).values
    if "waku" in df_race.columns:
        df["waku"] = df_race["waku"].fillna(0).values
    return df
