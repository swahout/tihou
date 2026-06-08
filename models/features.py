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
    "win_odds", "data_reliability",
]

SEX_MAP = {"牡": 0, "牝": 1, "騸": 2, "セ": 2}


def _bayes(num, den, prior, k):
    """ベイズ平滑化レート: (num + prior*k) / (den + k)"""
    return (num + prior * k) / (den + k)


def build_features(
    df_race: pd.DataFrame,
    df_history: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    df_race   : 出走表（1行=1頭、結果列なし可）
    df_history: 学習に使う過去レース（df_race の年を除外済みであること）
    Returns   : FEATURE_COLS 列の DataFrame（index は df_race に対応）
    """
    p = params or {}
    K = p.get("bayesian_k", 8)
    K_j = p.get("bayesian_k_jockey", 30)

    field_size = len(df_race)
    base_rate = min(3.0 / field_size, 1.0) if field_size > 0 else 0.33

    venue = df_race["venue"].iloc[0] if "venue" in df_race.columns else None
    distance = df_race["distance"].iloc[0] if "distance" in df_race.columns else None
    track_cond = df_race["track_cond"].iloc[0] if "track_cond" in df_race.columns else None
    race_date = df_race["race_date"].iloc[0] if "race_date" in df_race.columns else None

    if df_history.empty:
        return _empty_features(df_race, base_rate, field_size, distance)

    dh = df_history.copy()
    dh["is_top3"] = (dh["finish_position"] <= 3).astype(float)
    dh["is_win"] = (dh["finish_position"] == 1).astype(float)

    # ── 馬別集計 ──
    ht = dh.groupby("horse_name").agg(
        n_total=("finish_position", "count"),
        top3_total=("is_top3", "sum"),
        win_total=("is_win", "sum"),
    )
    hv = (
        dh[dh["venue"] == venue].groupby("horse_name").agg(
            n_venue=("finish_position", "count"),
            top3_venue=("is_top3", "sum"),
            win_venue=("is_win", "sum"),
            avg_pos_venue=("finish_position", "mean"),
        ) if venue else pd.DataFrame()
    )
    hd = (
        dh[dh["distance"] == distance].groupby("horse_name").agg(
            n_dist=("finish_position", "count"),
            top3_dist=("is_top3", "sum"),
        ) if distance else pd.DataFrame()
    )
    hvd = (
        dh[(dh["venue"] == venue) & (dh["distance"] == distance)].groupby("horse_name").agg(
            n_venue_dist=("finish_position", "count"),
            top3_venue_dist=("is_top3", "sum"),
        ) if (venue and distance) else pd.DataFrame()
    )
    hc = (
        dh[dh["track_cond"] == track_cond].groupby("horse_name").agg(
            n_cond=("finish_position", "count"),
            top3_cond=("is_top3", "sum"),
        ) if track_cond else pd.DataFrame()
    )

    # 近走（直近5走）
    recent5 = dh.sort_values("race_date").groupby("horse_name").tail(5)
    hr = recent5.groupby("horse_name").agg(
        recent_avg_pos=("finish_position", "mean"),
        recent_top3=("is_top3", "mean"),
        recent_win=("is_win", "mean"),
    )
    recent_venue5 = (
        dh[dh["venue"] == venue].sort_values("race_date").groupby("horse_name").tail(5)
        if venue else pd.DataFrame()
    )
    hrv = (
        recent_venue5.groupby("horse_name").agg(
            recent_avg_pos_venue=("finish_position", "mean"),
        ) if not recent_venue5.empty else pd.DataFrame()
    )

    # 最終走日
    last_date = dh.groupby("horse_name")["race_date"].max().rename("last_race_date")

    # ── 騎手・調教師集計 ──
    dh_v = dh[dh["venue"] == venue] if venue else dh
    jk = dh_v.groupby("jockey").agg(
        jockey_n=("finish_position", "count"),
        jockey_top3=("is_top3", "sum"),
        jockey_win=("is_win", "sum"),
    ) if "jockey" in dh_v.columns else pd.DataFrame()
    tr = dh_v.groupby("trainer").agg(
        trainer_n=("finish_position", "count"),
        trainer_top3=("is_top3", "sum"),
    ) if "trainer" in dh_v.columns else pd.DataFrame()

    # ── マージ ──
    df = df_race.reset_index(drop=True).copy()
    for agg, key in [(ht, "horse_name"), (hv, "horse_name"), (hd, "horse_name"),
                     (hvd, "horse_name"), (hc, "horse_name"),
                     (hr, "horse_name"), (hrv, "horse_name")]:
        if not agg.empty:
            df = df.merge(agg, on=key, how="left")
    if not last_date.empty:
        df = df.merge(last_date, on="horse_name", how="left")
    if not jk.empty and "jockey" in df.columns:
        df = df.merge(jk, on="jockey", how="left")
    if not tr.empty and "trainer" in df.columns:
        df = df.merge(tr, on="trainer", how="left")

    # ── 数値化 ──
    Z = lambda col: df.get(col, pd.Series([0]*len(df), dtype=float)).fillna(0)

    df["n_total"] = Z("n_total")
    df["n_venue"] = Z("n_venue")
    df["n_dist"] = Z("n_dist")
    df["n_venue_dist"] = Z("n_venue_dist")

    df["top3_rate_total"] = _bayes(Z("top3_total"), Z("n_total"), base_rate, K)
    df["top3_rate_venue"] = _bayes(Z("top3_venue"), Z("n_venue"), base_rate, K)
    df["top3_rate_dist"] = _bayes(Z("top3_dist"), Z("n_dist"), base_rate, K)
    df["top3_rate_venue_dist"] = _bayes(Z("top3_venue_dist"), Z("n_venue_dist"), base_rate, K)
    df["top3_rate_cond"] = _bayes(Z("top3_cond"), Z("n_cond"), base_rate, K)
    df["win_rate_total"] = _bayes(Z("win_total"), Z("n_total"), base_rate / 3, K)
    df["win_rate_venue"] = _bayes(Z("win_venue"), Z("n_venue"), base_rate / 3, K)

    df["recent_avg_pos"] = df.get("recent_avg_pos", pd.Series([float(field_size)/2]*len(df))).fillna(float(field_size) / 2)
    df["recent_top3"] = df.get("recent_top3", pd.Series([base_rate]*len(df))).fillna(base_rate)
    df["recent_win"] = df.get("recent_win", pd.Series([base_rate/3]*len(df))).fillna(base_rate / 3)
    df["recent_avg_pos_venue"] = df.get("recent_avg_pos_venue", pd.Series([float(field_size)/2]*len(df))).fillna(float(field_size) / 2)

    df["jockey_top3_rate"] = _bayes(Z("jockey_top3"), Z("jockey_n"), 0.3, K_j)
    df["jockey_win_rate"] = _bayes(Z("jockey_win"), Z("jockey_n"), 0.1, K_j)
    df["trainer_top3_rate"] = _bayes(Z("trainer_top3"), Z("trainer_n"), 0.3, K_j)

    if "last_race_date" in df.columns and race_date is not None:
        ref = pd.to_datetime(race_date)
        df["days_since_last"] = (ref - pd.to_datetime(df["last_race_date"])).dt.days.fillna(60).clip(0, 365)
    else:
        df["days_since_last"] = 60.0

    df["sex_enc"] = df.get("sex", pd.Series([""] * len(df))).map(SEX_MAP).fillna(0)
    df["age"] = df.get("age", pd.Series([4]*len(df))).fillna(4)
    df["weight_carried"] = df.get("weight_carried", pd.Series([55.0]*len(df))).fillna(55.0)
    df["weight_change"] = df.get("weight_change", pd.Series([0]*len(df))).fillna(0)
    df["horse_weight"] = df.get("horse_weight", pd.Series([480]*len(df))).fillna(480)
    df["umaban"] = df.get("horse_no", pd.Series([0]*len(df))).fillna(0)
    df["waku"] = df.get("waku", pd.Series([0]*len(df))).fillna(0)
    df["field_size"] = float(field_size)
    df["distance_f"] = float(distance) if distance else 0.0
    df["win_odds"] = df.get("win_odds", pd.Series([30.0]*len(df))).fillna(30.0)
    df["data_reliability"] = df["n_total"] / (df["n_total"] + 5)

    return df[FEATURE_COLS]


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
    df["win_odds"] = 30.0
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
