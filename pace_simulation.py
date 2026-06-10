#!/usr/bin/env python3
"""大井競馬 ペースシミュレーション

passage_rate（コーナー通過順）から各馬の脚質を判定し、
フィールド内の逃げ・先行馬数からペースを予測して確率を補正する。

Usage:
    python pace_simulation.py --date 2026/06/10
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

HIST_DIR = Path("data/historical_oi")
RACES_DIR = Path("data/races")

# ペース × 脚質 補正係数
PACE_MULTIPLIER = {
    "high":   {"逃げ": 0.65, "先行": 0.85, "差し": 1.25, "追い込み": 1.40},
    "medium": {"逃げ": 0.95, "先行": 1.15, "差し": 1.00, "追い込み": 0.90},
    "slow":   {"逃げ": 1.45, "先行": 1.20, "差し": 0.85, "追い込み": 0.65},
}

# 脚質判定しきい値（field_sizeで正規化した比率で判定）
# avg_pos / field_size の値で分類
STYLE_RATIO = {
    "逃げ":    (0.00, 0.18),
    "先行":    (0.18, 0.42),
    "差し":    (0.42, 0.72),
    "追い込み": (0.72, 1.01),
}


def parse_passage(passage_rate_str) -> list[int]:
    """'3-7-2' → [3, 7, 2]"""
    if not isinstance(passage_rate_str, str) or passage_rate_str.strip() in ("", "**"):
        return []
    nums = re.findall(r"\d+", passage_rate_str)
    return [int(n) for n in nums]


def get_running_style(horse_name: str, history: pd.DataFrame) -> dict:
    """直近10走の通過順（指数加重平均）から脚質を判定"""
    rows = (
        history[history["horse_name"] == horse_name]
        .dropna(subset=["passage_rate", "field_size"])
        .sort_values("race_date", ascending=False)
        .head(10)
    )
    if rows.empty:
        return {"style": "不明", "avg_pos": None, "avg_ratio": None, "n": 0}

    ratios, weights = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        nums = parse_passage(str(row["passage_rate"]))
        if not nums:
            continue
        last_corner = nums[-1]  # 最終コーナー通過順
        fs = max(int(row["field_size"]), 1)
        ratio = last_corner / fs
        ratios.append(ratio)
        weights.append(0.8 ** i)

    if not ratios:
        return {"style": "不明", "avg_pos": None, "avg_ratio": None, "n": 0}

    w = np.array(weights)
    r = np.array(ratios)
    avg_ratio = float(np.dot(w, r) / w.sum())

    style = "差し"
    for s, (lo, hi) in STYLE_RATIO.items():
        if lo <= avg_ratio < hi:
            style = s
            break

    # avg_posは表示用（field_size=10想定の絶対値に換算）
    avg_pos = avg_ratio * 10

    return {
        "style": style,
        "avg_pos": round(avg_pos, 2),
        "avg_ratio": round(avg_ratio, 3),
        "n": len(ratios),
    }


def predict_pace(styles: pd.Series) -> tuple[str, str]:
    nige   = (styles == "逃げ").sum()
    senkou = (styles == "先行").sum()

    if nige >= 2:
        pace = "high"
        reason = f"逃げ{nige}頭競合 → ハイ"
    elif nige == 1:
        pace = "slow"
        reason = f"逃げ1頭のみ → 単騎逃げ・スロー"
    elif senkou >= 4:
        pace = "medium"
        reason = f"逃げなし先行{senkou}頭 → ミドル"
    else:
        pace = "slow"
        reason = f"逃げなし先行{senkou}頭 → スロー"

    return pace, reason


def simulate_race(race_no: int, race_df: pd.DataFrame,
                  history: pd.DataFrame) -> pd.DataFrame | None:
    """1レース分のペース補正を行い結果DataFrameを返す"""
    if len(race_df) < 2:
        return None

    # 脚質判定
    style_rows = []
    for _, h in race_df.iterrows():
        info = get_running_style(h["horse_name"], history)
        style_rows.append({
            "horse_name": h["horse_name"],
            **info,
        })
    styles_df = pd.DataFrame(style_rows)

    pace, pace_reason = predict_pace(styles_df["style"])
    multipliers = PACE_MULTIPLIER[pace]

    merged = race_df.copy().reset_index(drop=True)
    merged = merged.merge(
        styles_df[["horse_name", "style", "avg_pos", "avg_ratio", "n"]],
        on="horse_name", how="left"
    )
    merged["style"] = merged["style"].fillna("不明")
    merged["pace"] = pace
    merged["pace_reason"] = pace_reason
    merged["pace_mult"] = merged["style"].map(multipliers).fillna(1.0)
    merged["top3_prob_pace"] = merged["top3_prob"] * merged["pace_mult"]

    # 正規化（合計を補正前と同じに保つ）
    total_before = merged["top3_prob"].sum()
    total_after  = merged["top3_prob_pace"].sum()
    if total_after > 0:
        merged["top3_prob_pace"] *= total_before / total_after

    merged = merged.sort_values("top3_prob_pace", ascending=False).reset_index(drop=True)
    merged["pace_rank"] = range(1, len(merged) + 1)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026/06/10")
    parser.add_argument("--pred-suffix", default="6mo_prediction",
                        help="予測CSVのサフィックス（デフォルト: 6mo_prediction）")
    args = parser.parse_args()

    dt = datetime.strptime(args.date, "%Y/%m/%d")
    date_tag = dt.strftime("%Y_%m%d")

    pred_path = RACES_DIR / f"oi_{date_tag}_{args.pred_suffix}.csv"
    if not pred_path.exists():
        print(f"予測ファイルが見つかりません: {pred_path}")
        return

    df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    print(f"予測ファイル: {pred_path} ({len(df_pred)}頭)")

    # 履歴データ読み込み
    csvs = sorted(HIST_DIR.glob("oi_*.csv"))
    dfs = []
    for csv in csvs:
        try:
            dfs.append(pd.read_csv(csv, encoding="utf-8-sig"))
        except Exception:
            pass
    if not dfs:
        print("履歴データが見つかりません")
        return
    history = pd.concat(dfs, ignore_index=True)
    history["race_date"] = pd.to_datetime(history["race_date"])
    pred_date = pd.Timestamp(dt)
    history = history[history["race_date"] < pred_date].copy()
    print(f"履歴データ: {len(history)}行（{pred_date.date()}より前）\n")

    all_results = []
    for race_no, race_df in df_pred.groupby("race_no"):
        result = simulate_race(race_no, race_df, history)
        if result is None:
            continue
        all_results.append(result)

    if not all_results:
        print("シミュレーション失敗")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    out_path = RACES_DIR / f"oi_{date_tag}_pace_prediction.csv"
    df_all.to_csv(out_path, encoding="utf-8-sig", index=False)

    # 表示
    print(f"\n{'='*72}")
    print(f"  大井競馬 {args.date} ペースシミュレーション補正後予測")
    print(f"{'='*72}")

    for race_no, grp in df_all.groupby("race_no"):
        meta = grp.iloc[0]
        pace_label = {"high": "ハイ", "medium": "ミドル", "slow": "スロー"}[meta["pace"]]
        print(f"\n【{race_no:2d}R】 {meta['race_name']}  {meta['distance']}m  "
              f"[{pace_label}] {meta['pace_reason']}")

        # 脚質分布
        style_counts = grp["style"].value_counts()
        style_summary = "  ".join(
            f"{s}:{style_counts[s]}頭"
            for s in ["逃げ", "先行", "差し", "追い込み", "不明"]
            if s in style_counts
        )
        print(f"         脚質: {style_summary}")

        top5 = grp.head(5)
        rows = []
        for _, h in top5.iterrows():
            # ML予測での順位を求める
            ml_rank = int(df_pred[(df_pred["race_no"] == race_no)]
                          .sort_values("top3_prob", ascending=False)
                          .reset_index(drop=True)
                          .index[df_pred[(df_pred["race_no"] == race_no)]
                                 .sort_values("top3_prob", ascending=False)
                                 ["horse_name"].reset_index(drop=True) == h["horse_name"]][0]) + 1
            change = ml_rank - int(h["pace_rank"])
            chg = f"↑{change}" if change > 0 else (f"↓{abs(change)}" if change < 0 else "─")
            rel = h.get("data_reliability", 0)
            warn = "⚠" if rel < 0.3 else ""
            rows.append([
                int(h["pace_rank"]),
                int(h["horse_no"]),
                h["horse_name"],
                f"{h['top3_prob_pace']:.1%}",
                h["style"],
                f"{h['pace_mult']:.2f}x",
                f"({h['top3_prob']:.1%})",
                chg,
                warn,
            ])
        print(tabulate(rows,
                       headers=["順","馬番","馬名","補正後確率","脚質","倍率","ML確率","変動",""],
                       tablefmt="simple"))

    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
