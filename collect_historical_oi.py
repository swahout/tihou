#!/usr/bin/env python3
"""大井競馬 過去レースデータ収集

Usage:
    python collect_historical_oi.py --years 2023 2024 2025
    python collect_historical_oi.py --date 2025/06/01

出力: data/historical_oi/oi_{year}.csv
各行: 1頭 × 1レースの生データ（特徴量はモデル学習時に計算）
"""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

from scrapers.keibago import KeibaGoSession, VENUE_MAP, get_race_list, get_race_entries, get_race_results

VENUE = "大井"
BABA_CODE = VENUE_MAP[VENUE]
OUT_DIR = Path("data/historical_oi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "race_date", "venue", "race_no", "race_name",
    "distance", "surface", "direction", "track_cond", "weather",
    "field_size", "waku", "horse_no", "horse_name",
    "sex", "age", "weight_carried", "horse_weight", "weight_change",
    "jockey", "trainer", "win_odds",
    "finish_position", "finish_time",
]


def collect_one_date(session: KeibaGoSession, date_str: str) -> list[dict]:
    races = get_race_list(session, date_str, BABA_CODE)
    if not races:
        return []

    rows = []
    for race in races:
        rno = race["race_no"]
        entries = get_race_entries(session, date_str, BABA_CODE, rno)
        results = get_race_results(session, date_str, BABA_CODE, rno)

        if not entries:
            continue

        result_map = {r["horse_no"]: r for r in results}
        field_size = len(entries)

        for h in entries:
            res = result_map.get(h["horse_no"], {})
            rows.append({
                "race_date": date_str,
                "venue": VENUE,
                "race_no": rno,
                "race_name": race["race_name"],
                "distance": race["distance"],
                "surface": race["surface"],
                "direction": race["direction"],
                "track_cond": race["track_cond"],
                "weather": race["weather"],
                "field_size": field_size,
                "waku": h["waku"],
                "horse_no": h["horse_no"],
                "horse_name": h["horse_name"],
                "sex": h["sex"],
                "age": h["age"],
                "weight_carried": h["weight_carried"],
                "horse_weight": h["horse_weight"],
                "weight_change": h["weight_change"],
                "jockey": h["jockey"],
                "trainer": h["trainer"],
                "win_odds": h["win_odds"],
                "finish_position": res.get("finish_position", ""),
                "finish_time": res.get("finish_time", ""),
            })
    return rows


def collect_year(year: int, session: KeibaGoSession) -> int:
    """指定年の大井競馬データを収集してCSVに追記。収集行数を返す。"""
    out_path = OUT_DIR / f"oi_{year}.csv"

    existing_dates: set[str] = set()
    if out_path.exists():
        import pandas as pd
        try:
            df_ex = pd.read_csv(out_path, encoding="utf-8-sig", usecols=["race_date"])
            existing_dates = set(df_ex["race_date"].dropna().unique())
            print(f"  既存: {len(existing_dates)}日分")
        except Exception:
            pass

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    if end > date.today() - timedelta(days=1):
        end = date.today() - timedelta(days=1)

    all_rows: list[dict] = []
    d = start
    while d <= end:
        date_str = d.strftime("%Y/%m/%d")
        if date_str not in existing_dates:
            rows = collect_one_date(session, date_str)
            if rows:
                all_rows.extend(rows)
                print(f"  {date_str}: {len(rows)}行", flush=True)
            # else: 開催なし（出力しない）
        d += timedelta(days=1)

    if not all_rows:
        print(f"  {year}年: 新規データなし")
        return 0

    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"  → 保存: {out_path} (+{len(all_rows)}行)")
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description="大井競馬 過去データ収集")
    parser.add_argument(
        "--years", type=int, nargs="+",
        help="収集年（複数指定可） e.g. --years 2023 2024 2025"
    )
    parser.add_argument(
        "--date", help="特定日のみ収集 e.g. --date 2025/06/01"
    )
    args = parser.parse_args()

    session = KeibaGoSession(delay=1.2)

    if args.date:
        print(f"{VENUE} {args.date} データ収集")
        rows = collect_one_date(session, args.date)
        if rows:
            year = int(args.date[:4])
            out_path = OUT_DIR / f"oi_{year}.csv"
            mode = "a" if out_path.exists() else "w"
            with open(out_path, mode, encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if mode == "w":
                    writer.writeheader()
                writer.writerows(rows)
            print(f"  {len(rows)}行保存: {out_path}")
        else:
            print("  開催なしまたは取得失敗")
        return

    years = args.years or list(range(date.today().year - 2, date.today().year))
    for year in years:
        print(f"\n{VENUE} {year}年 収集開始")
        collect_year(year, session)


if __name__ == "__main__":
    main()
