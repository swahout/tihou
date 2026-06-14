#!/usr/bin/env python3
"""川崎競馬 過去レースデータ収集

Usage:
    python collect_historical_kawasaki.py --years 2022 2023 2024 2025
    python collect_historical_kawasaki.py --date 2025/06/01

出力: data/historical_kawasaki/kawasaki_{year}.csv
"""
import argparse
import csv
from datetime import date
from pathlib import Path

from scrapers.keibago import (
    KeibaGoSession, VENUE_MAP,
    get_race_dates_for_venue, get_race_list,
    get_race_data,
)

VENUE = "川崎"
BABA_CODE = VENUE_MAP[VENUE]
OUT_DIR = Path("data/historical_kawasaki")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "race_date", "venue", "race_no", "race_name",
    "distance", "surface", "direction", "track_cond", "weather",
    "field_size", "waku", "horse_no", "horse_name",
    "sex", "age", "weight_carried", "horse_weight", "weight_change",
    "jockey", "trainer", "win_odds",
    "finish_position", "finish_time", "last_3f", "passage_rate",
    "popularity", "result_weight", "result_weight_change",
]


def collect_one_date(session: KeibaGoSession, date_str: str) -> list[dict]:
    races = get_race_list(session, date_str, BABA_CODE)
    if not races:
        return []

    rows = []
    for race in races:
        rno = race["race_no"]
        horses = get_race_data(session, date_str, BABA_CODE, rno)

        if not horses:
            continue

        field_size = len(horses)
        for h in horses:
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
                "finish_position": h.get("finish_position", ""),
                "finish_time": h.get("finish_time", ""),
                "last_3f": h.get("last_3f", ""),
                "passage_rate": h.get("passage_rate", ""),
                "popularity": h.get("popularity", ""),
                "result_weight": h.get("result_weight", ""),
                "result_weight_change": h.get("result_weight_change", ""),
            })
    return rows


def collect_year(year: int, session: KeibaGoSession) -> int:
    """指定年の川崎競馬データを収集してCSVに追記。収集行数を返す。"""
    out_path = OUT_DIR / f"kawasaki_{year}.csv"

    existing_dates: set[str] = set()
    if out_path.exists():
        import pandas as pd
        try:
            df_ex = pd.read_csv(out_path, encoding="utf-8-sig", usecols=["race_date"])
            existing_dates = set(df_ex["race_date"].dropna().unique())
            print(f"  既存: {len(existing_dates)}日分")
        except Exception:
            pass

    print(f"  開催日程取得中 (MonthlyConveneInfo)...", flush=True)
    race_dates = get_race_dates_for_venue(session, year, BABA_CODE)
    today = date.today().strftime("%Y/%m/%d")
    race_dates = [d for d in race_dates if d < today]
    to_collect = [d for d in race_dates if d not in existing_dates]
    print(f"  開催日: {len(race_dates)}日 → 未収集: {len(to_collect)}日", flush=True)

    if not to_collect:
        print(f"  {year}年: 収集済み")
        return 0

    all_rows: list[dict] = []
    for i, date_str in enumerate(to_collect, 1):
        rows = collect_one_date(session, date_str)
        if rows:
            all_rows.extend(rows)
            print(f"  [{i}/{len(to_collect)}] {date_str}: {len(rows)}行", flush=True)
        else:
            print(f"  [{i}/{len(to_collect)}] {date_str}: 取得失敗", flush=True)

    if not all_rows:
        print(f"  {year}年: データなし")
        return 0

    mode = "a" if out_path.exists() and existing_dates else "w"
    with open(out_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"  → 保存: {out_path} (+{len(all_rows)}行)", flush=True)
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description="川崎競馬 過去データ収集")
    parser.add_argument(
        "--years", type=int, nargs="+",
        help="収集年（複数指定可） e.g. --years 2022 2023 2024 2025"
    )
    parser.add_argument(
        "--date", help="特定日のみ収集 e.g. --date 2025/06/01"
    )
    parser.add_argument("--delay", type=float, default=0.8, help="リクエスト間隔(秒) デフォルト0.8")
    args = parser.parse_args()

    session = KeibaGoSession(delay=args.delay)

    if args.date:
        print(f"{VENUE} {args.date} データ収集")
        rows = collect_one_date(session, args.date)
        if rows:
            year = int(args.date[:4])
            out_path = OUT_DIR / f"kawasaki_{year}.csv"
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

    years = args.years or list(range(date.today().year - 3, date.today().year))
    for year in sorted(years):
        print(f"\n{VENUE} {year}年 収集開始")
        collect_year(year, session)


if __name__ == "__main__":
    main()
