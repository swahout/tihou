#!/usr/bin/env python3
"""南関東・各地方競馬場 過去レースデータ収集（汎用）

Usage:
    python collect_historical_nankan.py --venue 船橋 --years 2022 2023 2024 2025 2026
    python collect_historical_nankan.py --venue 浦和 --years 2022 2023 2024 2025 2026
    python collect_historical_nankan.py --venue 船橋 浦和 --years 2022 2023 2024 2025 2026

出力: data/historical_{venue_key}/{venue_key}_{year}.csv
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

VENUE_KEY_MAP = {
    "大井": "oi",
    "川崎": "kawasaki",
    "船橋": "funabashi",
    "浦和": "urawa",
    "門別": "monbetsu",
    "園田": "sonoda",
    "名古屋": "nagoya",
    "金沢": "kanazawa",
    "笠松": "kasamatsu",
    "高知": "kochi",
    "佐賀": "saga",
}

FIELDNAMES = [
    "race_date", "venue", "race_no", "race_name",
    "distance", "surface", "direction", "track_cond", "weather",
    "field_size", "waku", "horse_no", "horse_name",
    "sex", "age", "weight_carried", "horse_weight", "weight_change",
    "jockey", "trainer", "win_odds",
    "finish_position", "finish_time", "last_3f", "passage_rate",
    "popularity", "result_weight", "result_weight_change",
]


def collect_one_date(session: KeibaGoSession, date_str: str, venue: str, baba_code: int) -> list[dict]:
    races = get_race_list(session, date_str, baba_code)
    if not races:
        return []

    rows = []
    for race in races:
        rno = race["race_no"]
        horses = get_race_data(session, date_str, baba_code, rno)
        if not horses:
            continue

        field_size = len(horses)
        for h in horses:
            rows.append({
                "race_date": date_str,
                "venue": venue,
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


def collect_venue_year(venue: str, year: int, session: KeibaGoSession) -> int:
    baba_code = VENUE_MAP[venue]
    venue_key = VENUE_KEY_MAP[venue]
    out_dir = Path(f"data/historical_{venue_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{venue_key}_{year}.csv"

    existing_dates: set[str] = set()
    if out_path.exists():
        import pandas as pd
        try:
            df_ex = pd.read_csv(out_path, encoding="utf-8-sig", usecols=["race_date"])
            existing_dates = set(df_ex["race_date"].dropna().unique())
            print(f"  既存: {len(existing_dates)}日分")
        except Exception:
            pass

    print(f"  開催日程取得中...", flush=True)
    race_dates = get_race_dates_for_venue(session, year, baba_code)
    today = date.today().strftime("%Y/%m/%d")
    race_dates = [d for d in race_dates if d < today]
    to_collect = [d for d in race_dates if d not in existing_dates]
    print(f"  開催日: {len(race_dates)}日 → 未収集: {len(to_collect)}日", flush=True)

    if not to_collect:
        print(f"  {year}年: 収集済み")
        return 0

    all_rows: list[dict] = []
    for i, date_str in enumerate(to_collect, 1):
        rows = collect_one_date(session, date_str, venue, baba_code)
        if rows:
            all_rows.extend(rows)
            print(f"  [{i}/{len(to_collect)}] {date_str}: {len(rows)}行", flush=True)
        else:
            print(f"  [{i}/{len(to_collect)}] {date_str}: 取得失敗", flush=True)

    if not all_rows:
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
    parser = argparse.ArgumentParser(description="地方競馬 過去データ収集（汎用）")
    parser.add_argument("--venue", nargs="+", required=True,
                        choices=list(VENUE_KEY_MAP.keys()),
                        help="競馬場名（複数指定可）e.g. --venue 船橋 浦和")
    parser.add_argument("--years", type=int, nargs="+",
                        help="収集年 e.g. --years 2022 2023 2024 2025 2026")
    parser.add_argument("--date", help="特定日のみ e.g. --date 2025/06/01")
    parser.add_argument("--delay", type=float, default=1.2, help="リクエスト間隔(秒)")
    args = parser.parse_args()

    session = KeibaGoSession(delay=args.delay)

    if args.date:
        for venue in args.venue:
            baba_code = VENUE_MAP[venue]
            venue_key = VENUE_KEY_MAP[venue]
            print(f"{venue} {args.date} データ収集")
            rows = collect_one_date(session, args.date, venue, baba_code)
            if rows:
                year = int(args.date[:4])
                out_dir = Path(f"data/historical_{venue_key}")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{venue_key}_{year}.csv"
                mode = "a" if out_path.exists() else "w"
                with open(out_path, mode, encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    if mode == "w":
                        writer.writeheader()
                    writer.writerows(rows)
                print(f"  {len(rows)}行保存: {out_path}")
            else:
                print("  開催なし or 取得失敗")
        return

    years = args.years or list(range(2022, date.today().year + 1))
    for venue in args.venue:
        for year in sorted(years):
            print(f"\n{venue} {year}年 収集開始")
            collect_venue_year(venue, year, session)


if __name__ == "__main__":
    main()
