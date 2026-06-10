#!/usr/bin/env python3
"""南関東・主要地方競馬 過去レースデータ収集（大井以外）

Usage:
    python collect_historical_others.py --years 2025 2026
    python collect_historical_others.py --years 2025 2026 --venues 船橋 川崎 浦和

出力: data/historical_others/{venue_en}_{year}.csv
"""
import argparse
import csv
from datetime import date
from pathlib import Path

from scrapers.keibago import (
    KeibaGoSession, VENUE_MAP,
    get_race_dates_for_venue, get_race_list, get_race_data,
)

VENUE_EN = {
    "船橋": "funabashi", "川崎": "kawasaki", "浦和": "urawa",
    "門別": "monbetsu", "園田": "sonoda", "姫路": "himeji",
    "名古屋": "nagoya", "金沢": "kanazawa", "笠松": "kasamatsu",
    "高知": "kochi", "佐賀": "saga",
}

OUT_DIR = Path("data/historical_others")
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

DEFAULT_VENUES = ["船橋", "川崎", "浦和"]


def collect_year_venue(session: KeibaGoSession, year: int, venue: str) -> int:
    baba_code = VENUE_MAP[venue]
    venue_en = VENUE_EN.get(venue, venue)
    out_path = OUT_DIR / f"{venue_en}_{year}.csv"

    existing_dates: set[str] = set()
    if out_path.exists():
        import pandas as pd
        try:
            df_ex = pd.read_csv(out_path, encoding="utf-8-sig", usecols=["race_date"])
            existing_dates = set(df_ex["race_date"].dropna().unique())
            print(f"  [{venue}] 既存: {len(existing_dates)}日分")
        except Exception:
            pass

    print(f"  [{venue}] 開催日程取得中...", flush=True)
    race_dates = get_race_dates_for_venue(session, year, baba_code)
    today = date.today().strftime("%Y/%m/%d")
    race_dates = [d for d in race_dates if d < today]
    to_collect = [d for d in race_dates if d not in existing_dates]
    print(f"  [{venue}] 開催日: {len(race_dates)}日 → 未収集: {len(to_collect)}日", flush=True)

    if not to_collect:
        print(f"  [{venue}] {year}年: 収集済み")
        return 0

    all_rows: list[dict] = []
    for i, date_str in enumerate(to_collect, 1):
        races = get_race_list(session, date_str, baba_code)
        if not races:
            print(f"  [{venue}] [{i}/{len(to_collect)}] {date_str}: 開催なし", flush=True)
            continue
        day_rows = []
        for race in races:
            horses = get_race_data(session, date_str, baba_code, race["race_no"])
            if not horses:
                continue
            for h in horses:
                day_rows.append({
                    "race_date": date_str,
                    "venue": venue,
                    "race_no": race["race_no"],
                    "race_name": race["race_name"],
                    "distance": race["distance"],
                    "surface": race["surface"],
                    "direction": race["direction"],
                    "track_cond": race["track_cond"],
                    "weather": race["weather"],
                    "field_size": len(horses),
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
        if day_rows:
            all_rows.extend(day_rows)
            print(f"  [{venue}] [{i}/{len(to_collect)}] {date_str}: {len(day_rows)}行", flush=True)
        else:
            print(f"  [{venue}] [{i}/{len(to_collect)}] {date_str}: データなし", flush=True)

    if not all_rows:
        print(f"  [{venue}] {year}年: データなし")
        return 0

    mode = "a" if out_path.exists() and existing_dates else "w"
    with open(out_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"  [{venue}] → 保存: {out_path} (+{len(all_rows)}行)", flush=True)
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description="地方競馬（大井以外）過去データ収集")
    parser.add_argument("--years", type=int, nargs="+", required=True,
                        help="収集年 e.g. --years 2025 2026")
    parser.add_argument("--venues", nargs="+", default=DEFAULT_VENUES,
                        help=f"収集馬場 デフォルト: {DEFAULT_VENUES}")
    parser.add_argument("--delay", type=float, default=1.2)
    args = parser.parse_args()

    session = KeibaGoSession(delay=args.delay)
    total = 0
    for venue in args.venues:
        if venue not in VENUE_MAP:
            print(f"未知の馬場: {venue}  スキップ")
            continue
        for year in sorted(args.years):
            print(f"\n{venue} {year}年 収集開始")
            total += collect_year_venue(session, year, venue)

    print(f"\n合計 {total}行 収集完了")


if __name__ == "__main__":
    main()
