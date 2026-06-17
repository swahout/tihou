#!/usr/bin/env python3
"""川崎競馬 当日出走表収集

Usage:
    python collect_kawasaki_shutuba.py                    # 当日
    python collect_kawasaki_shutuba.py --date 2026/06/15  # 指定日

出力: data/races/kawasaki_{YYYY}_{MMDD}_shutuba.csv
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from scrapers.keibago import (
    KeibaGoSession, VENUE_MAP, get_race_list, get_deba_entries, get_tanfuku_odds,
)

VENUE = "川崎"
BABA_CODE = VENUE_MAP[VENUE]
OUT_DIR = Path("data/races")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "race_date", "venue", "race_no", "race_name",
    "distance", "surface", "direction", "track_cond", "weather",
    "field_size", "waku", "horse_no", "horse_name",
    "sex", "age", "weight_carried", "horse_weight", "weight_change",
    "jockey", "trainer", "win_odds",
]


def main():
    parser = argparse.ArgumentParser(description="川崎競馬 出走表収集")
    parser.add_argument("--date", help="対象日 YYYY/MM/DD（省略時=当日）")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y/%m/%d")
    session = KeibaGoSession(delay=1.2)

    print(f"{VENUE} {date_str} 出走表取得中...")
    races = get_race_list(session, date_str, BABA_CODE)
    if not races:
        print("開催なし または 取得失敗")
        return

    rows = []
    for race in races:
        rno = race["race_no"]
        entries = get_deba_entries(session, date_str, BABA_CODE, rno)
        if not entries:
            continue
        field_size = len(entries)

        # OddsTanFuku で単勝オッズを補完（DebaTable はレース前は空のことが多い）。
        # 市場人気ブレンドの入力になる。当日〜直近のみ取得可。
        odds_map = {}
        for o in get_tanfuku_odds(session, date_str, BABA_CODE, rno):
            if o.get("win_odds") is not None:
                odds_map[o["horse_no"]] = o["win_odds"]

        for h in entries:
            if h.get("win_odds") is None and h["horse_no"] in odds_map:
                h["win_odds"] = odds_map[h["horse_no"]]
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
            })

    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fname = f"kawasaki_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    out_path = OUT_DIR / fname
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(races)}レース {len(rows)}頭 → {out_path}")


if __name__ == "__main__":
    main()
