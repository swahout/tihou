#!/usr/bin/env python3
"""任意の地方競馬場の当日出走表を収集（汎用版）。

Usage:
    python collect_shutuba.py --venue 名古屋 --date 2026/06/18
    python collect_shutuba.py --venue 門別            # 当日

出力: data/races/{venue_en}_{YYYY_MMDD}_shutuba.csv
OddsTanFuku で単勝オッズも補完（市場ブレンド入力）。
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from scrapers.keibago import (
    KeibaGoSession, VENUE_MAP, get_race_list, get_deba_entries, get_tanfuku_odds,
)

VENUE_EN = {
    "川崎": "kawasaki", "大井": "oi", "船橋": "funabashi", "浦和": "urawa",
    "門別": "monbetsu", "園田": "sonoda", "姫路": "himeji", "名古屋": "nagoya",
    "金沢": "kanazawa", "笠松": "kasamatsu", "高知": "kochi", "佐賀": "saga",
}

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
    p = argparse.ArgumentParser(description="地方競馬 出走表収集（汎用）")
    p.add_argument("--venue", required=True, help="競馬場名（例: 名古屋）")
    p.add_argument("--date", help="対象日 YYYY/MM/DD（省略時=当日）")
    args = p.parse_args()

    venue = args.venue
    if venue not in VENUE_MAP:
        raise SystemExit(f"未知の競馬場: {venue}  選択肢: {list(VENUE_MAP)}")
    baba_code = VENUE_MAP[venue]
    venue_en = VENUE_EN.get(venue, venue)
    date_str = args.date or datetime.now().strftime("%Y/%m/%d")
    session = KeibaGoSession(delay=1.2)

    print(f"{venue} {date_str} 出走表取得中...")
    races = get_race_list(session, date_str, baba_code)
    if not races:
        print("開催なし または 取得失敗")
        return

    rows = []
    for race in races:
        rno = race["race_no"]
        entries = get_deba_entries(session, date_str, baba_code, rno)
        if not entries:
            continue
        field_size = len(entries)
        odds_map = {}
        for o in get_tanfuku_odds(session, date_str, baba_code, rno):
            if o.get("win_odds") is not None:
                odds_map[o["horse_no"]] = o["win_odds"]
        for h in entries:
            if h.get("win_odds") is None and h["horse_no"] in odds_map:
                h["win_odds"] = odds_map[h["horse_no"]]
            rows.append({
                "race_date": date_str, "venue": venue, "race_no": rno,
                "race_name": race["race_name"], "distance": race["distance"],
                "surface": race["surface"], "direction": race["direction"],
                "track_cond": race["track_cond"], "weather": race["weather"],
                "field_size": field_size, "waku": h["waku"], "horse_no": h["horse_no"],
                "horse_name": h["horse_name"], "sex": h["sex"], "age": h["age"],
                "weight_carried": h["weight_carried"], "horse_weight": h["horse_weight"],
                "weight_change": h["weight_change"], "jockey": h["jockey"],
                "trainer": h["trainer"], "win_odds": h["win_odds"],
            })

    dt = datetime.strptime(date_str, "%Y/%m/%d")
    out_path = OUT_DIR / f"{venue_en}_{dt.strftime('%Y_%m%d')}_shutuba.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    odds_n = sum(1 for r in rows if r["win_odds"] is not None)
    print(f"{len(races)}レース {len(rows)}頭 (オッズ{odds_n}頭) → {out_path}")


if __name__ == "__main__":
    main()
