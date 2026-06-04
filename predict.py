#!/usr/bin/env python3
"""名古屋競馬1日予想 - 3着内確率を計算してレース毎に上位5頭を選出"""

import argparse
import re
import sys
import math
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

BASE_URL = "https://www.keiba.go.jp"
BABA_CODE = 24  # 名古屋
BABA_NAME = "名古屋"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.2  # seconds between requests (礼儀として間隔を空ける)


# ─────────────────────────────────────────────
# HTTP utility
# ─────────────────────────────────────────────

def fetch(url: str) -> BeautifulSoup:
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "lxml")


# ─────────────────────────────────────────────
# レース一覧取得
# ─────────────────────────────────────────────

def get_race_list(date_str: str) -> list[dict]:
    """当日の名古屋競馬レース一覧を返す。date_str = 'YYYY/MM/DD'"""
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceList"
        f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}"
    )
    soup = fetch(url)

    table = soup.find("table")
    if not table:
        return []

    races = []
    for row in table.find_all("tr")[2:]:  # 先頭2行はヘッダー
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        links = [a.get("href", "") for a in row.find_all("a")]
        if len(cells) < 6:
            continue

        m = re.match(r"(\d+)R", cells[0])
        if not m:
            continue

        races.append({
            "race_no": int(m.group(1)),
            "time": cells[1],
            "race_type": cells[3],
            "race_name": cells[4],
            "course": cells[5],
            "weather": cells[6] if len(cells) > 6 else "",
            "track_cond": cells[7] if len(cells) > 7 else "",
            "deba_link": next((l for l in links if "DebaTable" in l), ""),
            "odds_link": next((l for l in links if "OddsTanFuku" in l), ""),
        })

    return races


# ─────────────────────────────────────────────
# 着別成績パース
# ─────────────────────────────────────────────

def _parse_record_row(cells: list[str]) -> tuple[int, int, int, int]:
    """['1-', '1-', '2-', '3'] -> (1, 1, 2, 3)"""
    try:
        return (
            int(cells[1].rstrip("-")),
            int(cells[2].rstrip("-")),
            int(cells[3].rstrip("-")),
            int(cells[4]),
        )
    except (ValueError, IndexError):
        return (0, 0, 0, 0)


def parse_arrival_table(table) -> dict[str, tuple]:
    """class='arrival noBorder' テーブルから {全/左/右/場/距: (1着,2着,3着,着外)} を返す"""
    result = {}
    for row in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) >= 5 and cells[0] in ("全", "左", "右", "場", "距"):
            result[cells[0]] = _parse_record_row(cells)
    return result


def top3_rate(record: tuple[int, int, int, int], prior: float = 0.33, alpha: int = 5) -> float:
    """ベイズ平滑化した3着内率を返す"""
    w, s, t, u = record
    total = w + s + t + u
    if total == 0:
        return prior
    raw = (w + s + t) / total
    # ベイズ平滑: prior強度 alpha レース分の事前分布と混合
    return (raw * total + prior * alpha) / (total + alpha)


# ─────────────────────────────────────────────
# 馬データ取得
# ─────────────────────────────────────────────

def get_horse_data(date_str: str, race_no: int) -> list[dict]:
    """オッズページ＋出馬表から出走馬データを取得"""
    date_enc = date_str.replace("/", "%2F")

    # ── オッズページ (単勝・複勝・基本情報) ──
    odds_url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/OddsTanFuku"
        f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}&k_raceNo={race_no}"
    )
    soup_odds = fetch(odds_url)
    odds_table = soup_odds.find("table", class_="odd_popular_table_02")
    if not odds_table:
        return []

    horses = []
    for row in odds_table.find_all("tr")[1:]:  # ヘッダースキップ
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 10:
            continue
        try:
            # 列: 枠(0) 馬番(1) 馬名(2) 単勝(3) 複勝下(4) 複勝上(5) 性齢(6)
            #     体重増減(7) 負担重量(8) 騎手(所属)(9) 所属(10) 調教師(11) 変更(12)
            horse_no = int(cells[1])
            odds_tan = float(cells[3])

            fukusho_raw = cells[4].rstrip("-").strip()
            fukusho_low = float(fukusho_raw) if fukusho_raw else 0.0

            weight_change = 0
            wc_match = re.search(r"([+-]?\d+)", cells[7])
            if wc_match:
                weight_change = int(wc_match.group(1))

            jockey = re.sub(r"[▲◇★☆△]", "", cells[9]).strip()

            horses.append({
                "horse_no": horse_no,
                "frame": cells[0],
                "name": cells[2],
                "odds_tan": odds_tan,
                "fukusho_low": fukusho_low,
                "sex_age": cells[6],
                "weight_change": weight_change,
                "burden_weight": float(cells[8]) if cells[8] else 0.0,
                "jockey": jockey,
                "trainer": cells[11] if len(cells) > 11 else "",
                "change_info": cells[12] if len(cells) > 12 else "",
                "arrival": {},
            })
        except (ValueError, IndexError):
            continue

    if not horses:
        return horses

    # ── 出馬表 (着別成績) ──
    deba_url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/DebaTable"
        f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}&k_raceNo={race_no}"
    )
    soup_deba = fetch(deba_url)

    # class に "arrival" を含むテーブルが馬番順に並ぶ
    arrival_tables = [
        t for t in soup_deba.find_all("table")
        if "arrival" in (t.get("class") or [])
    ]

    for i, horse in enumerate(horses):
        if i < len(arrival_tables):
            horse["arrival"] = parse_arrival_table(arrival_tables[i])

    return horses


# ─────────────────────────────────────────────
# スコアリング & 確率計算
# ─────────────────────────────────────────────

def calc_top3_probs(horses: list[dict]) -> list[dict]:
    """各馬に3着内確率(%)を付与して返す"""
    n = len(horses)
    if n == 0:
        return horses

    # 単勝オッズから市場インプライド確率を算出 (オーバーラウンド補正あり)
    p_raw = [1.0 / max(h["odds_tan"], 1.01) for h in horses]
    p_sum = sum(p_raw)
    p_market = [p / p_sum for p in p_raw]  # 合計1に正規化

    # 基準となる均等3着内率
    base_top3 = 3.0 / n

    # 各馬のスコアを計算
    scores = []
    for i, horse in enumerate(horses):
        arrival = horse["arrival"]
        overall_rec = arrival.get("全", (0, 0, 0, 0))
        track_rec = arrival.get("場", (0, 0, 0, 0))
        dist_rec = arrival.get("距", (0, 0, 0, 0))

        r_overall = top3_rate(overall_rec, prior=base_top3)
        r_track = top3_rate(track_rec, prior=base_top3)
        r_dist = top3_rate(dist_rec, prior=base_top3)

        # 馬体重変動ペナルティ (大きな変動はマイナス)
        abs_wc = abs(horse["weight_change"])
        weight_factor = 1.0 if abs_wc <= 6 else (1.0 - (abs_wc - 6) * 0.01)
        weight_factor = max(weight_factor, 0.85)

        # 対数線形スコア: 市場30% + 通算20% + 当場25% + 当距離25%
        log_score = (
            0.30 * math.log(max(p_market[i], 1e-4))
            + 0.20 * math.log(max(r_overall, 0.01))
            + 0.25 * math.log(max(r_track, 0.01))
            + 0.25 * math.log(max(r_dist, 0.01))
        )
        scores.append(math.exp(log_score) * weight_factor)

    # スコアを正規化: 合計が3になるよう調整 (3頭が3着以内)
    score_sum = sum(scores)
    for i, horse in enumerate(horses):
        raw_prob = scores[i] / score_sum * 3.0  # 期待値ベースの3着内確率
        horse["top3_prob"] = min(raw_prob * 100, 99.9)  # %表記, 上限99.9%

    return horses


# ─────────────────────────────────────────────
# 表示
# ─────────────────────────────────────────────

def format_record(rec: tuple[int, int, int, int]) -> str:
    if not rec or rec == (0, 0, 0, 0):
        return "-"
    w, s, t, u = rec
    return f"{w}-{s}-{t}-{u}"


def print_race_prediction(race: dict, horses: list[dict]) -> None:
    if not horses:
        print("  データなし\n")
        return

    # 3着内確率で降順ソートして上位5頭
    sorted_horses = sorted(horses, key=lambda h: h["top3_prob"], reverse=True)
    top5 = sorted_horses[:5]

    rows = []
    for rank, h in enumerate(top5, 1):
        arrival = h["arrival"]
        overall = format_record(arrival.get("全"))
        track = format_record(arrival.get("場"))
        dist = format_record(arrival.get("距"))
        wc_str = f"{h['weight_change']:+d}kg" if h["weight_change"] != 0 else "  ---"
        rows.append([
            rank,
            h["horse_no"],
            h["name"],
            f"{h['top3_prob']:.1f}%",
            h["odds_tan"],
            overall,
            track,
            dist,
            wc_str,
            h["jockey"],
        ])

    headers = ["順", "馬番", "馬名", "3着内確率", "単勝", "通算", "当場", "当距離", "体重変動", "騎手"]
    print(tabulate(rows, headers=headers, tablefmt="simple", colalign=("right",) * 2 + ("left",) + ("right",) * 2 + ("left",) * 5))
    print()


# ─────────────────────────────────────────────
# 実績取得（検証用）
# ─────────────────────────────────────────────

def get_race_result(date_str: str, race_no: int) -> list[tuple[int, int, str]]:
    """実際の着順を返す: [(着順, 馬番, 馬名), ...]"""
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceMarkTable"
        f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}&k_raceNo={race_no}"
    )
    soup = fetch(url)
    table = soup.find("table")
    if not table:
        return []

    results = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        try:
            pos = int(cells[0])
            horse_no = int(cells[2])
            horse_name = cells[3]
            results.append((pos, horse_no, horse_name))
        except ValueError:
            continue
    return results


def print_race_verification(race: dict, horses: list[dict], results: list[tuple]) -> None:
    """予測と実績を並べて表示"""
    if not horses:
        print("  データなし\n")
        return

    sorted_horses = sorted(horses, key=lambda h: h["top3_prob"], reverse=True)
    top5 = sorted_horses[:5]
    top5_nos = {h["horse_no"] for h in top5}
    actual_top3_nos = {no for pos, no, _ in results if pos <= 3}

    rows = []
    for rank, h in enumerate(top5, 1):
        hit = "◎" if h["horse_no"] in actual_top3_nos else "✗"
        # 実際の着順
        actual_pos = next((pos for pos, no, _ in results if no == h["horse_no"]), "-")
        rows.append([
            rank,
            h["horse_no"],
            h["name"],
            f"{h['top3_prob']:.1f}%",
            h["odds_tan"],
            hit,
            actual_pos,
        ])

    headers = ["予測順", "馬番", "馬名", "3着内確率", "単勝", "的中", "実着順"]
    print(tabulate(rows, headers=headers, tablefmt="simple",
                   colalign=("right",) * 2 + ("left",) + ("right",) * 2 + ("center", "right")))

    # 実際の3着以内
    actual_str = "  実績 3着内: " + "  ".join(
        f"{pos}着 {no}番 {name}" for pos, no, name in sorted(results, key=lambda x: x[0])[:3]
    )
    print(actual_str)

    # ヒット数
    hits = len(top5_nos & actual_top3_nos)
    print(f"  → 予測上位5頭中 {hits}/3頭 的中\n")
    return hits


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="名古屋競馬予想")
    parser.add_argument("date", nargs="?", help="対象日 YYYY/MM/DD (省略時=当日)")
    parser.add_argument("--verify", action="store_true", help="実績と照合して精度検証")
    args = parser.parse_args()

    if args.date:
        try:
            dt = datetime.strptime(args.date, "%Y/%m/%d")
        except ValueError:
            print("日付形式エラー: YYYY/MM/DD で指定してください")
            sys.exit(1)
    else:
        dt = datetime.now()

    date_str = dt.strftime("%Y/%m/%d")
    date_disp = dt.strftime("%Y年%-m月%-d日")
    mode = "検証" if args.verify else "予想"

    print(f"\n{'=' * 65}")
    print(f"  名古屋競馬 {mode}  {date_disp}")
    print(f"  ※ スコア = 市場オッズ30% + 通算成績20% + 当場25% + 当距離25%")
    print(f"{'=' * 65}\n")

    print("レース一覧を取得中...")
    races = get_race_list(date_str)

    if not races:
        print(f"{date_disp}の名古屋競馬開催情報が見つかりませんでした。")
        sys.exit(1)

    print(f"  {len(races)}レース確認。各レースのデータを取得します...\n")

    total_hits = 0
    total_races = 0

    for race in races:
        rno = race["race_no"]
        course = race["course"]
        rtype = f"[{race['race_type']}]" if race["race_type"] else ""
        header = (
            f"【{rno:2d}R】 {race['time']} "
            f"{course} {race['weather']}/{race['track_cond']}  "
            f"{rtype}{race['race_name']}"
        )
        print(header)
        print("-" * len(header))

        try:
            horses = get_horse_data(date_str, rno)
            horses = calc_top3_probs(horses)

            if args.verify:
                results = get_race_result(date_str, rno)
                if results:
                    hits = print_race_verification(race, horses, results)
                    total_hits += hits
                    total_races += 1
                else:
                    print("  成績データなし（未開催または取得失敗）\n")
            else:
                print_race_prediction(race, horses)

        except requests.HTTPError as e:
            print(f"  HTTP エラー: {e}\n")
        except Exception as e:
            print(f"  取得エラー: {e}\n")

    if args.verify and total_races > 0:
        rate = total_hits / (total_races * 3) * 100
        print("=" * 65)
        print(f"  【検証サマリー】 {total_races}レース")
        print(f"  予測上位5頭での3着内カバー率: {total_hits}/{total_races * 3} = {rate:.1f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
