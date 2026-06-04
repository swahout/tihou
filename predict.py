#!/usr/bin/env python3
"""地方競馬1日予想 - 3着内確率を計算してレース毎に上位7頭を選出"""

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
BABA_CODE = 24  # デフォルト: 名古屋
BABA_NAME = "名古屋"

VENUE_MAP = {
    "名古屋": 24,
    "船橋":   19,
    "大井":   20,
    "川崎":   21,
    "浦和":   18,
    "門別":   36,
    "園田":   27,
    "姫路":   28,
    "金沢":   22,
    "笠松":   25,
    "高知":   42,
    "佐賀":   44,
}
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

    # ── 過去走（馬場状態別成績用）──
    extract_past_races(soup_deba, horses)

    return horses


# ─────────────────────────────────────────────
# 過去走データ解析（馬場状態別成績）
# ─────────────────────────────────────────────

TRACK_CONDITIONS = {"重", "稍重", "良", "不良"}
_PAST_RACE_RE = re.compile(r"^(\d{1,2})(\d{2}\.\d{2}\.\d{2})")


def parse_past_race_cell(text: str) -> dict | None:
    """'526.05.05　重　5頭名古屋　右2000　3番' → {'pos': 5, 'track_cond': '重'}"""
    m = _PAST_RACE_RE.match(text)
    if not m:
        return None
    pos = int(m.group(1))
    parts = [p.strip() for p in re.split(r"[\s　]+", text) if p.strip()]
    track_cond = parts[1] if len(parts) > 1 and parts[1] in TRACK_CONDITIONS else None
    return {"pos": pos, "track_cond": track_cond}


def extract_past_races(soup_deba, horses: list[dict]) -> None:
    """出馬表 Table 0 から各馬の過去5走（馬場状態・着順）を horses に付与"""
    tables = soup_deba.find_all("table")
    if not tables:
        return
    name_to_idx = {h["name"]: i for i, h in enumerate(horses)}

    for row in tables[0].find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        matched_idx = next((name_to_idx[c] for c in cells if c in name_to_idx), None)
        if matched_idx is None:
            continue
        past = [p for c in cells if (p := parse_past_race_cell(c)) is not None]
        if past:
            horses[matched_idx]["past_races"] = past


# ─────────────────────────────────────────────
# 今日の馬場傾向（枠番バイアス）
# ─────────────────────────────────────────────

def get_today_trend(date_str: str, current_race_no: int) -> dict:
    """
    完了済みレースから枠番バイアスを計算。
    Returns: {'completed': N, 'gate_bias': {gate: factor}, 'enough_data': bool}
    """
    gate_stats: dict[int, list[int]] = {i: [0, 0] for i in range(1, 9)}  # {gate: [entries, top3]}
    completed = 0

    for rno in range(1, current_race_no):
        try:
            date_enc = date_str.replace("/", "%2F")
            url = (
                f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceMarkTable"
                f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}&k_raceNo={rno}"
            )
            time.sleep(0.6)
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")
            table = soup.find("table")
            if not table:
                continue
            rows = table.find_all("tr")[1:]
            if not rows:
                continue
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 3:
                    continue
                try:
                    pos  = int(cells[0])
                    gate = int(cells[1])
                    if 1 <= gate <= 8:
                        gate_stats[gate][0] += 1
                        if pos <= 3:
                            gate_stats[gate][1] += 1
                except ValueError:
                    continue
            completed += 1
        except Exception:
            continue

    total_entries = sum(v[0] for v in gate_stats.values())
    total_top3    = sum(v[1] for v in gate_stats.values())
    if total_entries == 0:
        return {"completed": 0, "gate_bias": {}, "enough_data": False}

    expected_rate = total_top3 / total_entries  # ≈ 3/N

    gate_bias: dict[int, float] = {}
    for gate, (entries, top3) in gate_stats.items():
        if entries >= 2:
            alpha    = 4  # ベイズ平滑化の強さ
            smoothed = (top3 + expected_rate * alpha) / (entries + alpha)
            bias     = smoothed / expected_rate
            gate_bias[gate] = min(max(bias, 0.75), 1.35)  # 外れ値をクリップ
        else:
            gate_bias[gate] = 1.0

    return {"completed": completed, "gate_bias": gate_bias, "enough_data": completed >= 3}


def format_trend(trend: dict) -> str:
    """枠番傾向を1行でフォーマット"""
    if not trend.get("enough_data"):
        n = trend.get("completed", 0)
        return f"  今日の傾向: データ不足（{n}レース完了）"
    bias = trend["gate_bias"]
    parts = []
    for gate in range(1, 9):
        b = bias.get(gate, 1.0)
        mark = "↑" if b >= 1.10 else ("↓" if b <= 0.90 else "－")
        parts.append(f"枠{gate}:{b:.2f}{mark}")
    return f"  今日の傾向({trend['completed']}R完了): " + "  ".join(parts)


# ─────────────────────────────────────────────
# スコアリング & 確率計算
# ─────────────────────────────────────────────

def calc_top3_probs(horses: list[dict], today_cond: str = "",
                    today_trend: dict | None = None) -> list[dict]:
    """各馬に3着内確率を付与して返す。
    - perf_prob : 成績確率（通算15%+当場30%+当距離30%+当馬場25%）
    - mkt_prob  : 市場確率（オッズのみ）
    - top3_prob : 最終3着内確率（幾何平均 × 枠番バイアス）
    """
    n = len(horses)
    if n == 0:
        return horses

    base_top3 = 3.0 / n
    gate_bias = (today_trend or {}).get("gate_bias", {}) if (today_trend or {}).get("enough_data") else {}

    perf_scores = []
    for horse in horses:
        arrival = horse["arrival"]
        r_overall = top3_rate(arrival.get("全", (0, 0, 0, 0)), prior=base_top3)
        r_track   = top3_rate(arrival.get("場", (0, 0, 0, 0)), prior=base_top3)
        r_dist    = top3_rate(arrival.get("距", (0, 0, 0, 0)), prior=base_top3)

        # 馬場状態別成績
        past = horse.get("past_races", [])
        cond_past = [p for p in past if p.get("track_cond") == today_cond]
        if cond_past:
            cond_top3  = sum(1 for p in cond_past if p["pos"] <= 3)
            alpha      = 3
            r_cond     = (cond_top3 + base_top3 * alpha) / (len(cond_past) + alpha)
        else:
            r_cond = base_top3  # データなし → 均等とみなす

        abs_wc = abs(horse["weight_change"])
        weight_factor = 1.0 if abs_wc <= 6 else max(1.0 - (abs_wc - 6) * 0.01, 0.85)

        # 通算15% + 当場30% + 当距離30% + 当馬場25%
        log_score = (
            0.15 * math.log(max(r_overall, 0.01))
            + 0.30 * math.log(max(r_track,   0.01))
            + 0.30 * math.log(max(r_dist,    0.01))
            + 0.25 * math.log(max(r_cond,    0.01))
        )
        perf_scores.append(math.exp(log_score) * weight_factor)

    mkt_raw   = [1.0 / max(h["odds_tan"], 1.01) for h in horses]
    mkt_sum   = sum(mkt_raw)
    mkt_scores = [v / mkt_sum for v in mkt_raw]

    perf_sum = sum(perf_scores)
    for i, horse in enumerate(horses):
        p_perf = perf_scores[i] / perf_sum * 3.0
        p_mkt  = mkt_scores[i] * 3.0

        try:
            bias = gate_bias.get(int(horse.get("frame", 0)), 1.0)
        except (ValueError, TypeError):
            bias = 1.0

        horse["_combined_raw"] = math.sqrt(p_perf * p_mkt) * bias
        horse["perf_prob"] = min(p_perf * 100, 99.9)
        horse["mkt_prob"]  = min(p_mkt  * 100, 99.9)

    combined_sum = sum(h["_combined_raw"] for h in horses)
    for horse in horses:
        horse["top3_prob"] = min(horse["_combined_raw"] / combined_sum * 3.0 * 100, 99.9)

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

    sorted_horses = sorted(horses, key=lambda h: h["top3_prob"], reverse=True)
    top5 = sorted_horses[:7]

    rows = []
    for rank, h in enumerate(top5, 1):
        rows.append([
            rank,
            h["horse_no"],
            h["name"],
            f"{h['perf_prob']:.1f}%",
            f"{h['mkt_prob']:.1f}%",
            f"{h['top3_prob']:.1f}%",
            h["odds_tan"],
            h["jockey"],
        ])

    headers = ["順", "馬番", "馬名", "成績確率", "市場確率", "最終3着内確率", "単勝", "騎手"]
    print(tabulate(rows, headers=headers, tablefmt="simple",
                   colalign=("right",) * 2 + ("left",) + ("right",) * 5 + ("left",)))
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
    top5 = sorted_horses[:7]
    top5_nos = {h["horse_no"] for h in top5}
    actual_top3_nos = {no for pos, no, _ in results if pos <= 3}

    rows = []
    for rank, h in enumerate(top5, 1):
        hit = "◎" if h["horse_no"] in actual_top3_nos else "✗"
        actual_pos = next((pos for pos, no, _ in results if no == h["horse_no"]), "-")
        rows.append([
            rank,
            h["horse_no"],
            h["name"],
            f"{h['perf_prob']:.1f}%",
            f"{h['mkt_prob']:.1f}%",
            f"{h['top3_prob']:.1f}%",
            h["odds_tan"],
            hit,
            actual_pos,
        ])

    headers = ["予測順", "馬番", "馬名", "成績確率", "市場確率", "最終3着内確率", "単勝", "的中", "実着順"]
    print(tabulate(rows, headers=headers, tablefmt="simple",
                   colalign=("right",) * 2 + ("left",) + ("right",) * 5 + ("center", "right")))

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
    global BABA_CODE, BABA_NAME
    parser = argparse.ArgumentParser(description="地方競馬予想")
    parser.add_argument("date", nargs="?", help="対象日 YYYY/MM/DD (省略時=当日)")
    parser.add_argument("--venue", default="名古屋",
                        help=f"競馬場名 (デフォルト: 名古屋) 対応: {', '.join(VENUE_MAP)}")
    parser.add_argument("--verify", action="store_true", help="実績と照合して精度検証")
    args = parser.parse_args()

    if args.venue not in VENUE_MAP:
        print(f"未対応の競馬場: {args.venue}")
        print(f"対応: {', '.join(VENUE_MAP)}")
        sys.exit(1)
    BABA_CODE = VENUE_MAP[args.venue]
    BABA_NAME = args.venue

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
    print(f"  {BABA_NAME}競馬 {mode}  {date_disp}")
    print(f"  ※ 成績確率(通算15%+当場30%+当距離30%+当馬場25%) × 市場確率 × 枠番傾向")
    print(f"{'=' * 65}\n")

    print("レース一覧を取得中...")
    races = get_race_list(date_str)

    if not races:
        print(f"{date_disp}の{BABA_NAME}競馬開催情報が見つかりませんでした。")
        sys.exit(1)

    print(f"  {len(races)}レース確認。各レースのデータを取得します...\n")

    # 予想モードのみ今日の傾向を先に取得
    today_trend: dict | None = None
    if not args.verify:
        first_rno = races[0]["race_no"]
        if first_rno > 1:
            print("今日の傾向を集計中...")
            today_trend = get_today_trend(date_str, first_rno)
            print(format_trend(today_trend))
            print()

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

        # レースが進むごとに傾向を更新（予想モードのみ）
        if not args.verify and today_trend is not None:
            today_trend = get_today_trend(date_str, rno)
            print(format_trend(today_trend))

        try:
            horses = get_horse_data(date_str, rno)
            horses = calc_top3_probs(
                horses,
                today_cond=race["track_cond"],
                today_trend=today_trend if not args.verify else None,
            )

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
