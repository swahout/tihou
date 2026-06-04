#!/usr/bin/env python3
"""
重みの最適化スクリプト
1. 過去レースデータを取得してキャッシュ (cache.json)
2. グリッドサーチで3着内カバー率が最大になる重みを探索
"""

import json
import math
import time
import itertools
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# predict.py の関数を流用
from predict import (
    BASE_URL, BABA_CODE, HEADERS,
    fetch, get_race_list, get_horse_data,
    calc_top3_probs, top3_rate, parse_arrival_table,
)

CACHE_FILE = Path("cache.json")
FETCH_DELAY = 0.8  # キャッシュ構築時は少し速め

TARGET_DATES = [
    "2026/06/03", "2026/06/02",
    "2026/05/25", "2026/05/22", "2026/05/21",
    "2026/05/20", "2026/05/19",
    "2026/05/08", "2026/05/07", "2026/05/06", "2026/05/05",
]


# ─────────────────────────────────────────────
# データ取得 & キャッシュ
# ─────────────────────────────────────────────

def get_race_result(date_str: str, race_no: int) -> list[tuple[int, int, str]]:
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceMarkTable"
        f"?k_raceDate={date_enc}&k_babaCode={BABA_CODE}&k_raceNo={race_no}"
    )
    time.sleep(FETCH_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    table = soup.find("table")
    if not table:
        return []
    results = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        try:
            results.append((int(cells[0]), int(cells[2]), cells[3]))
        except ValueError:
            continue
    return results


def build_cache() -> list[dict]:
    """全日程のレースデータを取得してキャッシュ保存"""
    all_races = []
    total_dates = len(TARGET_DATES)

    for di, date_str in enumerate(TARGET_DATES):
        print(f"\n[{di+1}/{total_dates}] {date_str} のデータを取得中...")
        races = get_race_list(date_str)
        if not races:
            print("  開催なし or 取得失敗")
            continue

        for race in races:
            rno = race["race_no"]
            print(f"  {rno:2d}R 取得中...", end="\r")
            try:
                horses = get_horse_data(date_str, rno)
                results = get_race_result(date_str, rno)
                if not horses or not results:
                    continue

                # 必要なデータだけ保存（JSONシリアライズ可能な形式）
                horse_data = []
                for h in horses:
                    arrival = h.get("arrival", {})
                    horse_data.append({
                        "horse_no": h["horse_no"],
                        "odds_tan": h["odds_tan"],
                        "arrival": {k: list(v) for k, v in arrival.items()},
                    })

                all_races.append({
                    "date": date_str,
                    "race_no": rno,
                    "horses": horse_data,
                    "results": [(p, n, nm) for p, n, nm in results],
                })
            except Exception as e:
                print(f"  {rno}R エラー: {e}")

        print(f"  完了: {len([r for r in all_races if r['date']==date_str])}レース取得")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_races, f, ensure_ascii=False, indent=2)
    print(f"\nキャッシュ保存: {CACHE_FILE} ({len(all_races)}レース)")
    return all_races


def load_cache() -> list[dict]:
    with open(CACHE_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# スコアリング（重みを外部から渡すバージョン）
# ─────────────────────────────────────────────

def score_race(horses_raw: list[dict], w_market: float, w_overall: float,
               w_track: float, w_dist: float) -> list[tuple[int, float]]:
    """(horse_no, top3_prob%) のリストを返す"""
    n = len(horses_raw)
    if n == 0:
        return []

    p_raw = [1.0 / max(h["odds_tan"], 1.01) for h in horses_raw]
    p_sum = sum(p_raw)
    p_market = [p / p_sum for p in p_raw]
    base_top3 = 3.0 / n

    scores = []
    for i, h in enumerate(horses_raw):
        arr = {k: tuple(v) for k, v in h.get("arrival", {}).items()}
        r_overall = top3_rate(arr.get("全", (0, 0, 0, 0)), prior=base_top3)
        r_track   = top3_rate(arr.get("場", (0, 0, 0, 0)), prior=base_top3)
        r_dist    = top3_rate(arr.get("距", (0, 0, 0, 0)), prior=base_top3)

        log_score = (
            w_market  * math.log(max(p_market[i], 1e-4))
            + w_overall * math.log(max(r_overall, 0.01))
            + w_track   * math.log(max(r_track,   0.01))
            + w_dist    * math.log(max(r_dist,    0.01))
        )
        scores.append(math.exp(log_score))

    score_sum = sum(scores)
    results = []
    for i, h in enumerate(horses_raw):
        prob = min(scores[i] / score_sum * 3.0 * 100, 99.9)
        results.append((h["horse_no"], prob))
    return results


def evaluate(races: list[dict], w_market: float, w_overall: float,
             w_track: float, w_dist: float) -> tuple[float, int, int]:
    """カバー率とヒット数/総数を返す"""
    total_hits = 0
    total_possible = 0

    for race in races:
        scored = score_race(race["horses"], w_market, w_overall, w_track, w_dist)
        if not scored:
            continue

        top5_nos = {no for no, _ in sorted(scored, key=lambda x: -x[1])[:5]}
        actual_top3 = {no for pos, no, _ in race["results"] if pos <= 3}

        total_hits += len(top5_nos & actual_top3)
        total_possible += len(actual_top3)

    rate = total_hits / total_possible if total_possible > 0 else 0
    return rate, total_hits, total_possible


# ─────────────────────────────────────────────
# グリッドサーチ
# ─────────────────────────────────────────────

def grid_search(races: list[dict], step: float = 0.05) -> list[dict]:
    """全重み組み合わせを探索して上位20件を返す"""
    vals = [round(v * step, 10) for v in range(1, int(1.0 / step))]  # 0.05〜0.95
    results = []
    combos = [(w1, w2, w3, w4)
              for w1, w2, w3, w4 in itertools.product(vals, repeat=4)
              if abs(w1 + w2 + w3 + w4 - 1.0) < 1e-9]

    print(f"探索する組み合わせ数: {len(combos)}")
    for i, (w1, w2, w3, w4) in enumerate(combos):
        if i % 200 == 0:
            print(f"  {i}/{len(combos)}...", end="\r")
        rate, hits, total = evaluate(races, w1, w2, w3, w4)
        results.append({
            "w_market": w1, "w_overall": w2, "w_track": w3, "w_dist": w4,
            "rate": rate, "hits": hits, "total": total,
        })

    return sorted(results, key=lambda x: -x["rate"])


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    # キャッシュがなければ取得
    if CACHE_FILE.exists():
        print(f"キャッシュ読み込み: {CACHE_FILE}")
        races = load_cache()
        print(f"  {len(races)}レース")
    else:
        print("キャッシュなし。データ取得開始（数分かかります）...")
        races = build_cache()

    print(f"\n合計 {len(races)} レースでグリッドサーチ開始 (step=0.05)...")
    top_results = grid_search(races, step=0.05)

    print(f"\n{'=' * 65}")
    print(f"  最適重み TOP20  (データ: {len(races)}レース)")
    print(f"{'=' * 65}")
    print(f"{'順':>3}  {'市場オッズ':>8}  {'通算':>6}  {'当場':>6}  {'当距離':>6}  {'カバー率':>8}  {'的中':>10}")
    print("-" * 65)
    for rank, r in enumerate(top_results[:20], 1):
        print(
            f"{rank:>3}  {r['w_market']:>8.0%}  {r['w_overall']:>6.0%}  "
            f"{r['w_track']:>6.0%}  {r['w_dist']:>6.0%}  "
            f"{r['rate']:>7.1%}  {r['hits']:>4}/{r['total']}"
        )

    best = top_results[0]
    print(f"\n最良設定: オッズ={best['w_market']:.0%}  通算={best['w_overall']:.0%}  "
          f"当場={best['w_track']:.0%}  当距離={best['w_dist']:.0%}  "
          f"→ {best['rate']:.1%} ({best['hits']}/{best['total']})")


if __name__ == "__main__":
    main()
