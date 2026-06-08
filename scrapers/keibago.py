"""NAR公式サイト (keiba.go.jp) スクレイパー"""
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import Optional

BASE_URL = "https://www.keiba.go.jp"
DEFAULT_DELAY = 1.2

VENUE_MAP = {
    "大井": 20, "船橋": 19, "川崎": 21, "浦和": 18,
    "門別": 36, "園田": 27, "姫路": 28, "名古屋": 24,
    "金沢": 22, "笠松": 25, "高知": 42, "佐賀": 44,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class KeibaGoSession:
    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        try:
            resp = self.session.get(url, timeout=15)
            self._last = time.time()
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "lxml")
        except Exception:
            self._last = time.time()
            return None


def get_race_list(session: KeibaGoSession, date_str: str, baba_code: int) -> list[dict]:
    """指定日・場のレース一覧を返す。date_str='YYYY/MM/DD'"""
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceList"
        f"?k_raceDate={date_enc}&k_babaCode={baba_code}"
    )
    soup = session.get_soup(url)
    if not soup:
        return []
    table = soup.find("table")
    if not table:
        return []

    races = []
    for row in table.find_all("tr")[2:]:
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 5:
            continue
        m = re.match(r"(\d+)R", cells[0])
        if not m:
            continue
        course = cells[5] if len(cells) > 5 else ""
        course_info = _parse_course(course)
        races.append({
            "race_no": int(m.group(1)),
            "race_name": cells[4] if len(cells) > 4 else "",
            "course_str": course,
            "weather": cells[6] if len(cells) > 6 else "",
            "track_cond": cells[7] if len(cells) > 7 else "",
            **course_info,
        })
    return races


def _parse_course(course_str: str) -> dict:
    m = re.search(r"(\d{3,4})", course_str)
    distance = int(m.group(1)) if m else 0
    surface = "芝" if "芝" in course_str else "ダ"
    direction = ""
    for d in ("右", "左", "外", "内"):
        if d in course_str:
            direction = d
            break
    return {"distance": distance, "surface": surface, "direction": direction}


def get_race_entries(session: KeibaGoSession, date_str: str,
                     baba_code: int, race_no: int) -> list[dict]:
    """OddsTanFuku から出走馬情報を取得"""
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/OddsTanFuku"
        f"?k_raceDate={date_enc}&k_babaCode={baba_code}&k_raceNo={race_no}"
    )
    soup = session.get_soup(url)
    if not soup:
        return []
    table = soup.find("table", class_="odd_popular_table_02")
    if not table:
        return []

    horses = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 9:
            continue
        try:
            horse_no = int(cells[1])
            waku_str = cells[0].strip()
            waku = int(waku_str) if waku_str.isdigit() else 0

            odds_str = cells[3].strip()
            win_odds = None
            if odds_str and odds_str not in ("---", "", "取消"):
                try:
                    win_odds = float(odds_str)
                except ValueError:
                    pass

            sex_age = cells[6].strip()
            sex = sex_age[0] if sex_age else ""
            age_str = sex_age[1:].strip() if len(sex_age) > 1 else ""
            age = int(age_str) if age_str.isdigit() else 0

            wc_text = cells[7].strip() if len(cells) > 7 else ""
            hw_match = re.match(r"(\d+)\(([+-]?\d+)\)", wc_text)
            if hw_match:
                horse_weight = int(hw_match.group(1))
                weight_change = int(hw_match.group(2))
            else:
                horse_weight = 0
                c_match = re.search(r"([+-]?\d+)", wc_text)
                weight_change = int(c_match.group(1)) if c_match else 0

            burden_str = cells[8].strip() if len(cells) > 8 else ""
            weight_carried = float(burden_str) if burden_str else 0.0

            jockey = re.sub(r"[▲◇★☆△〇]", "", cells[9]).strip() if len(cells) > 9 else ""
            trainer = cells[11].strip() if len(cells) > 11 else ""

            horses.append({
                "horse_no": horse_no,
                "waku": waku,
                "horse_name": cells[2].strip(),
                "win_odds": win_odds,
                "sex": sex,
                "age": age,
                "horse_weight": horse_weight,
                "weight_change": weight_change,
                "weight_carried": weight_carried,
                "jockey": jockey,
                "trainer": trainer,
            })
        except (ValueError, IndexError):
            continue
    return horses


def get_race_results(session: KeibaGoSession, date_str: str,
                     baba_code: int, race_no: int) -> list[dict]:
    """RaceMarkTable から着順を取得"""
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/RaceMarkTable"
        f"?k_raceDate={date_enc}&k_babaCode={baba_code}&k_raceNo={race_no}"
    )
    soup = session.get_soup(url)
    if not soup:
        return []
    table = soup.find("table")
    if not table:
        return []

    results = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        try:
            finish_pos = int(cells[0])
            horse_no = int(cells[2])
            horse_name = cells[3].strip()
            finish_time = cells[4].strip() if len(cells) > 4 else ""
            results.append({
                "horse_no": horse_no,
                "horse_name": horse_name,
                "finish_position": finish_pos,
                "finish_time": finish_time,
            })
        except ValueError:
            continue
    return results
