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


def get_race_dates_for_venue(session: "KeibaGoSession", year: int, baba_code: int) -> list[str]:
    """
    MonthlyConveneInfo から指定年・場の全開催日を取得。
    日付文字列リスト ['YYYY/MM/DD', ...] を返す。
    全日付スキャンより大幅に高速。
    """
    import re as _re
    dates = []
    for month in range(1, 13):
        url = (
            f"{BASE_URL}/KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop"
            f"?k_year={year}&k_month={month}"
        )
        soup = session.get_soup(url)
        if not soup:
            continue
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if f"babaCode={baba_code}" in href:
                m = _re.search(r"raceDate=([\d%2F]+)", href)
                if m:
                    dates.append(m.group(1).replace("%2F", "/"))
    return sorted(set(dates))


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


def get_race_data(session: KeibaGoSession, date_str: str,
                  baba_code: int, race_no: int) -> list[dict]:
    """RaceMarkTable から出走馬情報と結果を一括取得。
    OddsTanFuku は過去データで取得不可のため廃止。
    列順: [0]=着順,[1]=枠,[2]=馬番,[3]=馬名,[4]=所属,
          [5]=性齢,[6]=負担重量,[7]=騎手,[8]=調教師,
          [9]=馬体重(増減),[10]=タイム,[11]=着差,[12]=上がり3F,
          [13]=コーナー通過順,[14]=人気,[15]=単勝オッズ
    """
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

    horses = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        try:
            finish_pos_str = cells[0].strip()
            finish_pos = int(finish_pos_str) if finish_pos_str.isdigit() else None
            waku = int(cells[1]) if cells[1].isdigit() else 0
            horse_no = int(cells[2])
            horse_name = cells[3].strip()

            sex_age = cells[5].strip() if len(cells) > 5 else ""
            # 性齢: "牡 4" or "牡4"
            sex_age_clean = sex_age.replace("　", "").replace(" ", "")
            sex = sex_age_clean[0] if sex_age_clean else ""
            age_str = sex_age_clean[1:] if len(sex_age_clean) > 1 else ""
            age = int(age_str) if age_str.isdigit() else 0

            burden_str = cells[6].strip() if len(cells) > 6 else ""
            weight_carried = float(burden_str) if burden_str else 0.0

            jockey_raw = cells[7].strip() if len(cells) > 7 else ""
            jockey = re.sub(r"（[^）]*）", "", jockey_raw).strip()
            jockey = re.sub(r"[▲◇★☆△〇]", "", jockey).strip()

            trainer = cells[8].strip() if len(cells) > 8 else ""

            hw_text = cells[9].replace("（", "(").replace("）", ")") if len(cells) > 9 else ""
            hw_m = re.match(r"(\d+)\(([+-]?\d+)\)", hw_text)
            horse_weight = int(hw_m.group(1)) if hw_m else 0
            weight_change = int(hw_m.group(2)) if hw_m else 0

            finish_time = cells[10].strip() if len(cells) > 10 else ""
            last_3f = cells[12].strip() if len(cells) > 12 else ""
            passage_rate = cells[13].strip() if len(cells) > 13 else ""
            popularity = int(cells[14]) if len(cells) > 14 and cells[14].isdigit() else None
            odds_str = cells[15].strip() if len(cells) > 15 else ""
            win_odds = None
            if odds_str and odds_str not in ("---", "", "取消"):
                try:
                    win_odds = float(odds_str)
                except ValueError:
                    pass

            horses.append({
                "horse_no": horse_no,
                "waku": waku,
                "horse_name": horse_name,
                "sex": sex,
                "age": age,
                "weight_carried": weight_carried,
                "jockey": jockey,
                "trainer": trainer,
                "horse_weight": horse_weight,
                "weight_change": weight_change,
                "win_odds": win_odds,
                "finish_position": finish_pos,
                "finish_time": finish_time,
                "last_3f": last_3f,
                "passage_rate": passage_rate,
                "popularity": popularity,
                "result_weight": horse_weight,
                "result_weight_change": weight_change,
            })
        except (ValueError, IndexError):
            continue
    return horses


def get_deba_entries(session: KeibaGoSession, date_str: str,
                     baba_code: int, race_no: int) -> list[dict]:
    """
    DebaTable から出走表を取得（レース前の当日出走表収集用）。
    RaceMarkTable はレース前は空になるため、出走表には DebaTable を使う。

    各馬ブロックは11行構成:
      row+0 (38cells): waku, horse_no, horse_name, jockey(所属), odds, stats...
      row+7 (6cells) : sex+age, 毛色, 生年月日, weight_carried+成績, クラス...
      row+8 (7cells) : 父馬, trainer(所属), ...
    """
    date_enc = date_str.replace("/", "%2F")
    url = (
        f"{BASE_URL}/KeibaWeb/TodayRaceInfo/DebaTable"
        f"?k_raceDate={date_enc}&k_babaCode={baba_code}&k_raceNo={race_no}"
    )
    soup = session.get_soup(url)
    if not soup:
        return []
    tables = soup.find_all("table")
    if not tables:
        return []
    table = tables[0]
    rows = table.find_all("tr")

    BLOCK = 11  # 1頭あたりの行数
    START = 2   # ヘッダー2行をスキップ

    horses = []
    i = START
    while i + BLOCK <= len(rows):
        main_cells = [c.get_text(strip=True) for c in rows[i].find_all(["td", "th"])]
        if len(main_cells) < 5:
            i += 1
            continue

        # row+7: sex/age, 毛色, 生年月日, weight_carried...
        info_cells = [c.get_text(strip=True) for c in rows[i + 7].find_all(["td", "th"])]
        # row+8: 父馬, trainer(所属)...
        trainer_cells = [c.get_text(strip=True) for c in rows[i + 8].find_all(["td", "th"])]

        try:
            # 同一枠2頭目は枠番セルがrowspan=2で省略されるため37セルになる
            if len(main_cells) >= 38:
                waku = int(main_cells[0])
                horse_no = int(main_cells[1])
                horse_name = main_cells[2]
                jockey_raw = main_cells[3]
                win_odds_raw = main_cells[4] if len(main_cells) > 4 else ""
            else:
                # 枠番なし行: [馬番, 馬名, 騎手, オッズ, ...]
                waku = None
                horse_no = int(main_cells[0])
                horse_name = main_cells[1]
                jockey_raw = main_cells[2]
                win_odds_raw = main_cells[3] if len(main_cells) > 3 else ""

            # 騎手: '岡村健（船橋）' → '岡村健'
            jockey = jockey_raw.split("（")[0].strip() if "（" in jockey_raw else jockey_raw

            # オッズ（レース前は空のことが多い）
            try:
                win_odds = float(win_odds_raw)
            except (ValueError, TypeError):
                win_odds = None

            # sex/age: 'セン8' → sex='セン', age=8 / '牡5' → sex='牡', age=5
            sex_age = info_cells[0] if info_cells else ""
            sex = sex_age.rstrip("0123456789") or ""
            try:
                age = int(sex_age[len(sex):])
            except (ValueError, IndexError):
                age = None

            # weight_carried: info_cells[3] like '56.0　1-3-3-16' → 56.0
            wc_raw = info_cells[3] if len(info_cells) > 3 else ""
            try:
                weight_carried = float(wc_raw.split("　")[0].split()[0])
            except (ValueError, IndexError):
                weight_carried = None

            # trainer: '遠藤茂（大井）' → '遠藤茂'
            trainer_raw = trainer_cells[1] if len(trainer_cells) > 1 else ""
            trainer = trainer_raw.split("（")[0].strip() if "（" in trainer_raw else trainer_raw

        except (ValueError, IndexError):
            i += BLOCK
            continue

        horses.append({
            "waku": waku,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "jockey": jockey,
            "win_odds": win_odds,
            "sex": sex,
            "age": age,
            "weight_carried": weight_carried,
            "horse_weight": None,   # レース前は不明
            "weight_change": None,  # レース前は不明
            "trainer": trainer,
        })
        i += BLOCK

    return horses


def get_race_entries(session: KeibaGoSession, date_str: str,
                     baba_code: int, race_no: int) -> list[dict]:
    """後方互換ラッパー: get_race_data を呼ぶ"""
    return get_race_data(session, date_str, baba_code, race_no)


def get_race_results(session: KeibaGoSession, date_str: str,
                     baba_code: int, race_no: int) -> list[dict]:
    """後方互換ラッパー: get_race_data を呼ぶ（結果フィールドのみ返す）"""
    data = get_race_data(session, date_str, baba_code, race_no)
    return [{
        "horse_no": h["horse_no"],
        "horse_name": h["horse_name"],
        "finish_position": h["finish_position"],
        "finish_time": h["finish_time"],
        "last_3f": h["last_3f"],
        "passage_rate": h["passage_rate"],
        "popularity": h["popularity"],
        "result_weight": h["result_weight"],
        "result_weight_change": h["result_weight_change"],
    } for h in data]
