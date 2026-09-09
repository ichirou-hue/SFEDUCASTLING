import os
import re
import time
import requests
from typing import Set, List, Dict, Tuple, Any
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "parsed_texts")
OUT_FILE = os.path.join(OUT_DIR, "articles_parsed.txt")

# Корневые категории шахматной теории
ROOT_CATEGORIES = [
    {"lang": "ru", "category": "Категория:Шахматная_стратегия", "default_stage": "middlegame"},
    {"lang": "ru", "category": "Категория:Шахматная_тактика", "default_stage": "middlegame"},
    {"lang": "ru", "category": "Категория:Шахматные_окончания", "default_stage": "endgame"},
    {"lang": "ru", "category": "Категория:Шахматная_теория", "default_stage": "middlegame"},
    {"lang": "ru", "category": "Категория:Шахматные_термины", "default_stage": "middlegame"},
    {"lang": "ru", "category": "Категория:Пешечные_окончания", "default_stage": "endgame"},
    {"lang": "ru", "category": "Категория:Ладейные_окончания", "default_stage": "endgame"},
    {"lang": "ru", "category": "Категория:Шахматные_дебюты", "default_stage": "opening"},
    {"lang": "en", "category": "Category:Chess_strategy", "default_stage": "middlegame"},
    {"lang": "en", "category": "Category:Chess_tactics", "default_stage": "middlegame"},
    {"lang": "en", "category": "Category:Chess_endgames", "default_stage": "endgame"},
    {"lang": "en", "category": "Category:Pawn_endgames", "default_stage": "endgame"},
    {"lang": "en", "category": "Category:Rook_endgames", "default_stage": "endgame"},
    {"lang": "en", "category": "Category:Chess_openings", "default_stage": "opening"},
    {"lang": "en", "category": "Category:Chess_theory", "default_stage": "middlegame"},
    {"lang": "en", "category": "Category:Chess_terminology", "default_stage": "middlegame"}
]

# Исключаем биографии и турнирные списки
# В knowledge_base/parsers/auto_crawler.py обновляем списки фильтрации:

EXCLUDE_KEYWORDS = [
    "championship", "tournament", "grandmaster", "biography", "list of", "olympiad",
    "fide", "world cup", "matches", "controversies", "scandal", "memorial", "round-robin",
    "painting", "art", "museum", "culture", "film", "book", "novel", "poem", "sculpture",
    "чемпионат", "турнир", "гроссмейстер", "биография", "список", "олимпиада", "матч",
    "первенство", "мемориал", "кубок", "скандал", "чемпион", "мастер спорта", "рейтинг",
    "картина", "живопись", "художник", "музей", "искусство", "в культуре", "фильм", "гравюра"
]

SECTION_BLACKLIST = [
    "ранние годы", "детство", "юность", "личная жизнь", "семья", "смерть", "память",
    "early life", "childhood", "personal life", "death", "family", "illness",
    "турнирные результаты", "спортивные результаты", "таблица результатов", 
    "статистика", "турниры", "матчи", "tournament results", "career statistics", 
    "match results", "olympiad results", "примечания", "литература", "ссылки",
    "источники", "см. также", "references", "notes", "further reading", "external links",
    "в искусстве", "в культуре", "живопись", "галерея", "сюжет картины", "описание картины"
]

class WikiChessCrawler:
    def __init__(self, target_articles_limit: int = 300, max_depth: int = 2):
        self.limit = target_articles_limit
        self.max_depth = max_depth
        self.visited_titles: Set[str] = set()
        self.visited_categories: Set[str] = set()
        self.collected_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ChessCoachRAGPipeline/3.0 (Educational Academic Project; contact: student@sfedu.ru)"
        })
        self.translator = GoogleTranslator(source='auto', target='ru')
        os.makedirs(OUT_DIR, exist_ok=True)

    def fetch_category_members(self, lang: str, category_name: str) -> List[Dict[str, Any]]:
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_name,
            "cmlimit": "250",
            "format": "json"
        }
        try:
            time.sleep(0.08)
            r = self.session.get(api_url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("query", {}).get("categorymembers", [])
            return []
        except Exception as e:
            print(f"[WARN] Сбой запроса категории {category_name}: {e}")
            return []

    def fetch_article_text(self, lang: str, title: str) -> str:
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "format": "json"
        }
        try:
            time.sleep(0.08)
            r = self.session.get(api_url, params=params, timeout=10)
            sections = r.json().get("parse", {}).get("sections", [])
        except Exception:
            sections = []

        # Если разделов нет — забираем общий экстракт
        if not sections:
            return self._fetch_full_extract(lang, title)

        clean_blocks = []
        for sec in sections:
            sec_title = sec.get("line", "")
            sec_idx = sec.get("index")

            if any(bad in sec_title.lower() for bad in SECTION_BLACKLIST):
                continue

            sec_params = {
                "action": "parse",
                "page": title,
                "section": sec_idx,
                "prop": "text",
                "format": "json"
            }
            try:
                time.sleep(0.05)
                sec_r = self.session.get(api_url, params=sec_params, timeout=10)
                raw_html = sec_r.json().get("parse", {}).get("text", {}).get("*", "")
                soup = BeautifulSoup(raw_html, "html.parser")

                for tag in soup(["table", "nav", "style", "script", "sup", "span", "figure"]):
                    tag.decompose()

                paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 70]
                if paras:
                    clean_blocks.append(f"### {sec_title}\n" + "\n\n".join(paras))
            except Exception:
                continue

        return "\n\n".join(clean_blocks) if clean_blocks else self._fetch_full_extract(lang, title)

    def _fetch_full_extract(self, lang: str, title: str) -> str:
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": "1",
            "format": "json"
        }
        try:
            time.sleep(0.08)
            r = self.session.get(api_url, params=params, timeout=10)
            pages = r.json().get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid == "-1":
                    continue
                extract = pdata.get("extract", "")
                paragraphs = [p.strip() for p in extract.split("\n") if len(p.strip()) > 70]
                return "\n\n".join(paragraphs)
            return ""
        except Exception:
            return ""

    def translate_to_ru(self, text: str) -> str:
        """Быстрый пакетный перевод блоками по 4000 символов."""
        paragraphs = text.split("\n\n")
        batches = []
        current_batch = []
        current_len = 0

        for p in paragraphs:
            if not p.strip():
                continue
            if current_len + len(p) > 3800:
                batches.append("\n###\n".join(current_batch))
                current_batch = [p]
                current_len = len(p)
            else:
                current_batch.append(p)
                current_len += len(p) + 5

        if current_batch:
            batches.append("\n###\n".join(current_batch))

        translated_full = []
        for batch in batches:
            try:
                res = self.translator.translate(batch)
                translated_full.extend(res.split("###"))
                time.sleep(0.1)
            except Exception:
                translated_full.extend(batch.split("###"))

        return "\n\n".join(p.strip() for p in translated_full if p.strip())

    def is_valid_chess_article(self, title: str) -> bool:
        t_low = title.lower()
        if any(bad in t_low for bad in EXCLUDE_KEYWORDS):
            return False
        if "," in title and not any(k in t_low for k in ["защита", "гамбит", "атака", "вариант"]):
            return False
        return True

    def run(self):
        print(f"=== Запуск рекурсивного краулера (Цель: {self.limit} статей, глубина: {self.max_depth}) ===")
        
        # Перезапись файла в начале новой сборки
        with open(OUT_FILE, "w", encoding="utf-8") as out_f:
            for root_info in ROOT_CATEGORIES:
                if self.collected_count >= self.limit:
                    break

                lang = root_info["lang"]
                root_cat = root_info["category"]
                stage = root_info["default_stage"]

                queue: List[Tuple[str, int]] = [(root_cat, 0)]
                self.visited_categories.add(root_cat)

                while queue and self.collected_count < self.limit:
                    current_cat, depth = queue.pop(0)
                    print(f"\n[КРАУЛИНГ] {current_cat} (Глубина {depth}/{self.max_depth}, {lang.upper()})...")

                    members = self.fetch_category_members(lang, current_cat)

                    for item in members:
                        if self.collected_count >= self.limit:
                            break

                        ns = item.get("ns")
                        title = item.get("title", "")

                        if ns == 14 and depth < self.max_depth:
                            if title not in self.visited_categories and self.is_valid_chess_article(title):
                                self.visited_categories.add(title)
                                queue.append((title, depth + 1))

                        elif ns == 0 and title not in self.visited_titles:
                            self.visited_titles.add(title)
                            if not self.is_valid_chess_article(title):
                                continue

                            print(f"-> [{self.collected_count + 1}/{self.limit}] Сбор: {title}...")
                            text = self.fetch_article_text(lang, title)

                            if not text or len(text) < 140:
                                continue

                            if lang == "en":
                                print(f"   (Перевод на русский...)")
                                text = self.translate_to_ru(text)

                            header = f"[ТЕМА: {title.upper()}] [СТАДИЯ: {stage}] [АВТОР: Chess Theory] [КНИГА: {title}]\n"
                            full_entry = f"\n\n=== NEW_ARTICLE ===\n\n{header}{text}"
                            out_f.write(full_entry)
                            out_f.flush()
                            self.collected_count += 1

        print(f"\n[ГОТОВО] Сбор завершен! Всего добавлено уникальных статей: {self.collected_count}")

if __name__ == "__main__":
    crawler = WikiChessCrawler(target_articles_limit=350, max_depth=3)
    crawler.run()