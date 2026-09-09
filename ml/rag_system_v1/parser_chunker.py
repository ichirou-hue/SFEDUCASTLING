import os
import re
import sys
import hashlib
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"[INIT] Запуск классификатора-чанкера. База: {BASE_DIR}")

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARSED_TEXTS_DIR = os.path.join(DATA_DIR, "parsed_texts")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")


def get_chroma_collection(reset: bool = False):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    if reset:
        try:
            client.delete_collection("chess_knowledge_base")
        except Exception:
            pass
    return client.get_or_create_collection(
        name="chess_knowledge_base",
        embedding_function=embed_fn
    )


class ChessChunker:
    STAGE_KEYWORDS = {
        "opening": [
            "дебют", "рокировк", "развити", "гамбит", "темп", "центр", "мобилизац",
            "opening", "development", "castling", "gambit", "tempo", "center control", "fianchetto", "фианкетто"
        ],
        "endgame": [
            "эндшпил", "окончани", "пешечный эндшпиль", "ладейный эндшпиль", "пешечник", "ферзевый эндшпиль",
            "слоновый эндшпиль", "коневой эндшпиль", "правило квадрата", "проходная", "оппозиция", "цугцванг",
            "правило тарраша", "лусена", "филидор", "треугольник", "отталкивание плечом", "пешечный прорыв",
            "endgame", "pawn ending", "rook ending", "lucena", "philidor", "tarrasch rule", "opposition",
            "zugzwang", "passed pawn", "key squares", "triangulation", "shouldering"
        ]
    }

    CONCEPT_RULES = [
        ("борьба за темп и время", [
            "темп", "tempo", "потеря темпа", "выигрыш темпа", "быстрейшее развитие",
            "медлительность", "промедление", "инициатива", "активизация", "задержка"
        ]),
        ("активность ладьи и открытые линии", [
            "ладь", "ладья", "rook", "7-я горизонталь", "седьмая горизонталь", "7th rank", "open file",
            "открытая линия", "полуоткрытая линия", "батарея", "правило тарраша", "сдвоение ладей", "тяжелые фигуры"
        ]),
        ("пешечная структура и слабости", [
            "изолированная пешка", "слабая пешка", "отсталая пешка", "сдвоенные пешки", "пешечный островок",
            "пешечная цепь", "висячие пешки", "пешечные слабости", "база цепи", "подрыв пешек", "pawn structure",
            "isolated pawn", "backward pawn", "doubled pawns", "hanging pawns", "pawn weaknesses", "pawn lever"
        ]),
        ("поля и форпосты", [
            "слабое поле", "форпост", "блокада", "опорный пункт", "дыра в позиции", "пункт d5", "пункт e5", "пункт d4",
            "пункт e4", "вечный конь", "блокадник", "outpost", "weak square", "blockade", "hole", "strongpoint"
        ]),
        ("тактические мотивы и зевки", [
            "связка", "вилка", "двойной удар", "вскрытый шах", "вскрытое нападение", "отвлечение", "завлечение",
            "перегрузка", "рентген", "линейный удар", "капкан", "ловушка", "жертва", "мельница", "промежуточный ход",
            "pin", "fork", "double attack", "discovered attack", "deflection", "decoy", "overloading", "skewer",
            "zwischenzug", "intermediate move", "windmill", "desperado", "бешеная фигура"
        ]),
        ("атака на короля и безопасность", [
            "пешечный штурм", "разрушение прикрытия", "греческий дар", "жертва на h7", "матовая сеть",
            "вскрытие короля", "форточка", "ослабление рокировки", "разносторонние рокировки", "pawn storm",
            "king safety", "mating net", "greek gift", "luft", "attack on the king"
        ]),
        ("фигурные соотношения и размены", [
            "преимущество двух слонов", "плохой слон", "хороший слон", "разноцветные слоны", "конь против слона",
            "размен фигур", "упрощение позиции", "качество", "лишняя фигура", "bishop pair", "bad bishop",
            "opposite-colored bishops", "exchange of pieces", "trade", "simplification"
        ]),
        ("эндшпильная техника и геометрия", [
            "оппозиция", "ключевые поля", "правило квадрата", "построение моста", "защита по 3-й горизонтали",
            "пешечный прорыв", "создание проходной", "отталкивание плечом", "цугцванг", "пат", "крепость",
            "opposition", "key squares", "rule of the square", "bridge building", "third rank defense",
            "breakthrough", "fortress", "stalemate"
        ])
    ]

    @classmethod
    def classify_stage(cls, text: str) -> str:
        t = text.lower()
        for stage, kw_list in cls.STAGE_KEYWORDS.items():
            if any(k in t for k in kw_list):
                return stage
        return "middlegame"

    @classmethod
    def classify_concept(cls, text: str) -> str:
        t = text.lower()
        for concept_name, kw_list in cls.CONCEPT_RULES:
            if any(k in t for k in kw_list):
                return concept_name
        return "позиционная стратегия и принципы"

    @classmethod
    def clean_text(cls, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'[\<\>\:\#\$\%\&\|\@\(\)\{\}\[\]\*\=\\\_\—\–\-\+]{3,}', ' ', text)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @classmethod
    def parse_article_corpus(
        cls,
        file_path: str,
        min_chars: int = 90,
        max_chars: int = 650,
        overlap: int = 80
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            print(f"[ERROR] Файл не найден: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        articles = [a.strip() for a in raw_content.split("=== NEW_ARTICLE ===") if a.strip()]
        chunks = []
        seen_chunk_hashes = set()

        for a_idx, article in enumerate(articles):
            lines = article.split("\n")
            header = lines[0]
            body = "\n".join(lines[1:]).strip()

            topic = "Позиционная игра"
            if "[ТЕМА: " in header:
                topic = header.split("[ТЕМА: ")[1].split("]")[0].strip()

            author = "Шахматная теория"
            if "[АВТОР: " in header:
                author = header.split("[АВТОР: ")[1].split("]")[0].strip()

            book = "Статья"
            if "[КНИГА: " in header:
                book = header.split("[КНИГА: ")[1].split("]")[0].strip()

            cleaned_body = cls.clean_text(body)
            paragraphs = [p.strip() for p in cleaned_body.split("\n\n") if len(p.strip()) >= min_chars]

            for p_idx, para in enumerate(paragraphs):
                if len(para) > max_chars:
                    start = 0
                    sub_idx = 0
                    while start < len(para):
                        end = start + max_chars
                        sub_text = para[start:end].strip()
                        if len(sub_text) >= min_chars:
                            # Проверка на точный или почти точный дубликат чанка
                            ch_hash = hashlib.md5(sub_text[:120].encode('utf-8')).hexdigest()
                            if ch_hash not in seen_chunk_hashes:
                                seen_chunk_hashes.add(ch_hash)
                                stage = cls.classify_stage(sub_text)
                                concept = cls.classify_concept(sub_text)
                                chunks.append({
                                    "id": f"art_{a_idx}_{p_idx}_{sub_idx}_{stage}",
                                    "text": sub_text,
                                    "metadata": {
                                        "stage": stage,
                                        "concept": concept if concept != "позиционная стратегия и принципы" else topic.lower(),
                                        "author": author,
                                        "book": book
                                    }
                                })
                        start += (max_chars - overlap)
                        sub_idx += 1
                else:
                    ch_hash = hashlib.md5(para[:120].encode('utf-8')).hexdigest()
                    if ch_hash not in seen_chunk_hashes:
                        seen_chunk_hashes.add(ch_hash)
                        stage = cls.classify_stage(para)
                        concept = cls.classify_concept(para)
                        chunks.append({
                            "id": f"art_{a_idx}_{p_idx}_{stage}",
                            "text": para,
                            "metadata": {
                                "stage": stage,
                                "concept": concept if concept != "позиционная стратегия и принципы" else topic.lower(),
                                "author": author,
                                "book": book
                            }
                        })

        return chunks


def index_all_parsed_sources(reset_db: bool = True):
    chunker = ChessChunker()
    articles_file = os.path.join(PARSED_TEXTS_DIR, "articles_parsed.txt")
    
    print(f"--- 1. Чтение и парсинг корпуса: {articles_file} ---")
    chunks = chunker.parse_article_corpus(articles_file)

    if not chunks:
        print("[FAIL] Чанки не сформированы. Проверьте articles_parsed.txt")
        return

    print(f"[OK] Сформировано {len(chunks)} уникальных обогащенных чанков.")

    print("\n--- 2. Векторизация и загрузка в ChromaDB ---")
    collection = get_chroma_collection(reset=reset_db)

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    batch_size = 64
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=docs[i:i + batch_size],
            metadatas=metas[i:i + batch_size]
        )
        print(f"-> Векторизован батч {i // batch_size + 1} из {(len(ids) - 1) // batch_size + 1}")

    print(f"\n[SUCCESS] ChromaDB синхронизирована! Всего векторов в базе: {collection.count()}")


if __name__ == "__main__":
    index_all_parsed_sources(reset_db=True)