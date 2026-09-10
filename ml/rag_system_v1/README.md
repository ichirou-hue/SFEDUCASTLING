# RAG System v1 📚

Retrieval-Augmented Generation для шахматного тренера: собирает знания из шахматной литературы и статей, чанкит их в векторную базу **ChromaDB** и подмешивает релевантный контекст в промпты LLM.

---

## 📂 Структура

| Путь | Назначение |
|------|-----------|
| `parser_chunker.py` | Классификатор-чанкер: PDF/текст → чанки → ChromaDB (коллекция `chess_knowledge_base`). |
| `dynamic_retriever.py` | `DynamicChessRetriever` — умный запрос и поиск по стадии игры (opening/middlegame/endgame). |
| `chess_analysis/engine_analyzer.py` | `ChessEngineAnalyzer` — анализ позиции через python-chess + Stockfish (глубина 18). |
| `chess_analysis/prompt_builder.py` | `ChessPromptBuilder` — сборка строго структурированных RAG-промптов с few-shot защитой. |
| `parsers/auto_crawler.py` | Краулер категорий Wikipedia (RU/EN) по шахматной теории → `data/parsed_texts/articles_parsed.txt`. |
| `parsers/vlm_parser.py` | VLM-парсер (Qwen2-VL-2B) книг в PDF → чистый текст (`tarrasch_vlm_clean.txt`). |
| `data/raw_books/` | Исходные PDF-книги (например «300 шахматных партий» Тарраша). |
| `data/parsed_texts/` | Распарсенные тексты (OCR-сырьё и VLM-чистые версии). |
| `test_rag.py` | Тест на **20 верифицированных позициях** (дебют/миттельшпиль/эндшпиль) с эталонными ответами Stockfish. |
| `test_batch_coach.py` | Пакетная проверка тренерских ответов через облачную модель. |

---

## 🔧 Как устроен pipeline

```
     сбор источников                     индексация                     генерация
┌──────────────────────┐      ┌───────────────────────┐      ┌───────────────────────────┐
│ PDF-книги ──VLM/OCR──┤      │                       │      │                           │
│ Wikipedia ──краулер──┼─────▶│  parser_chunker.py    │─────▶│  DynamicChessRetriever    │
│ автостатьи           │      │  (чанки + тысячи стадий│      │  → релевантные чанки      │
└──────────────────────┘      │   → ChromaDB)          │      │  ChessPromptBuilder       │
                              └───────────────────────┘      │  → промпт для LLM          │
                                                       └──────┴───────────────────────────┘
```

1. **Парсинг**: книги распознаются через VLM (`Qwen2-VL-2B`) или OCR, тексты по шахматной теории собираются краулером из Wikipedia.
2. **Чанкинг и индексация**: `ChessChunker` разбивает текст на смысловые фрагменты, классифицирует по стадии игры и складывает в ChromaDB с эмбеддерами `paraphrase-multilingual-MiniLM-L12-v2`.
3. **Ретрив**: `DynamicChessRetriever.build_smart_query()` собирает запрос из типа ошибки, описаний ходов и стадии; `retrieve_context()` фильтрует по `stage` (для opening/endgame) и возвращает top-k чанков.
4. **Промпт**: `ChessPromptBuilder` объединяет анализ движка (Stockfish), извлечённый контекст и вопрос ученика в структурированный промпт для LLM.

---

## 🚀 Запуск

```bash
pip install -r requirements.txt

# Построить индексы из data/parsed_texts
python parser_chunker.py

# Проверить RAG на 20 верифицированных позициях
python test_rag.py

# Пакетная проверка тренера через облачную модель
python test_batch_coach.py
```

> Путь к Stockfish настраивается переменной окружения `STOCKFISH_PATH` (по умолчанию `stockfish` — бинарник должен быть в PATH).

---

## ⚠️ Особенности импортов

Модули ссылаются на пакет `knowledge_base` (например `from knowledge_base.dynamic_retriever import build_llm_rag_prompt`). Для этого в начале каждого файла выполняется вставка базовой директории в `sys.path`. При переносе кода убедитесь, что `BASE_DIR` указывает на корень, где доступен пакет `knowledge_base` (в текущей схеме — `ml/`).