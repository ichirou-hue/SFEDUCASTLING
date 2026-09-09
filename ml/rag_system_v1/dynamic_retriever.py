import os
import sys
import hashlib
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knowledge_base.parser_chunker import get_chroma_collection

class DynamicChessRetriever:
    def __init__(self):
        self.collection = get_chroma_collection(reset=False)

    def build_smart_query(
        self,
        mistake_type: str,
        best_move_desc: str,
        played_move_desc: str,
        stage: str
    ) -> str:
        terms = []
        if mistake_type:
            terms.extend([mistake_type, mistake_type])
        if played_move_desc:
            terms.append(f"ошибка {played_move_desc}")
        if best_move_desc:
            terms.append(f"стратегия {best_move_desc}")
        terms.append(f"шахматные принципы {stage}")
        return " ".join(terms)

    def retrieve_context(
        self,
        query_text: str,
        stage: Optional[str] = None,
        n_results: int = 4
    ) -> List[Dict[str, Any]]:
        where_clause = {"stage": stage} if stage in ["opening", "endgame"] else None
        fetch_k = n_results * 2

        try:
            if where_clause:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=fetch_k,
                    where=where_clause
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=fetch_k
                )
        except Exception as e:
            print(f"[WARN] Ошибка поиска с фильтром: {e}")
            results = None

        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []

        if len(docs) < 2 and where_clause is not None:
            fallback_res = self.collection.query(
                query_texts=[query_text],
                n_results=fetch_k
            )
            docs = fallback_res.get("documents", [[]])[0]
            metas = fallback_res.get("metadatas", [[]])[0]

        extracted = []
        seen_texts = set()

        for d, m in zip(docs, metas):
            norm_text = " ".join(d.strip().split())
            text_sig = hashlib.md5(norm_text[:90].encode('utf-8')).hexdigest()

            if text_sig in seen_texts:
                continue

            seen_texts.add(text_sig)
            extracted.append({
                "text": d,
                "author": m.get("author", "Шахматная классика"),
                "book": m.get("book", "Теория шахмат"),
                "concept": m.get("concept", "Позиционная игра"),
                "stage": m.get("stage", "middlegame")
            })

            if len(extracted) == n_results:
                break

        return extracted


def build_llm_rag_prompt(
    fen: str,
    sf_eval: float,
    best_move_sf: str,
    played_move_human: str,
    eval_after: float,
    query_ctx: str = "",
    stage: str = "middlegame",
    mistake_type: str = "позиционная ошибка"
) -> str:
    retriever = DynamicChessRetriever()
    
    if not query_ctx:
        smart_query = retriever.build_smart_query(
            mistake_type=mistake_type,
            best_move_desc=best_move_sf,
            played_move_desc=played_move_human,
            stage=stage
        )
    else:
        smart_query = query_ctx

    found_chunks = retriever.retrieve_context(
        query_text=smart_query,
        stage=stage,
        n_results=4
    )

    eval_loss = abs(sf_eval - eval_after)
    turn = "Белые" if " w " in fen else "Черные"

    context_blocks = []
    for idx, c in enumerate(found_chunks, 1):
        block = (
            f"[{idx}] {c['author']} — «{c['book']}» (Тема: {c['concept'].upper()}):\n"
            f"    \"{c['text']}\""
        )
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks) if context_blocks else "Общие принципы гармонии фигур и безопасности короля."

    prompt = f"""Ты — профессиональный шахматный тренер. Твоя задача — объяснить позицию ученику, связывая факты расчета движка с правилами классической литературы.

=== 1. ФАКТИЧЕСКИЕ ДАННЫЕ (STOCKFISH) ===
- Позиция (FEN): {fen}
- Очередь хода: {turn} | Стадия: {stage}
- Текущая объективная оценка: {sf_eval:+.2f} пешек
- Сильнейший ход Stockfish: {best_move_sf}
- Ход игрока: {played_move_human} (Потеря: {eval_loss:.2f} пешек, квалификация: {mistake_type})

=== 2. НАЙДЕННЫЙ КОНТЕКСТ ИЗ ШАХМАТНОЙ ЛИТЕРАТУРЫ (RAG) ===
{formatted_context}

=== 3. ИНСТРУКЦИЯ ДЛЯ ОТВЕТА ===
1. Весь ответ строго на русском языке.
2. В первой строке напиши строго: "Оценка: {sf_eval:+.2f}".
3. Объясни, почему ход Stockfish ({best_move_sf}) объективно сильнейший в этой позиции.
4. Разбери ошибку игрока ({played_move_human}): свяжи её с приведенными правилами из книг (почему это нарушает фундаментальные принципы).
5. Запрещено придумывать несуществующие ходы. Излагай четко, доходчиво и структурированно."""

    return prompt