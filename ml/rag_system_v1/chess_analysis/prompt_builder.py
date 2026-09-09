import os
import sys
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knowledge_base.dynamic_retriever import DynamicChessRetriever
from knowledge_base.chess_analysis.engine_analyzer import ChessEngineAnalyzer


class ChessPromptBuilder:
    """Генератор строго структурированных RAG-промптов с Few-Shot защитой."""

    def __init__(self, engine_analyzer: Optional[ChessEngineAnalyzer] = None):
        self.analyzer = engine_analyzer or ChessEngineAnalyzer()
        self.retriever = DynamicChessRetriever()

    def build_prompt_from_fen(
        self,
        fen: str,
        played_uci_move: Optional[str] = None,
        student_question: Optional[str] = None
    ) -> Dict[str, Any]:
        analysis = self.analyzer.evaluate_position(fen, played_uci_move)

        found_chunks = self.retriever.retrieve_context(
            query_text=analysis["rag_query"],
            stage=analysis["stage"],
            n_results=4
        )

        context_blocks = []
        for idx, c in enumerate(found_chunks, 1):
            context_blocks.append(
                f"[{idx}] {c['author']} — «{c['book']}»:\n"
                f"    \"{c['text']}\""
            )
        formatted_context = "\n\n".join(context_blocks) if context_blocks else "Принципы активности фигур и безопасности короля."

        q_block = f"Вопрос ученика: \"{student_question}\"\n\n" if student_question else ""

        prompt_text = f"""Ты — профессиональный шахматный тренер. Твоя задача — объяснить позицию ученику строго по фактам и цитатам из теории.

=== ЭТАЛОННЫЙ ПРИМЕР ОТВЕТА (ФОРМАТ) ===
Оценка: +1.50

1. Сильнейший ход: Слон с c4 берет на f7 (Bxf7+)
Идея хода: Белые наносят тактический удар по незащищенному пункту f7, разрушая прикрытие короля черных и лишая его права на рокировку. Это дает белым решающую инициативу при застрявшем в центре короле.

2. Разбор хода игрока: Пешка с d2 идет на d4 (d4)
Ошибка хода: Ход является позиционным промедлением. Белые упускают конкретную тактическую возможность вскрыть оборону соперника, позволяя черным завершить развитие и укрепить центр.
========================================

{q_block}=== ИСХОДНЫЕ ДАННЫЕ ДЛЯ АНАЛИЗА ===
- Позиция (FEN): {analysis['fen']}
- Очередь хода: {analysis['turn']} | Стадия: {analysis['stage']}
- Объективная оценка: {analysis['eval_before']:+.2f} пешек
- ХОД А (Сильнейший ход движка): {analysis['best_move_verbal']}
- ХОД Б (Сделанный ход игрока): {analysis['played_move_verbal']} (Потеря: {analysis['eval_loss']:.2f} пешек, квалификация: {analysis['mistake_type']})

=== ТЕОРЕТИЧЕСКАЯ БАЗА (RAG) ===
{formatted_context}

=== ИНСТРУКЦИЯ (СТРОГО СОБЛЮДАЙ ПРАВИЛА) ===
1. Первая строка строго: "Оценка: {analysis['eval_before']:+.2f}"
2. В блоке "1. Сильнейший ход:" скопируй название "ХОД А" без изменений и объясни его суть.
3. В блоке "2. Разбор хода игрока:" скопируй название "ХОД Б" без изменений и объясни, почему он плох, опираясь на правила из теории.
4. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать другие клетки, выдумывать ходы или менять нотацию. Используй только данные из раздела ИСХОДНЫЕ ДАННЫЕ."""

        return {
            "prompt": prompt_text,
            "analysis": analysis,
            "rag_context": found_chunks
        }