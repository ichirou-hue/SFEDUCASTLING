from typing import Dict, Any, Union, Optional
from src.schemas import AnalyzedPosition

SYSTEM_PROMPT = """Ты — элитный гроссмейстер и опытный шахматный тренер.
Твоя задача — составить методически глубокий, фактологически точный и понятный разбор позиции для ученика.

ПРИНЦИПЫ:
1. Строго опирайся на переданную геометрию доски (цвет фигур, висячие фигуры, связки) и расчеты движка.
2. Не придумывай фигуры, пешки и ходы, которых нет на доске или в линиях варианта.
3. Объясняй не просто ходы, а шахматные мотивы: перегрузка, связка, слабый король, отвлечение, слабость полей.
4. Ответ СТРОГО должен соответствовать JSON-схеме CoachExplanation.
"""

# Псевдоним для полной обратной совместимости
SYSTEM_COACH_PROMPT = SYSTEM_PROMPT


def build_user_prompt(
    fen: Optional[Union[str, AnalyzedPosition]] = None,
    engine_data: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs
) -> str:
    """
    Универсальная функция сборки промпта.
    Поддерживает:
      - Именованные параметры: build_user_prompt(fen=fen, engine_data=data)
      - Именованный объект: build_user_prompt(fen_or_pos=pos)
      - Позиционные аргументы: build_user_prompt(fen, engine_data) или build_user_prompt(pos)
    """
    pos_or_fen = fen if fen is not None else kwargs.get("fen_or_pos")
    if pos_or_fen is None and args:
        pos_or_fen = args[0]
        if len(args) > 1 and engine_data is None:
            engine_data = args[1]

    if isinstance(pos_or_fen, AnalyzedPosition):
        pos = pos_or_fen
        fen_str = pos.fen
        played_move = pos.played_move
        best_move = pos.best_move
        centipawn_loss = pos.centipawn_loss
        move_number = pos.move_number
        phase = pos.phase
        turn_val = pos.turn
        multipv = pos.multipv
        facts = engine_data.get("board_facts", {}) if engine_data else {}
    else:
        fen_str = str(pos_or_fen) if pos_or_fen else ""
        data = engine_data or {}
        played_move = data.get("played_move", "")
        best_move = data.get("best_move", "")
        centipawn_loss = data.get("centipawn_loss", 0)
        move_number = data.get("move_number", 1)
        phase = data.get("phase", "middlegame")
        turn_val = data.get("turn", "white")
        multipv = data.get("multipv", [])
        facts = data.get("board_facts", {})

    # Форматирование расчетных линий MultiPV Stockfish
    multipv_text = ""
    for line in multipv:
        if isinstance(line, dict):
            rank = line.get("rank", 1)
            move_san = line.get("move_san", "")
            eval_cp = line.get("eval_cp")
            eval_mate = line.get("eval_mate")
            pv_san = line.get("pv_san", [])
        else:
            rank = getattr(line, "rank", 1)
            move_san = getattr(line, "move_san", "")
            eval_cp = getattr(line, "eval_cp", None)
            eval_mate = getattr(line, "eval_mate", None)
            pv_san = getattr(line, "pv_san", [])

        eval_str = f"{eval_cp / 100:+.2f}" if eval_cp is not None else f"Mate in {eval_mate}"
        pv_moves = " -> ".join(pv_san)
        multipv_text += f"\n  - Линия {rank} ({move_san}, оценка: {eval_str}): {pv_moves}"

    hanging_str = "; ".join(facts.get("hanging_pieces", [])) if facts.get("hanging_pieces") else "нет"
    pins_str = "; ".join(facts.get("pins", [])) if facts.get("pins") else "нет"
    effects_str = "; ".join(facts.get("best_move_effects", [])) if facts.get("best_move_effects") else "позиционное усиление"

    turn_ru = "Белые" if str(turn_val).lower() == "white" else "Черные"

    return f"""Разбери позицию:
- FEN: {fen_str}
- Очередь хода: {turn_ru} (Ход №{move_number}, стадия: {phase})
- Сделанный ход в партии: {played_move} (Потеря: {centipawn_loss} cp)
- Рекомендуемый ход движка: {best_move}
- Геометрические факты позиции:
  * Незащищенные фигуры: {hanging_str}
  * Связки на доске: {pins_str}
  * Прямые действия и угрозы лучшего хода ({best_move}): {effects_str}
- Расчетные линии Stockfish (MultiPV):{multipv_text}

Сформируй понятный тренерский разбор в формате JSON по схеме CoachExplanation."""