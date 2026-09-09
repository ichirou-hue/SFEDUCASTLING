from typing import Any, Dict, Optional


class NagClassifier:

  @staticmethod
  def classify(
      cp_loss: Optional[int],
      is_best_move: bool,
      is_sacrifice: bool = False,
      prev_score_cp: Optional[int] = None,
      mate_missed: bool = False,
  ) -> Dict[str, Any]:
    """Классификация качества хода по стандартам Lichess / FIDE:

    - brilliant (!!)
    - best (★)
    - great (!)
    - excellent
    - good
    - inaccuracy (?!)
    - mistake (?)
    - blunder (??)
    """
    # 1. Если был упущен прямой мат
    if mate_missed:
      return {
          "glyph": "??",
          "nag_code": "$4",
          "category": "blunder",
          "label": "Грубый зевок (Упущен мат)",
      }

    # 2. Если нет числовой оценки (начало партии / дебютная книга)
    if cp_loss is None:
      return {
          "glyph": "",
          "nag_code": "$0",
          "category": "book",
          "label": "Теория / Без оценки",
      }

    # 3. Бриллиантовый (!!) — лучший ход движка с корректной жертвой материала
    if is_best_move and is_sacrifice:
      return {
          "glyph": "!!",
          "nag_code": "$3",
          "category": "brilliant",
          "label": "Бриллиантовый ход",
      }

    # 4. Лучший ход движка (★)
    if is_best_move or cp_loss == 0:
      return {
          "glyph": "★",
          "nag_code": "$1",
          "category": "best",
          "label": "Лучший ход",
      }

    # 5. Отличный ход (!)
    if cp_loss <= 15:
      return {
          "glyph": "!",
          "nag_code": "$1",
          "category": "great",
          "label": "Отличный ход",
      }

    # 6. Превосходный ход
    if cp_loss <= 30:
      return {
          "glyph": "",
          "nag_code": "$0",
          "category": "excellent",
          "label": "Превосходный ход",
      }

    # 7. Хороший ход
    if cp_loss <= 60:
      return {
          "glyph": "",
          "nag_code": "$0",
          "category": "good",
          "label": "Хороший ход",
      }

    # 8. Неточность (?!) — потеря от 61 до 150 сантипешек
    if 60 < cp_loss <= 150:
      return {
          "glyph": "?!",
          "nag_code": "$6",
          "category": "inaccuracy",
          "label": "Неточность",
      }

    # 9. Ошибка (?) — потеря от 151 до 300 сантипешек (например, упущен тактический удар)
    if 150 < cp_loss <= 300:
      return {
          "glyph": "?",
          "nag_code": "$2",
          "category": "mistake",
          "label": "Ошибка",
      }

    # 10. Грубый зевок (??) — потеря более 300 сантипешек (3 полных пешек/фигуры)
    return {
        "glyph": "??",
        "nag_code": "$4",
        "category": "blunder",
        "label": "Грубый зевок",
    }