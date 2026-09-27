import re

from backend.llm.explainer.grounding import _build_gigachess_grounding
from backend.llm.explainer.names import (
    _piece_name_accusative,
    _piece_word_accusative,
)
from backend.llm.explainer.prompts import (
    _compact_retry_focus,
    _render_compact_prompt,
)




def _render_protected_castling_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Repair рокировки с защищёнными слотами.

    Модель отвечает только за естественную русскую формулировку.
    Реальные координаты ей не передаются и подставляются Python после ответа.
    """
    side = "короткая" if move_facts.get("to") in {"g1", "g8"} else "длинная"

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй естественное короткое объяснение уже выполненной {side} рокировки.

Проверенная механика:
- Король переходит с <KING_FROM> на <KING_TO>.
- Ладья одновременно переходит с <ROOK_FROM> на <ROOK_TO>.
- Это уже выполненная {side} рокировка.
- В этом ходе нет взятия, шаха или мата.

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Сохрани маркеры <KING_FROM>, <KING_TO>, <ROOK_FROM>, <ROOK_TO> ТОЧНО в таком виде.
- Каждый из четырёх маркеров используй ровно один раз.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши вообще никаких других координат вида a1-h8.
- Не добавляй безопасность, защиту, развитие, активизацию, угрозы или стратегическую пользу.
- Не используй QB2, KR2, KB1 и подобную описательную нотацию.
- Прямо скажи, что это рокировка, уже выполняемая данным ходом.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    # В strict repair FEN намеренно отсутствует:
    # позицию уже разобрал Python, повторный анализ модели не нужен.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_castling_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    grounding: dict,
) -> tuple[str, list[str]]:
    """
    Проверяет защищённые слоты и только после этого подставляет координаты.
    """
    errors: list[str] = []

    slot_values = {
        "<KING_FROM>": str(explained_move.get("from") or "").lower(),
        "<KING_TO>": str(explained_move.get("to") or "").lower(),
        "<ROOK_FROM>": str(grounding.get("castling_rook_from") or "").lower(),
        "<ROOK_TO>": str(grounding.get("castling_rook_to") or "").lower(),
    }

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)
        if count != 1:
            errors.append(
                f"Защищённый маркер {slot} должен встречаться ровно один раз; "
                f"получено: {count}."
            )
        if not value:
            errors.append(
                f"Для маркера {slot} отсутствует проверенное значение."
            )

    # В raw-ответе вообще не должно быть шахматных координат:
    # только защищённые слоты.
    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected repair модель самостоятельно назвала координаты "
            f"{raw_squares}; координаты должны приходить только из Python-слотов."
        )

    descriptive = re.findall(
        r"(?<![A-Za-zА-Яа-я0-9])[KQRBN]{1,3}[1-8]"
        r"(?![A-Za-zА-Яа-я0-9])",
        raw_answer,
    )
    if descriptive:
        errors.append(
            "В protected repair использована неподдерживаемая описательная "
            f"нотация: {sorted(set(descriptive))}."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        materialized = materialized.replace(slot, value)

    return materialized, []




def _render_protected_capture_promotion_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Жёсткий repair для комбинации:
        capture + promotion [+ check | mate]

    Вместо множества мелких координатных slots модель получает
    несколько атомарных смысловых маркеров. Python позже заменяет
    каждый маркер целиком проверенным событием.

    GigaChess отвечает только за связность, порядок и естественный
    русский текст между событиями.
    """
    required_events = [
        "<CAPTURE_EVENT>",
        "<PROMOTION_EVENT>",
    ]

    if move_facts.get("is_checkmate"):
        required_events.append("<MATE_EVENT>")
    elif move_facts.get("is_check"):
        required_events.append("<CHECK_EVENT>")

    events_line = " → ".join(required_events)

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй короткое естественное объяснение одного шахматного хода.

Этот ход состоит из уже совершившихся событий.
Используй protected-маркеры строго в таком порядке:

{events_line}

Предыдущая генерация была отклонена:
{focus_text}

Что означают маркеры:
- <CAPTURE_EVENT> — уже выполненное взятие проверенной фигуры.
- <PROMOTION_EVENT> — уже выполненное превращение пешки.
- <CHECK_EVENT> — уже объявленный этим ходом шах.
- <MATE_EVENT> — уже поставленный этим ходом мат.

Жёсткие правила:
- Используй каждый маркер из указанной последовательности РОВНО один раз.
- Не пропускай ни один обязательный маркер.
- Не добавляй другие маркеры.
- Не заменяй маркеры своими названиями фигур или координатами.
- Не пиши собственные шахматные координаты вида a1-h8.
- Сохрани порядок событий: взятие → превращение → шах/мат.
- Можно добавлять только естественные связующие слова и пунктуацию.
- Не добавляй другие фигуры, взятия, угрозы, сдачу, ничью,
  победу, дебют или стратегические оценки.
- Верни 1-2 коротких предложения.
- Верни только объяснение.

Пример допустимой структуры без раскрытия фактов:
«Пешка <CAPTURE_EVENT> и <PROMOTION_EVENT>. После этого <CHECK_EVENT>.»
Используй только те маркеры, которые перечислены в обязательной
последовательности выше.
""".strip()

    # Никакого FEN и никаких реальных фигур/координат.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_capture_promotion_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
    derived_facts: dict,
) -> tuple[str, list[str]]:
    """
    Материализует атомарные события capture+promotion.
    """
    errors: list[str] = []

    from_square = str(
        explained_move.get("from") or ""
    ).lower()
    to_square = str(
        explained_move.get("to") or ""
    ).lower()

    captured_piece = str(
        move_facts.get("captured_piece") or ""
    )
    captured_acc = _piece_name_accusative(
        captured_piece
    )
    captured_square = str(
        move_facts.get("captured_square")
        or explained_move.get("to")
        or ""
    ).lower()

    promotion_piece = str(
        move_facts.get("promotion_piece") or ""
    ).lower()
    promotion_acc = _piece_word_accusative(
        promotion_piece
    )

    terminal = derived_facts.get("terminal") or {}
    king_square = str(
        terminal.get("opponent_king_square") or ""
    ).lower()

    event_values = {
        "<CAPTURE_EVENT>": (
            f"переходит с {from_square} на {to_square} и берёт "
            f"{captured_acc} на {captured_square}"
        ),
        "<PROMOTION_EVENT>": (
            f"на {to_square} превращается в {promotion_acc}"
        ),
    }

    required_events = [
        "<CAPTURE_EVENT>",
        "<PROMOTION_EVENT>",
    ]

    if move_facts.get("is_checkmate"):
        event_values["<MATE_EVENT>"] = (
            f"после превращения {promotion_piece} на {to_square} "
            f"ставит мат королю соперника на {king_square}"
        )
        required_events.append("<MATE_EVENT>")

    elif move_facts.get("is_check"):
        event_values["<CHECK_EVENT>"] = (
            f"после превращения {promotion_piece} на {to_square} "
            f"объявляет шах королю соперника на {king_square}"
        )
        required_events.append("<CHECK_EVENT>")

    # Каждый обязательный атомарный event — ровно один раз.
    for event in required_events:
        count = raw_answer.count(event)
        if count != 1:
            errors.append(
                f"Protected event {event} должен встречаться "
                f"ровно один раз; получено: {count}."
            )

    known_events = set(event_values)
    unknown_events = sorted(
        set(re.findall(r"<[A-Z_]+>", raw_answer))
        - known_events
    )
    if unknown_events:
        errors.append(
            "В capture-promotion repair использованы неизвестные "
            f"маркеры: {unknown_events}."
        )

    # Проверяем порядок маркеров.
    present_positions = [
        raw_answer.find(event)
        for event in required_events
        if raw_answer.find(event) >= 0
    ]
    if (
        len(present_positions) == len(required_events)
        and present_positions != sorted(present_positions)
    ):
        errors.append(
            "Нарушен порядок событий: сначала взятие, затем "
            "превращение, затем шах/мат."
        )

    # В raw-ответе модель не имеет права самостоятельно писать клетки.
    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected capture-promotion repair модель самостоятельно "
            f"назвала координаты {raw_squares}; координаты должен "
            "подставлять только Python."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for event, value in event_values.items():
        materialized = materialized.replace(
            event,
            value,
        )

    return materialized, []


def _render_protected_promotion_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
    derived_facts: dict,
) -> list[dict]:
    """
    Repair превращения с защищёнными слотами.

    GigaChess отвечает только за русский текст. Координаты и тип новой
    фигуры материализуются Python после генерации.
    """
    promotion_piece = str(
        move_facts.get("promotion_piece") or "фигура"
    ).lower()

    gives_mate = bool(move_facts.get("is_checkmate"))
    gives_check = bool(move_facts.get("is_check"))
    is_capture = bool(move_facts.get("is_capture"))

    event_lines = []

    if is_capture:
        event_lines.append(
            "- Пешка переходит с <PAWN_FROM> на <PROMOTION_SQUARE> "
            "и этим же ходом берёт <CAPTURED_PIECE_ACC> "
            "на <CAPTURE_SQUARE>."
        )
    else:
        event_lines.append(
            "- Пешка переходит с <PAWN_FROM> на <PROMOTION_SQUARE>."
        )

    event_lines.append(
        "- На <PROMOTION_SQUARE_EVENT> она превращается в "
        "<PROMOTED_PIECE_ACC>."
    )

    if gives_mate:
        event_lines.append(
            "- После превращения <PROMOTED_PIECE_NOM> на "
            "<CHECKER_SQUARE> ставит мат королю соперника на "
            "<KING_SQUARE>."
        )
    elif gives_check:
        event_lines.append(
            "- После превращения <PROMOTED_PIECE_NOM> на "
            "<CHECKER_SQUARE> объявляет шах королю соперника на "
            "<KING_SQUARE>."
        )

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    events_text = "\n".join(event_lines)

    capture_rules = ""
    if is_capture:
        capture_rules = """
- В этом ходе ОБЯЗАТЕЛЬНО опиши уже совершившееся взятие.
- Сохрани <CAPTURED_PIECE_ACC> и <CAPTURE_SQUARE> ТОЧНО.
- Каждый из этих двух маркеров используй ровно один раз.
- Не опускай взятую фигуру: превращение произошло через взятие.
""".strip()

    user = f"""
Сформулируй естественное короткое объяснение уже выполненного превращения пешки.

Проверенная механика:
{events_text}

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Координаты в итоговой фразе НЕ обязательны, кроме обязательных protected-маркеров,
  если они перечислены ниже.
- Если используешь какой-либо маркер в угловых скобках, сохрани его ТОЧНО.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши собственные координаты вида a1-h8.
{capture_rules}
- Можно кратко сказать, что пешка превратилась в проверенную новую фигуру
  и этим же ходом дала шах/мат, если это указано в механике.
- Превращение уже происходит этим ходом, а не готовится на будущее.
- После превращения пешка больше не является пешкой; дальнейшие действия
  совершает новая фигура.
- Если указан шах или мат, он уже возникает этим ходом, а не является угрозой.
- Не говори о сдаче, капитуляции, ничьей, победе или окончании партии,
  если этого нет в проверенной механике.
- Не добавляй стратегическую пользу, центр, развитие, защиту или угрозы.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_promotion_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
    derived_facts: dict,
) -> tuple[str, list[str]]:
    """
    Проверяет protected slots и подставляет только проверенные Python-факты.
    """
    errors: list[str] = []

    promotion_piece = str(
        move_facts.get("promotion_piece") or ""
    ).lower()
    promotion_piece_acc = _piece_word_accusative(
        promotion_piece
    )

    terminal = derived_facts.get("terminal") or {}
    king_square = str(
        terminal.get("opponent_king_square") or ""
    ).lower()

    slot_values = {
        "<PAWN_FROM>": str(explained_move.get("from") or "").lower(),
        "<PROMOTION_SQUARE>": str(explained_move.get("to") or "").lower(),
        "<PROMOTION_SQUARE_EVENT>": str(explained_move.get("to") or "").lower(),
        "<PROMOTED_PIECE_ACC>": promotion_piece_acc,
    }

    mandatory_slots: set[str] = set()

    if move_facts.get("is_capture"):
        captured_piece = str(
            move_facts.get("captured_piece") or ""
        )
        captured_square = str(
            move_facts.get("captured_square")
            or explained_move.get("to")
            or ""
        ).lower()

        slot_values.update({
            "<CAPTURED_PIECE_ACC>": _piece_name_accusative(
                captured_piece
            ),
            "<CAPTURE_SQUARE>": captured_square,
        })
        mandatory_slots.update({
            "<CAPTURED_PIECE_ACC>",
            "<CAPTURE_SQUARE>",
        })

    if move_facts.get("is_check") or move_facts.get("is_checkmate"):
        slot_values.update({
            "<PROMOTED_PIECE_NOM>": promotion_piece,
            "<CHECKER_SQUARE>": str(explained_move.get("to") or "").lower(),
            "<KING_SQUARE>": king_square,
        })

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)

        if count > 1:
            errors.append(
                f"Защищённый маркер {slot} нельзя дублировать; "
                f"получено вхождений: {count}."
            )

        if slot in mandatory_slots and count != 1:
            errors.append(
                f"Для превращения со взятием маркер {slot} "
                f"обязателен ровно один раз; получено: {count}."
            )

        if count == 1 and not value:
            errors.append(
                f"Для использованного маркера {slot} "
                "отсутствует проверенное значение."
            )

    known_slots = set(slot_values)
    unknown_slots = sorted(set(
        re.findall(r"<[A-Z_]+>", raw_answer)
    ) - known_slots)

    if unknown_slots:
        errors.append(
            "В protected promotion repair использованы неизвестные "
            f"маркеры: {unknown_slots}."
        )

    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected promotion repair модель самостоятельно назвала "
            f"координаты {raw_squares}; они должны приходить только из Python."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        if slot in materialized:
            materialized = materialized.replace(slot, value)

    return materialized, []



def _render_protected_en_passant_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Repair en passant с защищёнными координатами.

    Модель формулирует только естественный русский текст.
    Три критические клетки материализуются Python:
      - откуда идёт пешка;
      - куда она приходит;
      - откуда снимается чужая пешка.
    """
    mover_side = (
        "белая" if move_facts.get("piece_color") == "white"
        else "чёрная"
    )
    captured_side = (
        "чёрную" if move_facts.get("piece_color") == "white"
        else "белую"
    )

    focus_text = "\\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй короткое естественное объяснение уже выполненного взятия на проходе.

Проверенная механика:
- {mover_side.capitalize()} пешка выполняет взятие на проходе.
- Она переходит с <PAWN_FROM> на <PAWN_TO>.
- Этим же ходом она снимает {captured_side} пешку с <CAPTURED_PAWN_SQUARE>.
- Снятая пешка находится НЕ на поле назначения движущейся пешки.
- Партия после этого хода продолжается.

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Обязательно используй выражение «взятие на проходе».
- Сохрани <PAWN_FROM>, <PAWN_TO>, <CAPTURED_PAWN_SQUARE> ТОЧНО.
- Каждый из этих трёх маркеров используй ровно один раз.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши никаких других координат вида a1-h8.
- Не говори, что партия заканчивается, заканчивается вничью или победой.
- Не описывай взятие как будущую возможность.
- Не добавляй шах, мат, угрозы, дебют или стратегическую пользу.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    # FEN здесь намеренно не нужен: Python уже полностью разобрал механику.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_en_passant_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
) -> tuple[str, list[str]]:
    errors: list[str] = []

    slot_values = {
        "<PAWN_FROM>": str(
            explained_move.get("from") or ""
        ).lower(),
        "<PAWN_TO>": str(
            explained_move.get("to") or ""
        ).lower(),
        "<CAPTURED_PAWN_SQUARE>": str(
            move_facts.get("captured_square") or ""
        ).lower(),
    }

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)

        if count != 1:
            errors.append(
                f"Защищённый маркер {slot} должен встречаться ровно один раз; "
                f"получено: {count}."
            )

        if not value:
            errors.append(
                f"Для маркера {slot} отсутствует проверенное значение."
            )

    unknown_slots = sorted(
        set(re.findall(r"<[A-Z_]+>", raw_answer))
        - set(slot_values)
    )
    if unknown_slots:
        errors.append(
            "В protected en passant repair использованы неизвестные "
            f"маркеры: {unknown_slots}."
        )

    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected en passant repair модель самостоятельно назвала "
            f"координаты {raw_squares}; они должны приходить только из Python."
        )

    if not re.search(
        r"\bвзят\w*\s+на\s+проходе\b",
        raw_answer.lower(),
    ):
        errors.append(
            "Protected en passant repair должен прямо назвать "
            "«взятие на проходе»."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        materialized = materialized.replace(slot, value)

    return materialized, []


def _render_strict_special_repair_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
    retry_focus: list[str],
) -> list[dict]:
    """
    Строгий режим repair для специальных ходов.

    Python уже определил шахматную механику. Модель не анализирует позицию
    заново, а только превращает проверенные отношения в естественный текст.
    """
    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    special_name = "специальный ход"
    relation_rules: list[str] = []

    movement_type = str(
        grounding.get("movement_type")
        or move_facts.get("movement_type")
        or ""
    ).lower()

    movement_geometry_phrase = {
        "diagonal": "по диагонали",
        "horizontal": "по горизонтали",
        "vertical": "по вертикали",
        "knight_jump": "ходом буквой «Г»",
        "king_step": "на соседнюю клетку",
    }.get(movement_type)

    if (
        movement_type == "pawn"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        relation_rules.extend([
            "Ход выполняет именно пешка.",
            "В этом ходе нет взятия.",
            "Пешка идёт прямо вперёд на одну клетку.",
            (
                "Обязательно назови пешку и сохрани формулировку "
                "«прямо вперёд на одну клетку»."
            ),
            (
                "Не упоминай короля соперника, мат, победу, "
                "неизбежность превращения, неудержимость пешки "
                "или возможность/невозможность её догнать, "
                "если этого нет среди обязательных отношений."
            ),
        ])

    if (
        movement_type == "pawn"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        movement_geometry_phrase = "по диагонали вперёд"

    if (
        movement_type == "pawn"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        movement_geometry_phrase = "прямо вперёд на одну клетку"

    if (
        movement_type == "pawn"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        relation_rules.extend([
            "Ход выполняет именно пешка.",
            "Это уже совершившееся взятие.",
            "Пешка берёт фигуру на одну клетку по диагонали вперёд.",
            (
                "Обязательно назови пешку и используй формулировку "
                "«по диагонали вперёд»."
            ),
        ])

    if (
        grounding.get("movement_geometry_required")
        and movement_geometry_phrase
    ):
        relation_rules.extend([
            f"Проверенная траектория хода: {movement_geometry_phrase}.",
            (
                f"Обязательно используй формулировку «{movement_geometry_phrase}» "
                "и не заменяй её другим направлением."
            ),
            (
                "Не определяй траекторию по букве или цифре конечного поля; "
                "она уже вычислена Python по исходной и конечной клетке."
            ),
            (
                "Если проверенный тип — ход коня буквой «Г», не заменяй его "
                "диагональю, горизонталью или вертикалью."
            ),
            (
                "Если проверенный тип — обычный ход короля на соседнюю клетку, "
                "не называй его прыжком или рокировкой."
            ),
        ])

    if move_facts.get("is_castling"):
        special_name = "рокировка"
        king_from = str(explained_move["from"]).lower()
        king_to = str(explained_move["to"]).lower()
        rook_from = grounding.get("castling_rook_from")
        rook_to = grounding.get("castling_rook_to")

        relation_rules.extend([
            f"Король: {king_from} -> {king_to}.",
            f"Ладья: {rook_from} -> {rook_to}.",
            "Это уже выполненная рокировка данным ходом.",
            "Обязательно назови обе фигуры и все четыре координаты.",
            "Не объясняй цель, пользу или безопасность рокировки.",
        ])

    elif move_facts.get("is_checkmate"):
        special_name = "мат"
        relation_rules.extend([
            "Мат уже поставлен именно этим ходом.",
            "Не описывай мат как будущую возможность или угрозу.",
        ])

    elif move_facts.get("is_capture"):
        special_name = "взятие"
        relation_rules.extend([
            "Взятие уже совершено именно этим ходом.",
            "Не пиши «может взять» или другие будущие возможности.",
        ])

    elif move_facts.get("is_check"):
        special_name = "шах"
        relation_rules.extend([
            "Шах уже объявлен именно этим ходом.",
        ])

    elif move_facts.get("is_promotion"):
        special_name = "превращение"
        relation_rules.extend([
            "Превращение уже происходит именно этим ходом.",
        ])

    focus_text = "\n".join(f"- {item}" for item in retry_focus[:3])
    relations_text = "\n".join(f"- {item}" for item in relation_rules)

    user = f"""
Нужно заново сформулировать объяснение хода {explained_move['uci']} для игрока около {elo} Elo.

Режим: СТРОГОЕ ПЕРЕФРАЗИРОВАНИЕ проверенных шахматных отношений.
Позиция FEN намеренно не передаётся: не восстанавливай доску и не анализируй её самостоятельно.
Не анализируй позицию заново и не объясняй, зачем ход полезен.

Проверенные факты:
{grounding['text']}

Обязательные отношения для события «{special_name}»:
{relations_text}

После автоматической проверки предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Сохраняй все названные координаты без изменений.
- Не называй ни одной другой клетки.
- Если среди обязательных отношений указан тип траектории, назови его буквально и не заменяй другим.
- Не называй «вертикаль h», «горизонталь 1» или другую линию доски от себя, если такой формулировки нет в проверенных фактах.
- Не добавляй новые фигуры, угрозы, защиту, безопасность, дебют или стратегическую пользу.
- Не объявляй победу, ничью, сдачу или окончание партии, если этого нет в проверенных фактах.
- Не утверждай, что у соперника нет ходов или что он не может сделать ход, если такого факта нет среди обязательных отношений.
- Не добавляй «выигрывая партию», «принося победу» и похожие выводы, если немедленная победа не дана среди обязательных отношений.
- Не называй ход единственным или единственно возможным: такого проверенного отношения нет.
- Не добавляй прогнозы вроде «мат неизбежен», «пешку нельзя остановить», «король не может догнать пешку» или «превращение гарантировано», если их нет среди обязательных отношений.
- Пиши грамматически корректно: «белая ладья», «белый ферзь», «берёт чёрного ферзя», «берёт чёрную ладью».
- Не оставляй фразу «на поле» или «в клетку» без конкретной координаты; если координата не нужна, убери эту конструкцию целиком.
- Не используй описательную нотацию QB2, KR2, KB1 и подобную.
- Специальное событие происходит уже этим ходом, а не готовится на будущее.
- Для рокировки не заменяй конкретные клетки словами «безопасное поле» или «своя позиция».
- Верни 1-2 коротких естественных предложения.
- Верни только объяснение, без списков и комментариев.
""".strip()

    return [{
        "role": "user",
        "content": user,
    }]


def _build_repair_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
    previous_answer: str | None,
    validation_errors: list[str],
    attempt_number: int,
) -> list[dict]:
    """
    Clean regeneration.

    previous_answer намеренно НЕ используется: неправильный текст модели
    не возвращается ей обратно и не загрязняет следующую генерацию.
    """
    _ = previous_answer

    retry_focus = _compact_retry_focus(validation_errors)

    repair_grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    has_special_event = any([
        move_facts.get("is_capture"),
        move_facts.get("is_check"),
        move_facts.get("is_checkmate"),
        move_facts.get("is_castling"),
        move_facts.get("is_promotion"),
        move_facts.get("is_en_passant"),
    ])

    needs_strict_repair = bool(
        has_special_event
        or repair_grounding.get("movement_geometry_required")
        or repair_grounding.get("pawn_capture_geometry_required")
        or repair_grounding.get("pawn_single_step_geometry_required")
    )

    if move_facts.get("is_castling"):
        return _render_protected_castling_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if move_facts.get("is_en_passant"):
        return _render_protected_en_passant_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if (
        move_facts.get("is_promotion")
        and move_facts.get("is_capture")
    ):
        return _render_protected_capture_promotion_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if move_facts.get("is_promotion"):
        return _render_protected_promotion_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
            derived_facts=derived_facts,
        )

    if needs_strict_repair:
        return _render_strict_special_repair_prompt(
            fen=fen,
            explained_move=explained_move,
            move_facts=move_facts,
            position_facts=position_facts,
            derived_facts=derived_facts,
            elo=elo,
            retry_focus=retry_focus,
        )

    # Для обычных ходов без обязательной геометрии остаётся
    # прежняя clean regeneration.
    if attempt_number == 1:
        retry_focus.append(
            "Начни с корректной фигуры и опиши только смысл проверенного хода."
        )
    elif attempt_number == 2:
        retry_focus.append(
            "Не пиши UCI-запись в итоговом тексте; объясни ход обычными словами."
        )
    else:
        retry_focus.append(
            "Сделай максимально простое объяснение в 1-2 предложениях "
            "и не вводи новых шахматных сущностей."
        )

    return _render_compact_prompt(
        fen=fen,
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
        elo=elo,
        retry_focus=retry_focus[-3:],
    )
