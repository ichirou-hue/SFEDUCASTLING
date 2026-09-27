import re

from backend.llm.explainer.grounding import _build_gigachess_grounding
from backend.llm.explainer.names import (
    PIECE_NAMES,
    _capture_object_case_patterns,
    _piece_word_accusative,
)


def _is_negated_claim(lowered: str, keyword: str) -> bool:
    patterns = [
        rf"не\s+(?:да[её]т|ставит|является|делает|созда[её]т)\s+[^.!?]{{0,20}}{keyword}",
        rf"{keyword}[^.!?]{{0,8}}нет",
        rf"без\s+{keyword}",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _piece_word_present(
    text: str,
    canonical_piece_word: str,
) -> bool:
    """
    Проверяет русское название фигуры с учётом простых падежных форм:
    ферзь/ферзя, пешка/пешкой, король/короля и т.п.
    """
    patterns = {
        "пешка": r"\bпешк(?:а|и|е|у|ой|ою|ам|ами|ах)\b",
        "конь": r"\bкон(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
        "слон": r"\bслон(?:а|у|ом|е|ы|ов|ам|ами|ах)?\b",
        "ладья": r"\bладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)\b",
        "ферзь": r"\bферз(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
        "король": r"\bкорол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
    }
    pattern = patterns.get(canonical_piece_word)
    if pattern is None:
        return canonical_piece_word in text
    return bool(re.search(pattern, text, re.IGNORECASE))


def _validate_llm_explanation(
    text: str | None,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
) -> tuple[bool, list[str]]:
    """
    Factual validator для компактного grounding.

    В отличие от старой версии:
    - модель не обязана повторять все факты/координаты;
    - UCI проверяется отдельным regex (ловит d2e4/d2d4);
    - разрешены только те конкретные клетки/фигуры, которые реально
      присутствовали в СЖАТОМ prompt, а не во всём внутреннем Python-анализе;
    - одно хорошее предложение не бракуется только из-за формата;
    - фактические ошибки остаются hard constraints.
    """
    errors: list[str] = []

    if not isinstance(text, str):
        return False, ["Gigachess вернул не строку."]

    text = text.strip()
    if not text:
        return False, ["Gigachess вернул пустой ответ."]

    lowered = text.lower()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]

    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    expected_uci = grounding["uci"]
    from_square = grounding["from"]
    to_square = grounding["to"]
    moved_piece_word = grounding["moved_piece_word"]

    # --------------------------------------------------------
    # UCI — отдельная обязательная проверка.
    # \b между "2" и "e" не существует, поэтому square-regex сам по себе
    # никогда надёжно не поймает ошибку вида d2e4.
    # --------------------------------------------------------
    mentioned_uci = [
        value.lower()
        for value in re.findall(
            r"(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])",
            lowered,
        )
    ]
    for claimed_uci in mentioned_uci:
        if claimed_uci != expected_uci:
            errors.append(
                f"Модель назвала ход {claimed_uci}, но объясняется "
                f"проверенный ход {expected_uci}."
            )

    # --------------------------------------------------------
    # ФИГУРЫ — только из compact grounding.
    # --------------------------------------------------------
    mentioned_piece_words = {
        piece_name
        for piece_name in PIECE_NAMES.values()
        if _piece_word_present(lowered, piece_name)
    }
    unexpected_piece_words = (
        mentioned_piece_words - grounding["allowed_piece_words"]
    )
    if unexpected_piece_words:
        errors.append(
            "Названы фигуры, которых нет в переданном компактном контексте: "
            f"{sorted(unexpected_piece_words)}."
        )

    bad_subject_agreement = bool(
        re.search(
            r"\b(?:белые|ч[её]рные)\s+"
            r"(?:пешка|конь|слон|ладья|ферзь|король)\b",
            lowered,
        )
    )
    if bad_subject_agreement:
        errors.append(
            "Нарушено согласование цвета и названия фигуры "
            "(например, нужно «белая ладья», а не «белые ладья»)."
        )

    # --------------------------------------------------------
    # КООРДИНАТЫ — только те, которые реально были показаны модели.
    # --------------------------------------------------------
    used_squares = set(
        re.findall(r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])", lowered)
    )
    unexpected_squares = used_squares - grounding["allowed_squares"]
    if unexpected_squares:
        errors.append(
            "Использованы клетки, которых не было среди переданных "
            f"проверенных фактов: {sorted(unexpected_squares)}."
        )

    dangling_square_reference = bool(
        re.search(
            r"\b(?:на|в)\s+"
            r"(?:поле|клетку|клетке|клетке|клетки)"
            r"\s*(?:[.!?,;:]|$)",
            lowered,
        )
    )

    if dangling_square_reference:
        errors.append(
            "Ответ содержит незавершённую ссылку на клетку "
            "(например, «на поле» без координаты)."
        )

    # Явное движение "с X на Y".
    movement_pairs = re.findall(
        r"(?:с|из)\s+([a-h][1-8])[^.!?]{0,45}?(?:на|в)\s+([a-h][1-8])",
        lowered,
    )

    allowed_movement_pairs = {(from_square, to_square)}

    if move_facts.get("is_castling"):
        rook_from_for_move = grounding.get("castling_rook_from")
        rook_to_for_move = grounding.get("castling_rook_to")
        if rook_from_for_move and rook_to_for_move:
            allowed_movement_pairs.add(
                (rook_from_for_move, rook_to_for_move)
            )

    for claimed_from, claimed_to in movement_pairs:
        if (claimed_from, claimed_to) not in allowed_movement_pairs:
            if move_facts.get("is_castling"):
                expected_text = ", ".join(
                    f"{a}->{b}"
                    for a, b in sorted(allowed_movement_pairs)
                )
                errors.append(
                    "Неверно описано перемещение при рокировке: "
                    f"{claimed_from}->{claimed_to}; допустимы только "
                    f"проверенные пары {expected_text}."
                )
            else:
                errors.append(
                    "Неверно описано перемещение: "
                    f"{claimed_from}->{claimed_to}; проверенный ход "
                    f"{from_square}->{to_square}."
                )

    # Типичные конструкции с полем назначения.
    destination_patterns = [
        r"(?:ид[её]т|переход\w*|перемеща\w*|продвига\w*|"
        r"оказыва\w*)[^.!?]{0,24}?(?:на|в)\s+(?:поле\s+)?"
        r"([a-h][1-8])",
        r"занима\w*[^.!?]{0,20}?(?:поле\s+)?([a-h][1-8])",
    ]
    allowed_destination_squares = {to_square}
    if move_facts.get("is_castling"):
        rook_to_for_destination = grounding.get("castling_rook_to")
        if rook_to_for_destination:
            allowed_destination_squares.add(rook_to_for_destination)

    for pattern in destination_patterns:
        for claimed_to in re.findall(pattern, lowered):
            if claimed_to not in allowed_destination_squares:
                errors.append(
                    f"Неверно названо поле назначения {claimed_to}; "
                    "проверенные поля назначения — "
                    f"{sorted(allowed_destination_squares)}."
                )

    # --------------------------------------------------------
    # ГЕОМЕТРИЯ ХОДА — relation-specific validation.
    # --------------------------------------------------------
    expected_movement_type = str(
        grounding.get("movement_type") or ""
    ).lower()

    claimed_geometry: set[str] = set()

    if re.search(r"\bдиагонал\w*\b", lowered):
        claimed_geometry.add("diagonal")

    if re.search(r"\bвертикал\w*\b", lowered):
        claimed_geometry.add("vertical")

    if re.search(r"\bгоризонтал\w*\b", lowered):
        claimed_geometry.add("horizontal")

    knight_geometry_claim = bool(
        re.search(
            r"букв\w*\s*[«„\"']?г[»“\"']?",
            lowered,
        )
        or re.search(
            r"\bг[-\s]?образ\w*\b",
            lowered,
        )
    )
    if knight_geometry_claim:
        claimed_geometry.add("knight_jump")

    king_step_claim = bool(
        re.search(
            r"\bсоседн\w*\s+(?:поле|клетк\w*)\b",
            lowered,
        )
        or re.search(
            r"\bна\s+(?:один|одну|одно|одной|одного|одн\w*)\s+"
            r"(?:поле|клетк\w*|шаг\w*)\b",
            lowered,
        )
        or re.search(
            r"\b(?:один|одну|одно|одной|одного|одн\w*)\s+шаг\w*\b",
            lowered,
        )
    )
    if king_step_claim:
        claimed_geometry.add("king_step")

    known_geometries = {
        "diagonal",
        "vertical",
        "horizontal",
        "knight_jump",
        "king_step",
    }

    if (
        expected_movement_type in known_geometries
        and claimed_geometry
    ):
        wrong_geometry = (
            claimed_geometry - {expected_movement_type}
        )
        if wrong_geometry:
            geometry_names = {
                "diagonal": "диагональ",
                "vertical": "вертикаль",
                "horizontal": "горизонталь",
                "knight_jump": "ход буквой «Г»",
                "king_step": "ход на соседнюю клетку",
            }
            errors.append(
                "Неверно описана геометрия хода: указано "
                f"{[geometry_names[g] for g in sorted(wrong_geometry)]}; "
                "проверенный тип движения — "
                f"{geometry_names[expected_movement_type]}."
            )

    if grounding.get("pawn_single_step_geometry_required"):
        pawn_word_present = bool(
            re.search(
                r"\bпешк(?:а|и|у|ой|е|ам|ами|ах)\b",
                lowered,
            )
        )
        pawn_forward_claim = bool(
            re.search(r"\bвпер[её]д\b", lowered)
        )
        one_square_claim = bool(
            re.search(
                r"\b(?:на\s+)?(?:один|одну|одно|одной|одного|одн\w*)\s+"
                r"(?:поле|клетк\w*|шаг\w*)\b",
                lowered,
            )
            or re.search(
                r"\b(?:один|одну|одно|одной|одного|одн\w*)\s+шаг\w*\b",
                lowered,
            )
        )
        straight_claim = bool(
            re.search(
                r"\b(?:прямо|впер[её]д)\b",
                lowered,
            )
        )

        if not pawn_word_present:
            errors.append(
                "Для обычного хода пешки нужно явно назвать пешку."
            )

        if not pawn_forward_claim or not one_square_claim:
            errors.append(
                "Нужно явно указать правило обычного хода пешки: "
                "она идёт прямо вперёд на одну клетку."
            )

        if re.search(r"\bдиагонал\w*\b", lowered):
            errors.append(
                "Обычный ход пешки без взятия ошибочно описан как "
                "диагональный; пешка идёт прямо вперёд."
            )

        if re.search(r"\b(?:двойн\w*|на\s+две\s+клетк\w*)\b", lowered):
            errors.append(
                "Обычный ход пешки на одну клетку ошибочно описан "
                "как двойной ход."
            )

        if re.search(r"\b(?:бер[её]т|берут|взял\w*|забира\w*|снима\w*|бь[её]т|бьют)\b", lowered):
            errors.append(
                "Ответ выдумывает взятие, хотя проверенный ход пешки "
                "выполняется без взятия."
            )

    if grounding.get("pawn_capture_geometry_required"):
        pawn_word_present = bool(
            re.search(
                r"\bпешк(?:а|и|у|ой|е|ам|ами|ах)\b",
                lowered,
            )
        )
        pawn_diagonal_claim = bool(
            re.search(r"\bдиагонал\w*\b", lowered)
        )
        pawn_forward_claim = bool(
            re.search(r"\bвпер[её]д\b", lowered)
        )

        if not pawn_word_present:
            errors.append(
                "Для пешечного взятия нужно явно назвать пешку, "
                "которая выполняет ход."
            )

        if not pawn_diagonal_claim or not pawn_forward_claim:
            errors.append(
                "Нужно явно указать правило пешечного взятия: "
                "пешка берёт по диагонали вперёд."
            )

        if re.search(r"\b(?:горизонтал\w*|вертикал\w*)\b", lowered):
            errors.append(
                "Пешечное взятие ошибочно описано как движение "
                "по горизонтали или вертикали; пешка берёт "
                "по диагонали вперёд."
            )

        if re.search(r"\b(?:букв\w*\s*[«„\"']?г[»“\"']?|г[-\s]?образ\w*)\b", lowered):
            errors.append(
                "Пешечное взятие ошибочно описано как ход коня."
            )

    if expected_movement_type == "king_step":
        if re.search(r"\bрокиров\w*\b", lowered):
            errors.append(
                "Обычный ход короля ошибочно назван рокировкой."
            )

        if re.search(r"\b(?:прыга\w*|перепрыг\w*)\b", lowered):
            errors.append(
                "Обычный ход короля не является прыжком: "
                "король переходит на соседнюю клетку."
            )

    if grounding.get("movement_geometry_required"):
        expected_geometry_name = {
            "diagonal": "по диагонали",
            "horizontal": "по горизонтали",
            "vertical": "по вертикали",
            "knight_jump": "ходом буквой «Г»",
            "king_step": "на соседнюю клетку",
        }.get(expected_movement_type)

        if (
            expected_geometry_name
            and expected_movement_type not in claimed_geometry
            and not grounding.get("pawn_capture_geometry_required")
            and not grounding.get("pawn_single_step_geometry_required")
        ):
            errors.append(
                "Тип траектории должен быть назван явно: "
                f"проверенный ход выполняется {expected_geometry_name}."
            )

    # --------------------------------------------------------
    # КОРОЛЬ ПРИ РОКИРОВКЕ — relation-specific validation.
    # --------------------------------------------------------
    if move_facts.get("is_castling"):
        king_pair_pattern = (
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,45}?(?:с|из)\s+([a-h][1-8])"
            r"[^.!?]{0,35}?(?:на|в)\s+(?:поле\s+|клет(?:ку|ки|ке)\s+)?"
            r"([a-h][1-8])"
        )
        king_destination_pattern = (
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,30}?(?:переход\w*|перемеща\w*|ид[её]т|переш[её]л\w*)"
            r"[^.!?]{0,22}?(?:на|в)\s+(?:поле\s+|клет(?:ку|ки|ке)\s+)?"
            r"([a-h][1-8])"
        )

        for claimed_from, claimed_to in re.findall(
            king_pair_pattern,
            lowered,
        ):
            if claimed_from != from_square or claimed_to != to_square:
                errors.append(
                    "Неверно описано перемещение короля при рокировке: "
                    f"{claimed_from}->{claimed_to}; проверенное перемещение "
                    f"{from_square}->{to_square}."
                )

        for claimed_to in re.findall(
            king_destination_pattern,
            lowered,
        ):
            if claimed_to != to_square:
                errors.append(
                    "Неверно названо поле назначения короля при рокировке: "
                    f"{claimed_to}; проверенное поле — {to_square}."
                )

    # --------------------------------------------------------
    # ЛАДЬЯ ПРИ РОКИРОВКЕ — relation-specific validation.
    # --------------------------------------------------------
    castling_rook_from = grounding.get("castling_rook_from")
    castling_rook_to = grounding.get("castling_rook_to")

    if move_facts.get("is_castling") and castling_rook_from and castling_rook_to:
        rook_pair_pattern = (
            r"ладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)"
            r"[^.!?]{0,45}?(?:с|из)\s+([a-h][1-8])"
            r"[^.!?]{0,35}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
        )
        rook_destination_pattern = (
            r"ладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)"
            r"[^.!?]{0,25}?(?:переход\w*|перемеща\w*|ид[её]т)"
            r"[^.!?]{0,20}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
        )

        for claimed_from, claimed_to in re.findall(
            rook_pair_pattern,
            lowered,
        ):
            if (
                claimed_from != castling_rook_from
                or claimed_to != castling_rook_to
            ):
                errors.append(
                    "Неверно описано перемещение ладьи при рокировке: "
                    f"{claimed_from}->{claimed_to}; проверенное перемещение "
                    f"{castling_rook_from}->{castling_rook_to}."
                )

        for claimed_to in re.findall(
            rook_destination_pattern,
            lowered,
        ):
            if claimed_to != castling_rook_to:
                errors.append(
                    "Неверно названо поле назначения ладьи при рокировке: "
                    f"{claimed_to}; проверенное поле — {castling_rook_to}."
                )

    # --------------------------------------------------------
    # ПОЛЕ КОРОЛЯ СОПЕРНИКА — relation-specific validation.
    # --------------------------------------------------------
    opponent_king_square = grounding.get("opponent_king_square")

    if opponent_king_square:
        king_square_claims: set[str] = set()

        king_patterns = [
            # "король находится/стоит/остаётся на поле h8"
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,24}?"
            r"(?:находи\w*|стои\w*|оста[её]т\w*|располож\w*)"
            r"[^.!?]{0,18}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])",

            # "королю на h8 поставлен мат"
            # Здесь координата непосредственно относится к слову "король".
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"\s+(?:соперника\s+|ч[её]рных\s+|белых\s+)?"
            r"(?:на|в)\s+(?:поле\s+)?([a-h][1-8])",

            # "на поле h8 находится/стоит чёрный король"
            r"(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
            r"[^.!?]{0,18}?"
            r"(?:находи\w*|стои\w*|оста[её]т\w*|располож\w*)"
            r"[^.!?]{0,18}?"
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)",
        ]

        for pattern in king_patterns:
            king_square_claims.update(
                value.lower()
                for value in re.findall(pattern, lowered)
            )

        invalid_king_squares = (
            king_square_claims - {opponent_king_square}
        )
        if invalid_king_squares:
            errors.append(
                "Неверно указано поле короля соперника: "
                f"{sorted(invalid_king_squares)}; проверенное поле — "
                f"{opponent_king_square}."
            )

    # --------------------------------------------------------
    # КОНТРОЛЬ / АТАКА / ЗАЩИТА — проверяем именно ОТНОШЕНИЕ.
    # --------------------------------------------------------
    for sentence in sentences:
        sentence_lower = sentence.lower()

        control_match = re.search(r"контрол", sentence_lower)
        if control_match:
            tail = sentence_lower[control_match.start():]
            control_squares = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    tail,
                )
            )
            invalid = control_squares - grounding["moved_piece_controls"]
            if invalid:
                errors.append(
                    "Неверно описан контроль перемещённой фигуры: "
                    f"{sorted(invalid)} не входят в переданный проверенный "
                    f"набор {sorted(grounding['moved_piece_controls'])}."
                )

        if re.search(r"атаку|атакует|под атак", sentence_lower):
            attack_squares_in_sentence = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    sentence_lower,
                )
            )
            # FROM/TO сами по себе не являются целями атаки.
            attack_squares_in_sentence -= {from_square, to_square}
            invalid = (
                attack_squares_in_sentence - grounding["attack_squares"]
            )
            if invalid:
                errors.append(
                    "Неверно описана конкретная атака: "
                    f"{sorted(invalid)} не подтверждены compact grounding."
                )

        if "защищ" in sentence_lower or "защит" in sentence_lower:
            defended_in_sentence = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    sentence_lower,
                )
            )
            defended_in_sentence -= {from_square, to_square}
            invalid = (
                defended_in_sentence - grounding["defended_squares"]
            )
            if invalid:
                errors.append(
                    "Неверно описана конкретная защита: "
                    f"{sorted(invalid)} не подтверждены compact grounding."
                )

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------
    capture_completed_patterns = [
        r"\bбер(?:[её]т|ут|[её]м|[её]те)\b",
        r"\bвзял(?:а|и|о)?\b",
        r"\bвзяв\b",
        r"\bзабира(?:ет|ют|ем|ете)\b",
        r"\bзабрал(?:а|и|о)?\b",
        r"\bзахватыва(?:ет|ют|ем|ете|я)\b",
        r"\bзахватил(?:а|и|о)?\b",
        r"\bснима(?:ет|ют|ем|ете)\b",
        r"\bснял(?:а|и|о)?\b",
        r"\bбь(?:[её]т|ют|[её]м|[её]те)\b",
        r"\bпобил(?:а|и|о)?\b",
        r"\bвзят(?:ие|ия|ием|ую|ой)\b",
        r"\b(?:фигура|пешка|конь|слон|ладья|ферзь)\s+снят\w*\b",
    ]

    capture_future_patterns = [
        r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,35}\bвзять\b",
        r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,35}\bзабрать\b",
        r"\bугрожа\w*[^.!?]{0,35}\bвзять\b",
        r"\bготов\w*[^.!?]{0,35}\bвзят",
    ]

    capture_completed = any(
        re.search(pattern, lowered)
        for pattern in capture_completed_patterns
    )
    capture_future_claim = any(
        re.search(pattern, lowered)
        for pattern in capture_future_patterns
    )

    capture_negated = any(
        phrase in lowered
        for phrase in (
            "взятия нет",
            "без взятия",
            "не является взятием",
        )
    )
    positive_capture = capture_completed and not capture_negated

    if move_facts.get("is_capture"):
        if capture_future_claim:
            errors.append(
                "Ответ описывает взятие как будущую возможность, "
                "но проверенный ход уже совершает взятие."
            )

        if not positive_capture:
            errors.append(
                "Ход является уже совершившимся взятием. Нужно прямо сказать, "
                "что фигура этим ходом берёт/забирает/снимает указанную фигуру."
            )

        captured_piece = str(
            move_facts.get("captured_piece") or ""
        ).lower()
        captured_piece_word = next(
            (
                name
                for name in PIECE_NAMES.values()
                if name in captured_piece
            ),
            None,
        )
        if (
            captured_piece_word
            and not _piece_word_present(
                lowered,
                captured_piece_word,
            )
        ):
            errors.append(
                "Не названа взятая фигура "
                f"«{captured_piece_word}»."
            )

        if captured_piece_word:
            (
                captured_acc_pattern,
                captured_nom_pattern,
            ) = _capture_object_case_patterns(
                captured_piece_word
            )

            capture_verb_stem = (
                r"(?:бер(?:[её]т|ут|[её]м|[её]те)|"
                r"взял(?:а|и|о)?|взяв|"
                r"забира(?:ет|ют|ем|ете)|"
                r"забрал(?:а|и|о)?|"
                r"снима(?:ет|ют|ем|ете)|"
                r"снял(?:а|и|о)?|"
                r"бь(?:[её]т|ют|[её]м|[её]те)|"
                r"захватыва(?:ет|ют|ем|ете|я)|"
                r"захватил(?:а|и|о)?)"
            )

            bad_capture_object = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b"
                    rf"[^.!?]{{0,18}}?"
                    rf"\b(?:белый|ч[её]рный)?\s*"
                    rf"{captured_nom_pattern}\b",
                    lowered,
                )
            )

            good_capture_object = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b"
                    rf"[^.!?]{{0,18}}?"
                    rf"\b(?:белого|ч[её]рного|белую|ч[её]рную)?\s*"
                    rf"{captured_acc_pattern}\b",
                    lowered,
                )
            )

            # Для существительного "взятие" допускается родительный:
            # "взятие ферзя", "взятие ладьи" и т.п.
            nominal_capture_ok = bool(
                re.search(
                    rf"\bвзят\w*\b[^.!?]{{0,16}}?"
                    rf"\b(?:белого|ч[её]рного|белой|ч[её]рной)?\s*"
                    rf"(?:{captured_acc_pattern}|"
                    rf"{'ладьи' if captured_piece_word == 'ладья' else 'пешки' if captured_piece_word == 'пешка' else captured_acc_pattern})\b",
                    lowered,
                )
            )

            if bad_capture_object and not good_capture_object:
                errors.append(
                    "Взятая фигура названа в неверном падеже: "
                    f"после глагола взятия нужно использовать форму "
                    f"«{_piece_word_accusative(captured_piece_word)}»."
                )

            # Если используется именно глагольная конструкция взятия,
            # объект должен быть грамматически выражен корректно.
            has_capture_verb = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b",
                    lowered,
                )
            )
            if (
                has_capture_verb
                and not good_capture_object
                and not nominal_capture_ok
                and not bad_capture_object
            ):
                errors.append(
                    "После глагола взятия не удалось однозначно найти "
                    "взятую фигуру в корректной форме."
                )

        if move_facts.get("is_en_passant"):
            en_passant_completed = bool(
                re.search(
                    r"\bвзят\w*\s+на\s+проходе\b"
                    r"|\bбер\w*\s+на\s+проходе\b"
                    r"|\ben\s*passant\b",
                    lowered,
                )
            )

            en_passant_future = bool(
                re.search(
                    r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,40}"
                    r"\b(?:взять|бить)\b[^.!?]{0,20}\bна\s+проходе\b"
                    r"|\bготов\w*[^.!?]{0,35}\bвзят\w*\s+на\s+проходе\b",
                    lowered,
                )
            )

            if en_passant_future:
                errors.append(
                    "Ответ описывает взятие на проходе как будущую "
                    "возможность, но оно уже выполняется этим ходом."
                )

            if not en_passant_completed:
                errors.append(
                    "Ход является взятием на проходе, но ответ не называет "
                    "этот специальный механизм."
                )

            captured_square = str(
                move_facts.get("captured_square") or ""
            ).lower()
            if captured_square and captured_square not in used_squares:
                errors.append(
                    "Для взятия на проходе нужно назвать поле снятой пешки "
                    f"{captured_square}."
                )
    elif positive_capture or capture_future_claim:
        errors.append(
            "Ответ выдумывает взятие или возможность взятия, "
            "которых нет в проверенном ходе."
        )

    # --------------------------------------------------------
    # ШАХ / МАТ / РОКИРОВКА / ПРЕВРАЩЕНИЕ
    # --------------------------------------------------------
    has_check_word = bool(
        re.search(r"\bшах(?:а|у|ом|е)?\b", lowered)
    )

    check_completed_patterns = [
        # Активные / деепричастные формы.
        r"\bобъяв(?:ля\w*|ил(?:а|и|о)?|ив)\s+шах\b",
        r"\bда[её]т\w*\s+шах\b",
        r"\bдал(?:а|и|о)?\s+шах\b",
        r"\bпоставил(?:а|и|о)?\s+шах\b",
        r"\bставит\s+шах\b",

        # Пассивные завершённые формы.
        r"\b(?:был\s+)?объявлен\s+шах\b",
        r"\bшах\s+(?:был\s+)?объявлен\b",
        r"\b(?:был\s+)?дан\s+шах\b",
        r"\bшах\s+(?:был\s+)?дан\b",

        # Краткие устойчивые формы.
        r"\bс\s+шахом\b",
        r"\bпод\s+шахом\b",
        r"\bшахует\b",
    ]
    check_completed = any(
        re.search(pattern, lowered)
        for pattern in check_completed_patterns
    )

    check_future_patterns = [
        r"\bугрожа\w*[^.!?]{0,40}\bшах(?:ом|а)?\b",
        r"\bмож(?:ет|но|гут)\b[^.!?]{0,45}"
        r"(?:объявить|дать|поставить)\s+шах\b",
        r"\bготовит\w*[^.!?]{0,35}\bшах\b",
        r"\bпозволя\w*[^.!?]{0,45}"
        r"(?:объявить|дать|поставить)\s+шах\b",
    ]
    check_future_claim = any(
        re.search(pattern, lowered)
        for pattern in check_future_patterns
    )

    check_positive = (
        check_completed
        and not _is_negated_claim(
            lowered,
            r"шах(?:а|у|ом|е)?",
        )
    )

    has_mate_word = bool(
        re.search(r"\bмат(?:а|у|ом|е)?\b", lowered)
    )

    # "Есть слово мат" недостаточно. Для реального checkmate нужен
    # завершённый факт: "ставит мат", "это мат", "матует" и т.п.
    mate_completed_patterns = [
        r"\bстав(?:ит|ят|я)\s+(?:шах\s+и\s+)?мат\b",
        r"\bпоставил(?:а|и|о)?\s+(?:шах\s+и\s+)?мат\b",
        r"\bпоставлен\w*\s+(?:шах\s+и\s+)?мат\b",
        r"\bобъявля(?:ет|ют)\s+мат\b",
        r"\bматует\b",
        r"\bзаматовал(?:а|и|о)?\b",
        r"\bэто\s+(?:и\s+есть\s+)?мат\b",
        r"\bшах\s+и\s+мат\b",
        r"\bполучается\s+мат\b",
    ]
    mate_completed = any(
        re.search(pattern, lowered)
        for pattern in mate_completed_patterns
    )

    # Формулировки будущей возможности/угрозы НЕ описывают ход,
    # который уже является матом.
    mate_future_patterns = [
        r"\bмож(?:но|ет|ем|ете|гут)\b[^.!?]{0,45}\bпоставить\s+мат\b",
        r"\bпозволя\w*\b[^.!?]{0,45}\bпоставить\s+мат\b",
        r"\bготовит\w*\b[^.!?]{0,35}\bмат\b",
        r"\bугрожа\w*\b[^.!?]{0,35}\bмат(?:ом|а)?\b",
        r"\bсозда[её]т\w*\b[^.!?]{0,35}\bугроз\w*\b[^.!?]{0,25}\bмат",
        r"\bмат\s+в\s+один\s+ход\b",
    ]
    mate_future_claim = any(
        re.search(pattern, lowered)
        for pattern in mate_future_patterns
    )

    mate_positive = (
        mate_completed
        and not _is_negated_claim(
            lowered,
            r"мат(?:а|у|ом|е)?",
        )
    )

    has_castling = "рокиров" in lowered

    castling_completed_patterns = [
        r"\bвыполня\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bвыполненн\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bдела\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bсоверша\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bрокиру\w*",
        r"\bэто\s+(?:уже\s+)?(?:и\s+есть\s+)?"
        r"(?:выполненн\w*\s+)?"
        r"(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bпроисход\w*\s+рокиров",
        r"\bознача\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bявля\w*\s+(?:собой\s+)?(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
    ]
    castling_completed = any(
        re.search(pattern, lowered)
        for pattern in castling_completed_patterns
    )

    castling_future_patterns = [
        r"\bготов\w*\s+(?:к\s+)?рокиров",
        r"\bподготавлива\w*\s+рокиров",
        r"\bпозволя\w*[^.!?]{0,35}\bрокиров",
        r"\bмож(?:но|ет|ем|ете|гут)\b[^.!?]{0,35}\bрокиров",
        r"\bсозда[её]т\w*[^.!?]{0,35}\bвозможност\w*[^.!?]{0,20}\bрокиров",
    ]
    castling_future_claim = any(
        re.search(pattern, lowered)
        for pattern in castling_future_patterns
    )

    castling_positive = (
        castling_completed
        and not any(
            phrase in lowered
            for phrase in (
                "рокировки нет",
                "без рокировки",
                "не является рокировкой",
            )
        )
    )

    has_promotion = "превращ" in lowered

    promotion_completed_patterns = [
        r"\bпревраща\w*\s+в\b",
        r"\bпревратил(?:ась|ся|и|ось)?\s+в\b",
        r"\bпревративш\w*\s+в\b",
        r"\bстановит(?:ся|ься)\b[^.!?]{0,20}"
        r"(?:ферз|ладь|слон|кон)",
    ]
    promotion_completed = any(
        re.search(pattern, lowered)
        for pattern in promotion_completed_patterns
    )

    promotion_future_patterns = [
        r"\bмож(?:ет|но|гут)\b[^.!?]{0,40}\bпреврат",
        r"\bготов\w*[^.!?]{0,35}\bпревращ",
        r"\bприближа\w*[^.!?]{0,35}\bпол[юя]\s+превращ",
        r"\bпродвига\w*[^.!?]{0,35}\b(?:к|до)\s+пол[яю]\s+превращ",
        r"\bид[её]т\w*[^.!?]{0,30}\bк\s+пол[юя]\s+превращ",
    ]
    promotion_future_claim = any(
        re.search(pattern, lowered)
        for pattern in promotion_future_patterns
    )

    promotion_positive = (
        promotion_completed
        and not any(
            phrase in lowered
            for phrase in (
                "превращения нет",
                "без превращения",
                "не является превращением",
            )
        )
    )

    if move_facts.get("is_check"):
        if check_future_claim:
            errors.append(
                "Ответ описывает шах как угрозу или будущую возможность, "
                "но проверенный ход уже объявляет шах."
            )
        if not check_positive and not mate_positive:
            errors.append(
                "Ход уже даёт шах, но ответ не описывает шах "
                "как совершившееся событие."
            )
    elif check_positive or check_future_claim:
        errors.append("Ответ выдумывает шах или угрозу шаха.")

    if move_facts.get("is_checkmate"):
        if mate_future_claim:
            errors.append(
                "Ответ описывает мат как будущую возможность или угрозу, "
                "но проверенный ход уже сам ставит мат."
            )
        if not mate_positive:
            errors.append(
                "Ход уже ставит мат. Ответ должен прямо описать завершённое "
                "событие: «ставит мат», «это мат» или эквивалентную формулировку."
            )
    elif mate_positive or mate_future_claim:
        errors.append("Ответ выдумывает мат или угрозу мата.")

    if move_facts.get("is_castling"):
        if castling_future_claim:
            errors.append(
                "Ответ описывает рокировку как будущую возможность или "
                "подготовку, но проверенный ход уже сам является рокировкой."
            )
        if not castling_positive:
            errors.append(
                "Ход уже является рокировкой. Ответ должен прямо сказать, "
                "что рокировка выполняется этим ходом."
            )
        if not re.search(r"\bладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)\b", lowered):
            errors.append(
                "При рокировке одновременно перемещается ладья; "
                "объяснение должно упомянуть ладью."
            )

        castling_required_squares = {
            str(explained_move.get("from") or "").lower(),
            str(explained_move.get("to") or "").lower(),
            str(grounding.get("castling_rook_from") or "").lower(),
            str(grounding.get("castling_rook_to") or "").lower(),
        }
        castling_required_squares.discard("")

        missing_castling_squares = {
            square
            for square in castling_required_squares
            if not re.search(
                rf"(?<![a-z0-9]){re.escape(square)}(?![a-z0-9])",
                lowered,
            )
        }

        if missing_castling_squares:
            errors.append(
                "Для объяснения рокировки должны быть сохранены точные "
                "координаты обоих перемещений. Не названы клетки: "
                f"{sorted(missing_castling_squares)}."
            )
    elif castling_positive or castling_future_claim:
        errors.append("Ответ выдумывает рокировку или подготовку к ней.")

    if move_facts.get("is_promotion"):
        promotion_piece = str(
            move_facts.get("promotion_piece") or ""
        ).lower()

        if promotion_future_claim:
            errors.append(
                "Ответ описывает превращение как будущую возможность или "
                "приближение к полю превращения, но оно уже происходит "
                "этим ходом."
            )

        if not promotion_positive:
            errors.append(
                "Ход уже превращает пешку в новую фигуру, но ответ "
                "не описывает завершённое превращение."
            )

        if (
            promotion_piece
            and not _piece_word_present(
                lowered,
                promotion_piece,
            )
        ):
            errors.append(
                "Не названа фигура превращения "
                f"«{promotion_piece}»."
            )

        promotion_square_mentions: set[str] = set()

        # "на поле a7 превращается...", "на b8 превращается..."
        for match in re.finditer(
            r"\bна\s+(?:поле\s+)?([a-h][1-8])"
            r"\s+превращ\w*",
            lowered,
        ):
            promotion_square_mentions.add(match.group(1))

        # "превращается ... на поле b8"
        for match in re.finditer(
            r"\bпревращ\w*[^.!?]{0,30}"
            r"\bна\s+(?:поле\s+)?([a-h][1-8])\b",
            lowered,
        ):
            promotion_square_mentions.add(match.group(1))

        expected_promotion_square = str(
            move_facts.get("to") or explained_move.get("to") or ""
        ).lower()

        wrong_promotion_squares = sorted(
            square
            for square in promotion_square_mentions
            if expected_promotion_square
            and square != expected_promotion_square
        )

        if wrong_promotion_squares:
            errors.append(
                "Неверно указано поле превращения: "
                f"{wrong_promotion_squares}; проверенное поле — "
                f"{expected_promotion_square}."
            )

        pawn_continues_after_promotion = bool(
            re.search(
                r"\bпешк\w*[^.!?]{0,45}\bпродолж\w*\s+движ",
                lowered,
            )
            or re.search(
                r"\bпозволя\w*\s+(?:ей|пешк\w*)"
                r"[^.!?]{0,35}\bпродолж\w*\s+движ",
                lowered,
            )
            or re.search(
                r"\bпосле\s+превращ\w*[^.!?]{0,35}"
                r"\bпешк\w*\b",
                lowered,
            )
        )
        if pawn_continues_after_promotion:
            errors.append(
                "После превращения пешка как пешка больше не существует; "
                "ответ ошибочно приписывает ей дальнейшее движение "
                "или действие после превращения."
            )
    elif promotion_positive or promotion_future_claim:
        errors.append("Ответ выдумывает превращение или подготовку к нему.")

    # --------------------------------------------------------
    # АТАКА НА КОРОЛЯ / УГРОЗЫ
    # --------------------------------------------------------
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if (
            "корол" in sentence_lower
            and re.search(
                r"атаку|атакует|под атак|угрож",
                sentence_lower,
            )
            and not move_facts.get("is_check")
        ):
            errors.append(
                "Ответ утверждает непосредственную атаку/угрозу королю, "
                "но после проверенного хода шаха нет."
            )

    # --------------------------------------------------------
    # УТВЕРЖДЕНИЯ О СЛЕДУЮЩЕМ ХОДЕ СОПЕРНИКА.
    # --------------------------------------------------------
    opponent_legal_moves = int(
        grounding.get("opponent_legal_moves") or 0
    )
    terminal_state = grounding.get("terminal") or {}

    opponent_move_claim = bool(
        re.search(
            r"вынужден\w*[^.!?]{0,55}?"
            r"(?:сделать\s+ход|пойти|перейти|ходить)",
            lowered,
        )
        or re.search(
            r"(?:должен|должны|прид[её]тся)[^.!?]{0,45}?"
            r"(?:сделать\s+ход|пойти|перейти|ходить)",
            lowered,
        )
        or re.search(
            r"сделать\s+ход\s+корол[её]?м",
            lowered,
        )
        or re.search(
            r"ход\s+корол[её]?м\s+(?:на|в)\s+(?:поле\s+)?[a-h][1-8]",
            lowered,
        )
    )

    if (
        opponent_move_claim
        and (
            terminal_state.get("is_game_over")
            or opponent_legal_moves == 0
        )
    ):
        errors.append(
            "Ответ утверждает, что соперник должен сделать следующий ход, "
            "но после проверенного хода у соперника нет легальных ходов."
        )

    no_opponent_moves_claim = bool(
        re.search(
            r"\b(?:больше\s+)?не\s+мог\w*[^.!?]{0,24}"
            r"(?:сделать\s+)?ход\w*\b",
            lowered,
        )
        or re.search(
            r"\bнет\s+(?:ни\s+одного\s+)?"
            r"(?:(?:легальн|доступн|возможн)\w*\s+)?"
            r"ход\w*\b",
            lowered,
        )
        or re.search(
            r"\bход(?:ов|а)?\s+(?:больше\s+)?не\s+"
            r"остал\w*\b",
            lowered,
        )
        or re.search(
            r"\bне\s+остал\w*[^.!?]{0,24}\bход\w*\b",
            lowered,
        )
        or re.search(
            r"\bлиш[её]н\w*[^.!?]{0,20}\bход\w*\b",
            lowered,
        )
    )

    if no_opponent_moves_claim and opponent_legal_moves > 0:
        errors.append(
            "Ответ утверждает, что у соперника не осталось доступных ходов, "
            f"но Python нашёл {opponent_legal_moves} легальн"
            + (
                "ый ответ."
                if opponent_legal_moves == 1
                else "ых ответа."
                if 2 <= opponent_legal_moves <= 4
                else "ых ответов."
            )
        )

    # --------------------------------------------------------
    # ИТОГ ПАРТИИ / НИЧЬЯ / ПОБЕДА
    # --------------------------------------------------------
    terminal = grounding.get("terminal") or {}
    actual_result = terminal.get("result") or "*"
    is_game_over = bool(terminal.get("is_game_over"))

    draw_claim = bool(
        re.search(
            r"\bничь(?:я|ей|ю|е)\b|\bвничью\b",
            lowered,
        )
    )
    white_win_claim = bool(
        re.search(
            r"(?:партия|игра)\s+(?:сразу\s+)?"
            r"(?:заканчива\w*|заверша\w*)\s+"
            r"(?:победой\s+)?бел",
            lowered,
        )
        or re.search(
            r"\bпобед(?:а|е|ой|у)\s+бел\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:привод\w*|привел\w*|привёл\w*|"
            r"вед[её]т|гарантир\w*)\s+к\s+"
            r"побед\w*\s+бел\w*\b",
            lowered,
        )
        or re.search(
            r"\bбел\w*\s+"
            r"(?:выигр\w*|побед(?:ил\w*|ят\w*|ил|или|ила|ило|а|ают|ит))\b",
            lowered,
        )
    )
    black_win_claim = bool(
        re.search(
            r"(?:партия|игра)\s+(?:сразу\s+)?"
            r"(?:заканчива\w*|заверша\w*)\s+"
            r"(?:победой\s+)?ч[её]рн",
            lowered,
        )
        or re.search(
            r"\bпобед(?:а|е|ой|у)\s+ч[её]рн\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:привод\w*|привел\w*|привёл\w*|"
            r"вед[её]т|гарантир\w*)\s+к\s+"
            r"побед\w*\s+ч[её]рн\w*\b",
            lowered,
        )
        or re.search(
            r"\bч[её]рн\w*\s+"
            r"(?:выигр\w*|побед(?:ил\w*|ят\w*|ил|или|ила|ило|а|ают|ит))\b",
            lowered,
        )
    )
    unsupported_future_claim = bool(
        re.search(
            r"\bмат\s+(?:неизбеж\w*|гарантир\w*|неотвратим\w*)\b",
            lowered,
        )
        or re.search(
            r"\b(?:неизбеж\w*|гарантир\w*|неотвратим\w*)\s+мат\b",
            lowered,
        )
        or re.search(
            r"\bпешк\w*\s+(?:нельзя|невозможно)\s+остановить\b",
            lowered,
        )
        or re.search(
            r"\b(?:неудержим\w*|неостановим\w*)\s+пешк\w*\b",
            lowered,
        )
        or re.search(
            r"\bкорол\w*\s+не\s+мож\w*\s+догнать\s+пешк\w*\b",
            lowered,
        )
        or re.search(
            r"\bпешк\w*\s+гарантир\w*\s+превращ\w*\b",
            lowered,
        )
    )

    unique_move_claim = bool(
        re.search(
            r"\bединственн\w*\s+"
            r"(?:(?:возможн\w*|допустим\w*)\s+)?"
            r"(?:ход|вариант|решени\w*)\b",
            lowered,
        )
        or re.search(
            r"\bтолько\s+(?:этот|данный|такой)\s+ход\b",
            lowered,
        )
        or re.search(
            r"\b(?:другого|иных|альтернативн\w*)\s+"
            r"(?:хода|ходов|варианта|вариантов)\s+нет\b",
            lowered,
        )
        or re.search(
            r"\bнет\s+(?:другого|иных|альтернативн\w*)\s+"
            r"(?:хода|ходов|варианта|вариантов)\b",
            lowered,
        )
    )

    generic_win_claim = bool(
        re.search(
            r"\b(?:выигрывая|побеждая)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)\b",
            lowered,
        )
        or re.search(
            r"\b(?:обеспечивая|принося|гарантируя)\s+"
            r"(?:немедленную\s+|сразу\s+)?побед\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:обеспечива\w*|принос\w*|гарантир\w*)\s+"
            r"(?:немедленную\s+|сразу\s+)?побед\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:вед[её]т|привод\w*|привел\w*|привёл\w*)\s+"
            r"к\s+(?:немедленн\w*\s+)?(?:выигрыш\w*|побед\w*)\b",
            lowered,
        )
    )

    resignation_claim = bool(
        re.search(
            r"\b(?:"
            r"сдаться|"
            r"сда[её]тся|сдаются|"
            r"сдал(?:ся|ась|ись)|"
            r"капитулиру\w*|"
            r"капитуляц\w*"
            r")\b",
            lowered,
        )
    )

    if resignation_claim:
        errors.append(
            "Ответ выдумывает сдачу или капитуляцию. "
            "Проверенные факты описывают позицию после хода, "
            "но не содержат события сдачи."
        )

    generic_terminal_claim = bool(
        re.search(
            r"(?:партия|игра)\s+"
            r"(?:заканчива\w*|заверша\w*|оканчива\w*|окончен\w*)",
            lowered,
        )
        or re.search(
            r"(?:заканчива\w*|заверша\w*|оканчива\w*)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)",
            lowered,
        )
        or re.search(
            r"(?:завершая|заканчивая|оканчивая)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)",
            lowered,
        )
        or re.search(
            r"(?:партия|игра)\s+(?:уже\s+)?"
            r"(?:закончена|завершена|окончена)",
            lowered,
        )
    )

    if unsupported_future_claim:
        errors.append(
            "Ответ делает неподтверждённый прогноз о будущем позиции "
            "(например, неизбежный мат, неудержимая пешка или невозможность "
            "догнать её). Такие выводы не были подтверждены Python."
        )

    if unique_move_claim:
        errors.append(
            "Ответ утверждает, что ход является единственным, "
            "но уникальность хода не была подтверждена Python."
        )

    if generic_win_claim and not is_game_over:
        errors.append(
            "Ответ утверждает, что этим ходом партия уже выигрывается, "
            "но после проверенного хода позиция не является терминальной."
        )

    if draw_claim and actual_result != "1/2-1/2":
        errors.append(
            "Ответ утверждает ничью, но после проверенного хода "
            "партия не заканчивается вничью."
        )

    if white_win_claim and actual_result != "1-0":
        errors.append(
            "Ответ утверждает немедленную победу белых, но такой "
            "результат после этого хода не наступает."
        )

    if black_win_claim and actual_result != "0-1":
        errors.append(
            "Ответ утверждает немедленную победу чёрных, но такой "
            "результат после этого хода не наступает."
        )

    # Любое "партия заканчивается ..." запрещаем в нетерминальной позиции,
    # даже если модель не уточнила результат.
    if generic_terminal_claim and not is_game_over:
        errors.append(
            "Ответ утверждает, что партия заканчивается после этого хода, "
            "но позиция после хода не является терминальной."
        )

    decisive_finish_claim = bool(
        re.search(
            r"(?:реша\w*\s+(?:исход|партию|игру)|"
            r"заверша\w*\s+(?:партию|игру)|"
            r"заканчива\w*\s+(?:партию|игру))",
            lowered,
        )
    )
    if decisive_finish_claim and not is_game_over:
        errors.append(
            "Ответ приписывает ходу немедленное завершение партии, "
            "хотя после хода игра продолжается."
        )

    # --------------------------------------------------------
    # НЕПОДТВЕРЖДЁННЫЕ ЯРЛЫКИ И СИЛЬНЫЕ ВЫВОДЫ
    # --------------------------------------------------------
    if re.search(r"\bгамбит\w*\b", lowered):
        errors.append(
            "Название гамбита не подтверждено программным контекстом."
        )

    if re.search(
        r"\b(?:дебют|защита|система)\s+[а-яёa-z-]+",
        lowered,
    ):
        errors.append(
            "Конкретное название дебюта/системы не подтверждено."
        )

    strong_patterns = [
        r"выигрыва\w*\s+(?:ферз|ладь|слон|кон|пеш|фигур|материал)",
        r"теря\w*\s+(?:ферз|ладь|слон|кон|пеш|фигур|материал)",
        r"жертв\w*",
        r"вынужда\w*\s+соперник",
        r"форсир\w*\s+(?:выигрыш|мат)",
        r"решающ\w*\s+преимуществ",
        r"победн\w*\s+позици",
        r"вед[её]т\s+к\s+(?:ничь|побед)",
        r"гарантир\w*\s+(?:ничь|побед)",
    ]
    for pattern in strong_patterns:
        if re.search(pattern, lowered):
            errors.append(
                "Ответ содержит сильный тактический/материальный вывод, "
                "которого нет в compact grounding."
            )
            break

    # --------------------------------------------------------
    # НЕПОДТВЕРЖДЁННЫЕ ОЦЕНОЧНЫЕ ЦЕЛИ / БЕЗОПАСНОСТЬ.
    # --------------------------------------------------------
    king_safety_supported = bool(
        grounding.get("king_safety_supported")
    )
    generic_defense_supported = bool(
        grounding.get("generic_defense_supported")
    )

    safety_claim = bool(
        re.search(
            r"\bбезопасн\w*\b"
            r"|\bукрыва\w*\s+корол"
            r"|\bобезопас\w*\s+корол"
            r"|\bзащища\w*\s+корол",
            lowered,
        )
    )
    if safety_claim and not king_safety_supported:
        errors.append(
            "Ответ делает вывод о безопасности/укрытии короля, "
            "которого нет в compact grounding."
        )

    generic_defense_claim = bool(
        re.search(
            r"\bдля\s+защит\w*\b"
            r"|\bс\s+целью\s+защит\w*\b"
            r"|\bдля\s+обороны\b"
            r"|\bмож(?:ет|гут|но)\b[^.!?]{0,30}\bзащища\w*\b"
            r"|\bзащища\w*\s+корол",
            lowered,
        )
    )
    if generic_defense_claim and not generic_defense_supported:
        errors.append(
            "Ответ приписывает ходу общую защитную цель, "
            "которая не подтверждена compact grounding."
        )

    if move_facts.get("is_castling"):
        unsupported_castling_strategy_claim = bool(
            re.search(
                r"\bсоединя\w*\b"
                r"|\bактивизир\w*\b"
                r"|\bвводи\w*\s+ладь"
                r"|\bразвива\w*\s+ладь"
                r"|\bразвити\w*\s+ладь"
                r"|\bулучша\w*\s+(?:положени|позици)\w*\s+ладь",
                lowered,
            )
        )
        if unsupported_castling_strategy_claim:
            errors.append(
                "Ответ добавляет стратегический эффект рокировки, "
                "которого нет среди переданных проверенных фактов."
            )

    # --------------------------------------------------------
    # ПЕДАГОГИЧЕСКИЕ ВЫВОДЫ
    # --------------------------------------------------------
    if (
        re.search(
            r"контрол\w*\s+центр|влияни\w*\s+на\s+центр|"
            r"занима\w*\s+центр|борьб\w*\s+за\s+центр",
            lowered,
        )
        and not grounding["center_supported"]
    ):
        errors.append(
            "Вывод о центре не подтверждён compact grounding."
        )

    if (
        "развит" in lowered
        and not grounding["development_supported"]
    ):
        errors.append(
            "Вывод о развитии фигур не подтверждён compact grounding."
        )

    if (
        re.search(
            r"открыва\w*\s+(?:лини|диагон)|"
            r"освобожда\w*\s+(?:лини|диагон)",
            lowered,
        )
        and not grounding["line_opening_supported"]
    ):
        errors.append(
            "Утверждение об открывшейся линии/диагонали не подтверждено."
        )

    # --------------------------------------------------------
    # НЕСТАНДАРТНАЯ ОПИСАТЕЛЬНАЯ НОТАЦИЯ.
    # GigaChess иногда генерирует QB2-KR2 / KB1 и похожие обозначения.
    # Для нашего API допустимы только обычные клетки a1-h8 и UCI/SAN.
    # --------------------------------------------------------
    descriptive_notation = re.findall(
        r"(?<![A-Za-zА-Яа-я0-9])"
        r"[KQRBN]{1,3}[1-8]"
        r"(?![A-Za-zА-Яа-я0-9])",
        text,
    )

    if descriptive_notation:
        errors.append(
            "Ответ использует неподдерживаемую описательную шахматную "
            f"нотацию {sorted(set(descriptive_notation))}. "
            "Используй только обычные координаты a1-h8 или естественный текст."
        )

    # --------------------------------------------------------
    # ФОРМАТ: мягче, чем раньше.
    # Одно содержательное предложение допустимо; факты важнее формы.
    # --------------------------------------------------------
    if len(text) > 1400:
        errors.append(
            f"Ответ слишком длинный ({len(text)} символов)."
        )

    if len(sentences) > 5:
        errors.append(
            f"Ответ слишком раздроблен: {len(sentences)} предложений."
        )

    if any(token in text for token in ("```", "{", "}")):
        errors.append(
            "Нужен обычный текст без Markdown/JSON."
        )

    errors = list(dict.fromkeys(errors))
    return not errors, errors
