from backend.llm.explainer.move_facts import _join_squares
from backend.llm.explainer.names import (
    PIECE_NAMES,
    _piece_name_accusative,
    _piece_word_accusative,
)


def _build_gigachess_grounding(
    *,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
) -> dict:
    """
    Semantic fact compressor.

    Python может знать десятки точных клеток и отношений, но GigaChess
    получает только небольшой набор наиболее полезных фактов.

    ВАЖНО:
    - подробные move_facts/position_facts остаются внутри Python;
    - в prompt попадает максимум 5 коротких фактов;
    - validator разрешает конкретные клетки/фигуры только из этого
      сжатого контекста, а не из всего внутреннего анализа.
    """
    facts: list[str] = []
    allowed_squares: set[str] = set()
    allowed_piece_words: set[str] = set()
    moved_piece_controls: set[str] = set()
    attack_squares: set[str] = set()
    defended_squares: set[str] = set()

    from_square = str(explained_move["from"]).lower()
    to_square = str(explained_move["to"]).lower()
    uci = str(explained_move["uci"]).lower()
    piece_text = str(explained_move["piece"])
    piece_lower = piece_text.lower()

    allowed_squares.update({from_square, to_square})

    castling_rook_from = None
    castling_rook_to = None

    if move_facts.get("is_castling"):
        castling_map = {
            ("e1", "g1"): ("h1", "f1"),
            ("e1", "c1"): ("a1", "d1"),
            ("e8", "g8"): ("h8", "f8"),
            ("e8", "c8"): ("a8", "d8"),
        }
        rook_move = castling_map.get((from_square, to_square))
        if rook_move:
            castling_rook_from, castling_rook_to = rook_move
            allowed_squares.update(
                {castling_rook_from, castling_rook_to}
            )
            allowed_piece_words.add("ладья")

    moved_piece_word = next(
        (
            name
            for name in PIECE_NAMES.values()
            if name in piece_lower
        ),
        None,
    )
    if moved_piece_word:
        allowed_piece_words.add(moved_piece_word)

    mover_is_white = move_facts.get("piece_color") == "white"
    mover_side = "белых" if mover_is_white else "чёрных"
    terminal = derived_facts.get("terminal") or {}

    promotion_piece_word = None
    promotion_piece_acc = None

    if move_facts.get("is_promotion"):
        raw_promotion_piece = str(
            move_facts.get("promotion_piece") or ""
        ).lower()

        promotion_piece_word = next(
            (
                name
                for name in PIECE_NAMES.values()
                if name in raw_promotion_piece
            ),
            raw_promotion_piece or "фигура",
        )

        promotion_piece_acc = _piece_word_accusative(
            promotion_piece_word
        )
        allowed_piece_words.add(promotion_piece_word)

    # После promotion шах/мат даёт уже новая фигура, а не пешка.
    post_move_piece_word = (
        promotion_piece_word
        if move_facts.get("is_promotion")
        else moved_piece_word
    )

    movement_type = str(
        move_facts.get("movement_type") or ""
    ).lower()

    movement_geometry_phrase = ""

    if (
        movement_type == "diagonal"
        and moved_piece_word in {"слон", "ферзь"}
    ):
        movement_geometry_phrase = " по диагонали"

    elif (
        movement_type == "horizontal"
        and moved_piece_word in {"ладья", "ферзь"}
    ):
        movement_geometry_phrase = " по горизонтали"

    elif (
        movement_type == "vertical"
        and moved_piece_word in {"ладья", "ферзь"}
    ):
        movement_geometry_phrase = " по вертикали"

    elif (
        movement_type == "knight_jump"
        and moved_piece_word == "конь"
    ):
        movement_geometry_phrase = " ходом буквой «Г»"

    elif (
        movement_type == "king_step"
        and moved_piece_word == "король"
    ):
        movement_geometry_phrase = " на соседнюю клетку"

    elif (
        movement_type == "pawn"
        and moved_piece_word == "пешка"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        movement_geometry_phrase = " по диагонали вперёд на одну клетку"

    elif (
        movement_type == "pawn"
        and moved_piece_word == "пешка"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        movement_geometry_phrase = " прямо вперёд на одну клетку"

    def add_fact(value: str) -> None:
        value = value.strip()
        if value and value not in facts and len(facts) < 5:
            facts.append(value)

    # 1. Сам ход. Для взятия сразу формулируем завершённое событие,
    # чтобы GigaChess не превращал его в "может взять".
    if move_facts.get("is_capture"):
        captured_piece = str(
            move_facts.get("captured_piece") or "фигура соперника"
        )
        captured_square = str(
            move_facts.get("captured_square") or to_square
        ).lower()

        allowed_squares.add(captured_square)

        for name in PIECE_NAMES.values():
            if name in captured_piece.lower():
                allowed_piece_words.add(name)

        captured_acc = _piece_name_accusative(captured_piece)

        if move_facts.get("is_en_passant"):
            add_fact(
                f"{piece_text.capitalize()} идёт с {from_square} на {to_square} "
                f"взятием на проходе и этим ходом снимает {captured_acc} "
                f"с поля {captured_square}."
            )
        else:
            move_fact = (
                f"{piece_text.capitalize()} идёт с {from_square} на {to_square}"
                f"{movement_geometry_phrase} и этим ходом берёт "
                f"{captured_acc} на {captured_square}"
            )
            if move_facts.get("is_promotion") and promotion_piece_acc:
                move_fact += (
                    f", после чего на {to_square} превращается "
                    f"в {promotion_piece_acc}"
                )
            move_fact += "."
            add_fact(move_fact)
    else:
        move_fact = (
            f"{piece_text.capitalize()} идёт с {from_square} на {to_square}"
            f"{movement_geometry_phrase}"
        )
        if move_facts.get("pawn_double_step"):
            move_fact += " двойным ходом"
        if move_facts.get("is_promotion") and promotion_piece_acc:
            move_fact += (
                f" и на {to_square} превращается "
                f"в {promotion_piece_acc}"
            )
        move_fact += "."
        add_fact(move_fact)

    # 2. Остальные критические события хода.

    opponent_king_square = terminal.get("opponent_king_square")

    if move_facts.get("is_checkmate"):
        allowed_piece_words.add("король")
        if opponent_king_square:
            allowed_squares.add(opponent_king_square)
            mate_result_text = ""
            if terminal.get("result") == "1-0":
                mate_result_text = " Партия сразу заканчивается победой белых."
            elif terminal.get("result") == "0-1":
                mate_result_text = " Партия сразу заканчивается победой чёрных."

            add_fact(
                f"После этого хода {post_move_piece_word or 'фигура'} на {to_square} "
                f"ставит мат королю соперника на {opponent_king_square}; "
                f"у соперника нет легальных ходов."
                f"{mate_result_text}"
            )
        else:
            add_fact(
                "После этого хода королю соперника поставлен мат, "
                "и у соперника нет легальных ходов."
            )
    elif move_facts.get("is_check"):
        allowed_piece_words.add("король")
        if opponent_king_square:
            allowed_squares.add(opponent_king_square)
            add_fact(
                f"После этого хода {post_move_piece_word or 'фигура'} на {to_square} "
                f"объявляет шах королю соперника на {opponent_king_square}."
            )
        else:
            add_fact("После хода королю соперника объявлен шах.")

    if move_facts.get("is_castling"):
        if to_square in {"g1", "g8"}:
            castling_name = "короткая рокировка"
        elif to_square in {"c1", "c8"}:
            castling_name = "длинная рокировка"
        else:
            castling_name = "рокировка"

        if castling_rook_from and castling_rook_to:
            add_fact(
                f"Этим ходом выполняется {castling_name}: "
                f"король переходит с {from_square} на {to_square}, "
                f"а ладья одновременно переходит с "
                f"{castling_rook_from} на {castling_rook_to}."
            )
        else:
            add_fact(
                f"Этим ходом выполняется {castling_name}."
            )

    # Терминальный результат партии добавляем только если он реально
    # наступает сразу после этого хода.
    if (
        not move_facts.get("is_checkmate")
        and terminal.get("is_game_over")
    ):
        if terminal.get("is_stalemate"):
            add_fact(
                "После этого хода возникает пат, поэтому партия "
                "сразу заканчивается вничью."
            )
        elif terminal.get("result") == "1/2-1/2":
            add_fact(
                "После этого хода партия сразу заканчивается вничью."
            )
        elif terminal.get("result") == "1-0":
            add_fact(
                "После этого хода партия сразу заканчивается победой белых."
            )
        elif terminal.get("result") == "0-1":
            add_fact(
                "После этого хода партия сразу заканчивается победой чёрных."
            )

    # 3. Конкретный контроль перемещённой фигуры.
    # Не передаём модели длинные списки. Максимум 3 клетки.
    candidate_controls = [
        str(s).lower()
        for s in (
            move_facts.get("new_controls")
            or move_facts.get("controls_after")
            or []
        )
        if isinstance(s, str)
    ]

    # Приоритет центральным полям, затем остальным.
    center_names = {"d4", "e4", "d5", "e5"}
    ordered_controls = sorted(
        dict.fromkeys(candidate_controls),
        key=lambda s: (s not in center_names, s),
    )
    selected_controls = ordered_controls[:3]

    has_critical_event = bool(
        move_facts.get("is_capture")
        or move_facts.get("is_check")
        or move_facts.get("is_checkmate")
        or move_facts.get("is_castling")
        or move_facts.get("is_promotion")
    )

    if (
        selected_controls
        and not has_critical_event
        and len(facts) < 5
    ):
        moved_piece_controls.update(selected_controls)
        allowed_squares.update(selected_controls)
        add_fact(
            "После хода перемещённая фигура контролирует "
            f"{_join_squares(selected_controls)}."
        )

    # 4. Педагогический вывод о центре без новых координат.
    center_supported = bool(
        move_facts.get("occupies_center")
        or move_facts.get("new_center_controls")
    )
    if center_supported and len(facts) < 5 and not has_critical_event:
        add_fact(f"Ход усиливает влияние {mover_side} на центр.")

    # 5. Педагогический вывод о развитии без перечисления открывшихся клеток.
    line_openings = derived_facts.get("line_openings", [])
    development_supported = bool(
        not move_facts.get("is_castling")
        and (
            move_facts.get("knight_develops")
            or line_openings
        )
    )
    if development_supported and len(facts) < 5:
        add_fact(f"Ход помогает дальнейшему развитию фигур {mover_side}.")

    # Если ещё есть место — одна конкретная новая атака ИЛИ защита.
    # Для critical-event ходов этот слой подавляем: он часто дублирует
    # шах/мат и добавляет лишние сущности.
    if len(facts) < 5 and not has_critical_event:
        newly_attacked = [
            item
            for item in position_facts.get("newly_attacked_pieces", [])
            if (
                isinstance(item, dict)
                and item.get("piece")
                and item.get("square")
            )
        ]
        if newly_attacked:
            item = newly_attacked[0]
            square = str(item["square"]).lower()
            piece = str(item["piece"])
            allowed_squares.add(square)
            attack_squares.add(square)
            for name in PIECE_NAMES.values():
                if name in piece.lower():
                    allowed_piece_words.add(name)
            add_fact(
                f"После хода под атакой оказывается {piece} на {square}."
            )

    if len(facts) < 5 and not has_critical_event:
        newly_defended = [
            item
            for item in position_facts.get("newly_defended", [])
            if (
                isinstance(item, dict)
                and item.get("piece")
                and item.get("square")
            )
        ]
        if newly_defended:
            item = newly_defended[0]
            square = str(item["square"]).lower()
            piece = str(item["piece"])
            allowed_squares.add(square)
            defended_squares.add(square)
            for name in PIECE_NAMES.values():
                if name in piece.lower():
                    allowed_piece_words.add(name)
            add_fact(
                f"Ход усиливает защиту {piece} на {square}."
            )

    # Для тихих ходов полезно явно зафиксировать отсутствие критических
    # событий, но только если это не вытесняет более полезные позитивные факты.
    if (
        len(facts) < 5
        and not move_facts.get("is_capture")
        and not move_facts.get("is_check")
        and not move_facts.get("is_checkmate")
    ):
        add_fact("В этом ходе нет взятия, шаха или мата.")

    return {
        "facts": facts,
        "text": "\n".join(f"- {fact}" for fact in facts),
        "uci": uci,
        "from": from_square,
        "to": to_square,
        "moved_piece_word": moved_piece_word,
        "movement_type": movement_type,
        "pawn_capture_geometry_required": bool(
            movement_type == "pawn"
            and moved_piece_word == "пешка"
            and move_facts.get("is_capture")
            and not move_facts.get("is_en_passant")
            and not move_facts.get("is_promotion")
        ),
        "pawn_single_step_geometry_required": bool(
            movement_type == "pawn"
            and moved_piece_word == "пешка"
            and not move_facts.get("is_capture")
            and not move_facts.get("is_en_passant")
            and not move_facts.get("is_promotion")
            and not move_facts.get("pawn_double_step")
        ),
        "movement_geometry_required": bool(
            (
                movement_type == "diagonal"
                and moved_piece_word in {"слон", "ферзь"}
            )
            or (
                movement_type in {"horizontal", "vertical"}
                and moved_piece_word in {"ладья", "ферзь"}
            )
            or (
                movement_type == "knight_jump"
                and moved_piece_word == "конь"
            )
            or (
                movement_type == "king_step"
                and moved_piece_word == "король"
            )
            or (
                movement_type == "pawn"
                and moved_piece_word == "пешка"
                and move_facts.get("is_capture")
                and not move_facts.get("is_en_passant")
                and not move_facts.get("is_promotion")
            )
            or (
                movement_type == "pawn"
                and moved_piece_word == "пешка"
                and not move_facts.get("is_capture")
                and not move_facts.get("is_en_passant")
                and not move_facts.get("is_promotion")
                and not move_facts.get("pawn_double_step")
            )
        ),
        "allowed_squares": allowed_squares,
        "allowed_piece_words": allowed_piece_words,
        "moved_piece_controls": moved_piece_controls,
        "attack_squares": attack_squares,
        "defended_squares": defended_squares,
        "center_supported": center_supported,
        "development_supported": development_supported,
        "line_opening_supported": bool(line_openings),
        "terminal": terminal,
        "opponent_king_square": opponent_king_square,
        "opponent_legal_moves": int(
            position_facts.get("opponent_legal_moves") or 0
        ),
        "castling_rook_from": castling_rook_from,
        "castling_rook_to": castling_rook_to,
        "king_safety_supported": False,
        "generic_defense_supported": bool(defended_squares),
        "is_capture": bool(move_facts.get("is_capture")),
        "is_check": bool(move_facts.get("is_check")),
        "is_checkmate": bool(move_facts.get("is_checkmate")),
        "is_castling": bool(move_facts.get("is_castling")),
        "is_promotion": bool(move_facts.get("is_promotion")),
    }
