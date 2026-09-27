import re
import time

import chess

from backend.api_gateway.models import ExplainMoveRequest
from backend.api_gateway.routes.game import _maia3_move_strict
from backend.api_gateway.state import get_gigachess
from backend.llm.gigachess import GigachessError
from backend.llm.explainer.constants import (
    GIGACHESS_INPUT_TYPE,
    GIGACHESS_TRANSPORT_ATTEMPTS,
    GIGACHESS_TRANSPORT_BACKOFF_SECONDS,
    MAX_GIGACHESS_ATTEMPTS,
)
from backend.llm.explainer.derived_facts import _build_derived_explanation_facts
from backend.llm.explainer.deterministic import _deterministic_explanation
from backend.llm.explainer.grounding import _build_gigachess_grounding
from backend.llm.explainer.move_facts import _move_facts, _move_info
from backend.llm.explainer.position_facts import _position_change_facts
from backend.llm.explainer.prompts import (
    _build_prompt,
    _build_verified_explanation_context,
)
from backend.llm.explainer.repair import (
    _build_repair_prompt,
    _materialize_protected_capture_promotion_answer,
    _materialize_protected_castling_answer,
    _materialize_protected_en_passant_answer,
    _materialize_protected_promotion_answer,
)
from backend.llm.explainer.stockfish import _stockfish_analysis
from backend.llm.explainer.validation import _validate_llm_explanation


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def explain_move(
    req: ExplainMoveRequest,
    played_move: str | None = None,
) -> dict:
    # ========================================================
    # FEN
    # ========================================================
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        return {
            "ok": False,
            "error": f"Некорректный FEN: {e}",
        }

    fen_before = board.fen()

    # ========================================================
    # ХОД ПОЛЬЗОВАТЕЛЯ — только для API-ответа
    # ========================================================
    actual_played_move = played_move or getattr(req, "move", None)
    played_move_info = None

    if actual_played_move:
        try:
            played_chess_move = chess.Move.from_uci(actual_played_move)
        except ValueError:
            return {
                "ok": False,
                "error": f"Некорректный UCI ход: {actual_played_move}",
            }

        if played_chess_move not in board.legal_moves:
            return {
                "ok": False,
                "error": (
                    f"Ход {actual_played_move} нелегален для переданного FEN."
                ),
            }

        played_move_info = _move_info(board, played_chess_move)

    # ========================================================
    # STOCKFISH — единственный источник объясняемого хода
    # ========================================================
    stockfish = _stockfish_analysis(board)
    if not stockfish.get("available"):
        return {
            "ok": False,
            "error": "Stockfish недоступен.",
            "played_move": played_move_info,
            "stockfish": stockfish,
        }

    stockfish_move_info = stockfish.get("best_move")
    if not stockfish_move_info:
        return {
            "ok": False,
            "error": "Stockfish не вернул лучший ход.",
            "played_move": played_move_info,
            "stockfish": stockfish,
        }

    try:
        stockfish_move = chess.Move.from_uci(stockfish_move_info["uci"])
    except (ValueError, KeyError):
        return {
            "ok": False,
            "error": "Stockfish вернул некорректный ход.",
        }

    if stockfish_move not in board.legal_moves:
        return {
            "ok": False,
            "error": "Ход Stockfish отсутствует среди легальных ходов.",
        }

    stockfish_move_info = _move_info(board, stockfish_move)
    if stockfish_move_info is None:
        return {
            "ok": False,
            "error": "Не удалось получить информацию о ходе Stockfish.",
        }

    # ========================================================
    # ПОЗИЦИЯ ПОСЛЕ ХОДА + ПРОВЕРЕННЫЕ ФАКТЫ
    # ========================================================
    board_after = board.copy()
    board_after.push(stockfish_move)
    fen_after = board_after.fen()

    move_facts = _move_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )
    position_facts = _position_change_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )
    derived_facts = _build_derived_explanation_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )

    print("[ChessExplainer] === STOCKFISH MOVE ===")
    print(f"[ChessExplainer] played_move={played_move_info}")
    print(f"[ChessExplainer] stockfish_move={stockfish_move_info}")
    print(f"[ChessExplainer] FEN before={fen_before}")
    print(f"[ChessExplainer] FEN after={fen_after}")

    # ========================================================
    # MAIA3 — не влияет на объясняемый ход
    # ========================================================
    try:
        maia_move = _maia3_move_strict(req, board.copy())
    except Exception as e:
        print(f"[ChessExplainer] Maia3 error: {e}")
        maia_move = None

    maia3 = {
        "available": maia_move is not None,
        "move": _move_info(board, maia_move),
        "elo": req.elo,
    }
    same_as_maia3 = (
        maia_move is not None
        and maia_move == stockfish_move
    )

    deterministic_explanation = _deterministic_explanation(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
        facts=move_facts,
    )

    verified_context = _build_verified_explanation_context(
        explained_move=stockfish_move_info,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )
    gigachess_grounding = _build_gigachess_grounding(
        explained_move=stockfish_move_info,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    print("[ChessExplainer] === GIGACHESS INPUT FORMAT ===")
    print("[ChessExplainer] FEN is duplicated: content + attachments")
    print(f"[ChessExplainer] FEN in content={fen_before}")
    print("[ChessExplainer] === COMPACT GROUNDING SENT TO GIGACHESS ===")
    print(verified_context)

    base_result = {
        "ok": True,
        "played_move": played_move_info,
        "stockfish": stockfish,
        "maia3": maia3,
        "same_as_maia3": same_as_maia3,
        "fen_before": fen_before,
        "fen_after": fen_after,
        "move_facts": move_facts,
        "position_facts": position_facts,
        "derived_facts": derived_facts,
        "explained_move": stockfish_move_info,
        "gigachess_grounding": {
            "facts": gigachess_grounding["facts"],
            "allowed_squares": sorted(gigachess_grounding["allowed_squares"]),
            "allowed_piece_words": sorted(
                gigachess_grounding["allowed_piece_words"]
            ),
            "moved_piece_controls": sorted(
                gigachess_grounding["moved_piece_controls"]
            ),
            "attack_squares": sorted(gigachess_grounding["attack_squares"]),
            "defended_squares": sorted(
                gigachess_grounding["defended_squares"]
            ),
            "terminal": gigachess_grounding["terminal"],
            "opponent_king_square": gigachess_grounding[
                "opponent_king_square"
            ],
            "opponent_legal_moves": gigachess_grounding[
                "opponent_legal_moves"
            ],
            "castling_rook_from": gigachess_grounding[
                "castling_rook_from"
            ],
            "castling_rook_to": gigachess_grounding[
                "castling_rook_to"
            ],
            "king_safety_supported": gigachess_grounding[
                "king_safety_supported"
            ],
            "generic_defense_supported": gigachess_grounding[
                "generic_defense_supported"
            ],
        },
    }

    # ========================================================
    # GIGACHESS + ITERATIVE REPAIR LOOP
    # ========================================================
    client = get_gigachess()
    if client is None:
        base_result["gigachess"] = {
            "available": False,
            "used": False,
            "retry": False,
            "attempt_count": 0,
            "input_type": GIGACHESS_INPUT_TYPE,
        }
        base_result["explanation_source"] = "deterministic"
        base_result["explanation"] = deterministic_explanation
        return base_result

    from backend.config.settings import settings

    attempts_debug: list[dict] = []
    previous_answer: str | None = None
    validation_errors: list[str] = []

    for attempt in range(1, MAX_GIGACHESS_ATTEMPTS + 1):
        if attempt == 1:
            generation_mode = "initial_compact"
            messages = _build_prompt(
                fen=fen_before,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
                elo=req.elo,
            )
            temperature = 0.0
            top_p = 1.0
            max_tokens = min(
                int(settings.gigachess.max_tokens),
                280,
            )
        else:
            if move_facts.get("is_castling"):
                generation_mode = "protected_castling_rewrite"
            elif move_facts.get("is_en_passant"):
                generation_mode = "protected_en_passant_rewrite"
            elif (
                move_facts.get("is_promotion")
                and move_facts.get("is_capture")
            ):
                generation_mode = "protected_capture_promotion_rewrite"
            elif move_facts.get("is_promotion"):
                generation_mode = "protected_promotion_rewrite"
            elif any([
                move_facts.get("is_capture"),
                move_facts.get("is_check"),
                move_facts.get("is_checkmate"),
                move_facts.get("is_en_passant"),
                gigachess_grounding.get("movement_geometry_required"),
                gigachess_grounding.get("pawn_capture_geometry_required"),
                gigachess_grounding.get("pawn_single_step_geometry_required"),
            ]):
                generation_mode = "strict_special_rewrite"
            else:
                generation_mode = "compact_repair"
            messages = _build_repair_prompt(
                fen=fen_before,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
                elo=req.elo,
                previous_answer=previous_answer,
                validation_errors=validation_errors,
                attempt_number=attempt - 1,
            )
            repair_temperatures = {
                2: 0.12,
                3: 0.18,
                4: 0.22,
            }
            temperature = repair_temperatures.get(attempt, 0.18)
            top_p = 1.0
            max_tokens = 280

        print(
            "[ChessExplainer] === GIGACHESS SEMANTIC ATTEMPT "
            f"{attempt}/{MAX_GIGACHESS_ATTEMPTS} ==="
        )

        answer = None
        transport_debug: list[dict] = []
        transport_errors: list[str] = []

        for transport_attempt in range(
            1,
            GIGACHESS_TRANSPORT_ATTEMPTS + 1,
        ):
            print(
                "[ChessExplainer] --- transport attempt "
                f"{transport_attempt}/"
                f"{GIGACHESS_TRANSPORT_ATTEMPTS} "
                f"for semantic attempt {attempt} ---"
            )

            if transport_attempt > 1:
                delay_index = min(
                    transport_attempt - 1,
                    len(GIGACHESS_TRANSPORT_BACKOFF_SECONDS) - 1,
                )
                delay = GIGACHESS_TRANSPORT_BACKOFF_SECONDS[
                    delay_index
                ]
                if delay > 0:
                    print(
                        "[ChessExplainer] transport backoff: "
                        f"{delay:.1f}s"
                    )
                    time.sleep(delay)

            try:
                answer = client.chat(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                transport_debug.append({
                    "transport_attempt": transport_attempt,
                    "ok": True,
                    "error": None,
                })
                break

            except GigachessError as e:
                error_message = (
                    f"Gigachess request error: {e}"
                )
                print(
                    "[ChessExplainer] "
                    f"{error_message}"
                )
                transport_errors.append(error_message)
                transport_debug.append({
                    "transport_attempt": transport_attempt,
                    "ok": False,
                    "error": error_message,
                })

        if answer is None:
            validation_errors = transport_errors or [
                "Gigachess transport failed without response."
            ]
            previous_answer = None

            attempts_debug.append({
                "attempt": attempt,
                "mode": generation_mode,
                "temperature": temperature,
                "answer": None,
                "valid": False,
                "errors": validation_errors,
                "transport_attempt_count": len(transport_debug),
                "transport_attempts": transport_debug,
            })

            print(
                "[ChessExplainer] Transport retries exhausted "
                "for current semantic attempt. "
                "Semantic prompt was not evaluated."
            )

            # Не переходим к следующей semantic attempt:
            # без ответа модели нет причины менять repair prompt
            # или температуру. Сразу используем deterministic fallback.
            break

        print("[ChessExplainer] GIGACHESS RESPONSE:")
        print(repr(answer))

        raw_answer = answer
        protected_errors: list[str] = []

        if generation_mode == "protected_castling_rewrite":
            current_grounding = _build_gigachess_grounding(
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
            )
            answer, protected_errors = _materialize_protected_castling_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                grounding=current_grounding,
            )

            if not protected_errors:
                print("[ChessExplainer] PROTECTED CASTLING MATERIALIZED:")
                print(repr(answer))

        elif generation_mode == "protected_en_passant_rewrite":
            answer, protected_errors = _materialize_protected_en_passant_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
            )

            if not protected_errors:
                print("[ChessExplainer] PROTECTED EN PASSANT MATERIALIZED:")
                print(repr(answer))

        elif generation_mode == "protected_capture_promotion_rewrite":
            answer, protected_errors = (
                _materialize_protected_capture_promotion_answer(
                    raw_answer,
                    explained_move=stockfish_move_info,
                    move_facts=move_facts,
                    derived_facts=derived_facts,
                )
            )

            if not protected_errors:
                print(
                    "[ChessExplainer] "
                    "PROTECTED CAPTURE PROMOTION MATERIALIZED:"
                )
                print(repr(answer))

        elif generation_mode == "protected_promotion_rewrite":
            answer, protected_errors = _materialize_protected_promotion_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                derived_facts=derived_facts,
            )

            if not protected_errors:
                print(
                    "[ChessExplainer] "
                    "PROTECTED PROMOTION MATERIALIZED:"
                )
                print(repr(answer))

        if protected_errors:
            valid = False
            validation_errors = protected_errors
        else:
            valid, validation_errors = _validate_llm_explanation(
                answer,
                stockfish_move_info,
                move_facts,
                position_facts,
                derived_facts,
            )

        attempt_debug = {
            "attempt": attempt,
            "mode": generation_mode,
            "temperature": temperature,
            "answer": answer,
            "valid": valid,
            "errors": validation_errors,
            "transport_attempt_count": len(transport_debug),
            "transport_attempts": transport_debug,
        }

        if generation_mode in {
            "protected_castling_rewrite",
            "protected_en_passant_rewrite",
            "protected_promotion_rewrite",
            "protected_capture_promotion_rewrite",
        }:
            attempt_debug["raw_answer"] = raw_answer

        attempts_debug.append(attempt_debug)

        if valid:
            base_result["gigachess"] = {
                "available": True,
                "used": True,
                "retry": (
                    attempt > 1
                    or len(transport_debug) > 1
                ),
                "attempt_count": attempt,
                "input_type": GIGACHESS_INPUT_TYPE,
                "attempts": attempts_debug,
            }
            base_result["explanation_source"] = (
                "gigachess"
                if attempt == 1
                else f"gigachess_repair_{attempt - 1}"
            )
            base_result["explanation"] = answer.strip()
            return base_result

        print("[ChessExplainer] GIGACHESS VALIDATION FAILED:")
        for index, error in enumerate(validation_errors, start=1):
            print(f"[ChessExplainer]   {index}. {error}")

        previous_answer = answer

    # ========================================================
    # FALLBACK ПОСЛЕ ВСЕХ НЕУДАЧНЫХ ПОПЫТОК
    # ========================================================
    print(
        "[ChessExplainer] Gigachess generation unavailable or all "
        "semantic attempts rejected. Using deterministic explanation."
    )

    base_result["gigachess"] = {
        "available": True,
        "used": False,
        "retry": True,
        "attempt_count": len(attempts_debug),
        "input_type": GIGACHESS_INPUT_TYPE,
        "attempts": attempts_debug,
        "validation_errors": validation_errors,
    }
    base_result["explanation_source"] = "deterministic"
    base_result["explanation"] = deterministic_explanation
    return base_result
