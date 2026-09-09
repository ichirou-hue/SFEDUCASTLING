import sys
from pathlib import Path

# 1. Фикс путей импорта
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import re
from typing import Any, Dict, List, Set, Tuple

import chess
from loguru import logger
from tqdm import tqdm
from src.schemas import GeneratedExample

PIECE_NAMES = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король",
}


def _get_board_grounding(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    """Формирует полный белый список: занятые поля, поля атак и лучи дальнобойных фигур."""
    # 1. Все занятые поля на доске
    all_occupied_squares = {chess.square_name(sq) for sq in board.piece_map()}

    # 2. Все поля атак ВСЕХ фигур на доске (поля контроля/угроз)
    all_attacked_squares = set()
    for sq in board.piece_map():
        all_attacked_squares.update(
            chess.square_name(attack_sq) for attack_sq in board.attacks(sq)
        )

    # 3. Полные лучи дальнобойных фигур (диагонали a2-g8, b3-f7, вертикали и т.д.)
    ray_squares = set()
    for sq, piece in board.piece_map().items():
        if piece.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            for target_sq in chess.SQUARES:
                if chess.SquareSet.ray(sq, target_sq):
                    ray_squares.add(chess.square_name(target_sq))

    # 4. Поля, связанные непосредственно с ходом
    board_after = board.copy()
    board_after.push(move)

    move_related_squares = {
        chess.square_name(move.from_square),
        chess.square_name(move.to_square),
    }
    move_related_squares.update(
        chess.square_name(sq) for sq in board_after.attacks(move.to_square)
    )

    # Объединенный глобальный контекст доски (допустим для любых текстовых полей)
    valid_board_context = (
        all_occupied_squares
        | all_attacked_squares
        | ray_squares
        | move_related_squares
        | {"d4", "e4", "d5", "e5", "f7", "f2"}
    )

    return {
        "valid_board_context": valid_board_context,
        "is_capture": board.is_capture(move),
        "is_check": board_after.is_check(),
        "is_checkmate": board_after.is_checkmate(),
        "is_game_over": board_after.is_game_over(claim_draw=True),
    }


def _validate_field_text(text: str, field_name: str, grounding: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    lowered = text.lower()

    # 1. Проверка координат
    used_squares = set(re.findall(r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])", lowered))
    invalid_squares = used_squares - grounding["valid_board_context"]
    if invalid_squares:
        errors.append(f"Упомянуты несуществующие на доске поля: {sorted(invalid_squares)}")

    # 2. Проверка взятий (только если поле относится к разбору конкретного хода)
    if field_name == "why_best":
        capture_words = bool(re.search(r"\b(?:бер(?:[её]т|ут)|взял\w*|забира\w*|снима\w*|бь(?:[её]т|ют))\b", lowered))
        if not grounding["is_capture"] and capture_words:
            if not any(neg in lowered for neg in ["без взятия", "взятия нет", "не берёт", "не забирает", "не спешите менять"]):
                errors.append("Текст хода why_best утверждает взятие, которого нет в ходе.")

        # 3. Проверка шаха/мата
        has_mate_word = bool(re.search(r"\bмат(?:а|у|ом|е)?\b|\bматует\b", lowered))
        if has_mate_word and not grounding["is_checkmate"]:
            if not any(neg in lowered for neg in ["не мат", "защита от мата", "угроза мата", "потенциал"]):
                errors.append("В тексте why_best заявлен мат, но позиция не является матовой.")

    # 4. Проверка ложных заявлений о завершении партии
    if re.search(r"(?:партия|игра)\s+(?:сразу\s+)?(?:заканчивается|завершена|выиграна)", lowered):
        if not grounding["is_game_over"]:
            errors.append("Заявлено завершение партии, но игра продолжается.")

    # 5. Проверка технических артефактов
    if any(tok in text for tok in ["```", "json", "undefined", "null"]):
        errors.append("Обнаружен технический мусор форматирования.")

    return len(errors) == 0, errors


def validate_chess_record(record: GeneratedExample) -> Tuple[bool, List[str]]:
    errors = []

    # 1. Валидация FEN
    try:
        board = chess.Board(record.fen)
        if not board.is_valid():
            return False, [f"Невалидный FEN: {record.fen}"]
    except Exception as e:
        return False, [f"Ошибка парсинга FEN: {e}"]

    # 2. Валидация хода
    try:
        try:
            best_move_obj = board.parse_san(record.best_move)
        except ValueError:
            best_move_obj = chess.Move.from_uci(record.best_move)

        if best_move_obj not in board.legal_moves:
            return False, [f"Ход {record.best_move} нелегален в позиции."]
    except Exception as e:
        return False, [f"Синтаксическая ошибка в ходе: {e}"]

    # 3. Валидация схемы
    exp = record.coach_explanation
    min_length_checks = {
        "position_summary": (exp.position_summary, 20),
        "root_problem": (exp.root_problem, 15),
        "why_best": (exp.why_best, 20),
        "strategic_concept": (exp.strategic_concept, 10),
        "main_line": (exp.main_line, 10),
        "practical_advice": (exp.practical_advice, 15),
    }
    for f_name, (val, min_l) in min_length_checks.items():
        if not val or len(val.strip()) < min_l:
            errors.append(f"Поле '{f_name}' слишком короткое (< {min_l} симв.)")

    if errors:
        return False, errors

    # 4. Семантический аудит полей
    grounding = _get_board_grounding(board, best_move_obj)
    for field in ["why_best", "position_summary", "root_problem"]:
        val = getattr(exp, field, "")
        ok, f_errors = _validate_field_text(val, field, grounding)
        if not ok:
            for err in f_errors:
                errors.append(f"[{field}] {err}")

    return len(errors) == 0, errors


def run_validation(input_path: str, output_path: str, rejected_path: str):
    in_file = Path(input_path)
    if not in_file.exists():
        logger.error(f"Файл {input_path} не найден!")
        return

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rej_file = Path(rejected_path)
    rej_file.parent.mkdir(parents=True, exist_ok=True)

    valid_count, rej_count = 0, 0
    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout, \
         open(rej_file, "w", encoding="utf-8") as frej:

        for line in tqdm(fin, desc="Семантическая валидация"):
            line = line.strip()
            if not line:
                continue
            try:
                record = GeneratedExample.model_validate_json(line)
            except Exception as e:
                frej.write(json.dumps({"raw": line, "errors": [str(e)]}, ensure_ascii=False) + "\n")
                rej_count += 1
                continue

            is_valid, errs = validate_chess_record(record)
            if is_valid:
                fout.write(record.model_dump_json() + "\n")
                valid_count += 1
            else:
                data = record.model_dump()
                data["validation_errors"] = errs
                frej.write(json.dumps(data, ensure_ascii=False) + "\n")
                rej_count += 1

    logger.success(f"Валидно: {valid_count} | Отклонено: {rej_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/03_generated/master_explanations.jsonl")
    parser.add_argument("--out", default="data/04_validated/master_validated.jsonl")
    parser.add_argument("--rejected", default="data/04_validated/rejected.jsonl")
    args = parser.parse_args()
    run_validation(args.input, args.out, args.rejected)