"""
Генерация обучающего датасета для LLM из шахматных партий.

Что делает:
1. Читает PGN файл (партии с Lichess)
2. Каждые 5 ходов берёт позицию
3. Прогоняет через Stockfish — получает оценку и лучший ход
4. Формирует текстовое описание позиции
5. Сохраняет в JSONL формат

Запуск:
    python prepare_dataset.py --pgn lichess_2024.pgn --stockfish ./stockfish --output dataset.jsonl --limit 10000
"""

import chess
import chess.pgn
import chess.engine
import json
import argparse


def detect_phase(board):
    pieces = len(board.piece_map())
    if board.fullmove_number <= 10 and pieces > 28:
        return "дебют"
    elif pieces <= 12:
        return "эндшпиль"
    return "миттельшпиль"


def eval_to_text(score):
    if score.is_mate():
        moves = score.mate()
        if moves > 0:
            return f"Мат в {moves} ходов за белых"
        else:
            return f"Мат в {abs(moves)} ходов за чёрных"
    cp = score.score()
    if cp is None:
        return "Оценка недоступна"
    pawns = cp / 100.0
    if abs(pawns) < 0.3:
        return f"Позиция равная ({pawns:+.1f})"
    elif pawns > 0:
        return f"Преимущество белых ({pawns:+.1f})"
    else:
        return f"Преимущество чёрных ({pawns:+.1f})"


def describe_position(board, info):
    lines = []

    score = info["score"].white()
    lines.append(eval_to_text(score) + ".")

    phase = detect_phase(board)
    lines.append(f"Стадия: {phase}.")

    turn = "Ход белых." if board.turn == chess.WHITE else "Ход чёрных."
    lines.append(turn)

    if board.is_check():
        lines.append("Королю объявлен шах.")

    pv = info.get("pv", [])
    if pv:
        best = board.san(pv[0])
        lines.append(f"Лучший ход: {best}.")

    white_pieces = 0
    black_pieces = 0
    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            white_pieces += 1
        else:
            black_pieces += 1
    lines.append(f"Фигур на доске: белых {white_pieces}, чёрных {black_pieces}.")

    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    lines.append(f"Пешек: белых {white_pawns}, чёрных {black_pawns}.")

    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    w_center = 0
    b_center = 0
    for sq in center:
        p = board.piece_at(sq)
        if p:
            if p.color == chess.WHITE:
                w_center += 1
            else:
                b_center += 1
    if w_center > b_center:
        lines.append("Белые контролируют центр.")
    elif b_center > w_center:
        lines.append("Чёрные контролируют центр.")
    else:
        lines.append("Центр оспаривается.")

    if board.has_kingside_castling_rights(chess.WHITE):
        lines.append("Белые могут рокироваться на королевский фланг.")
    if board.has_queenside_castling_rights(chess.WHITE):
        lines.append("Белые могут рокироваться на ферзевый фланг.")

    return " ".join(lines)


def process_pgn(pgn_path, stockfish_path, output_path, limit, depth):
    print(f"Открываю движок: {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 4, "Hash": 2048})

    count = 0
    games_read = 0

    print(f"Читаю партии из: {pgn_path}")
    print(f"Цель: {limit} позиций, глубина анализа: {depth}")

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file, \
         open(output_path, "w", encoding="utf-8") as out_file:

        while count < limit:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                print("PGN файл закончился.")
                break

            games_read += 1
            board = game.board()
            moves = list(game.mainline_moves())

            for i in range(0, len(moves)):
                board.push(moves[i])

                if (i + 1) % 5 != 0:
                    continue

                if count >= limit:
                    break

                if len(board.piece_map()) < 4:
                    continue

                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=depth))
                except Exception as e:
                    print(f"Ошибка анализа: {e}")
                    continue

                fen = board.fen()
                analysis = describe_position(board, info)

                entry = {
                    "instruction": f"Оцени шахматную позицию: {fen}",
                    "output": analysis,
                }

                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

                if count % 100 == 0:
                    print(f"  [{count}/{limit}] позиций обработано, партий прочитано: {games_read}")

    engine.quit()
    print(f"\nГотово! Сгенерировано {count} примеров -> {output_path}")
    print(f"Прочитано партий: {games_read}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация шахматного датасета для LLM")
    parser.add_argument("--pgn", required=True, help="Путь к PGN файлу с партиями (скачать с database.lichess.org)")
    parser.add_argument("--stockfish", required=True, help="Путь к бинарнику Stockfish")
    parser.add_argument("--output", default="dataset.jsonl", help="Выходной JSONL файл")
    parser.add_argument("--limit", type=int, default=10000, help="Сколько позиций сгенерировать")
    parser.add_argument("--depth", type=int, default=18, help="Глубина анализа Stockfish (18 = хорошо, 22 = долго но точно)")
    args = parser.parse_args()

    process_pgn(args.pgn, args.stockfish, args.output, args.limit, args.depth)
