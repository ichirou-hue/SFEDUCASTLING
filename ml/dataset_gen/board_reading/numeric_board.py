"""Version 1 of the numeric board prompt used for SFT and inference."""

import chess


def render_board(query, include_state):
    board = chess.Board(query['fen'])
    lines = [
        'Числовая доска. Положительные числа — белые, отрицательные — чёрные.',
        'По модулю: 0 — пусто, 1 — пешка, 2 — конь, 3 — слон, '
        '4 — ладья, 5 — ферзь, 6 — король.',
        ' ' + ''.join(f'{file:>3}' for file in chess.FILE_NAMES),
    ]
    for rank in reversed(range(8)):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.piece_type * (1 if piece.color else -1) if piece else 0)
        lines.append(str(rank + 1) + ''.join(f'{cell:>3}' for cell in cells))
    if include_state:
        lines.extend([
        f'Ходят {"белые" if board.turn else "чёрные"}.',
        f'Права рокировки: {board.castling_xfen()}.',
        'Поле взятия на проходе: ' + (
            chess.square_name(board.ep_square) if board.ep_square is not None else '-'
        ) + '.',
        f'Полуходов без взятия или хода пешкой: {board.halfmove_clock}. Номер хода: {board.fullmove_number}.',
        ])
    lines.extend(['', query['prompt']])
    return '\n'.join(lines)


def format_query(query):
    return render_board(query, include_state=True)


def format_cell_query(query):
    return render_board(query, include_state=False)
