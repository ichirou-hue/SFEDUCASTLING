import chess
import numpy as np
import torch

def fen_to_lc0_112_planes(fen: str) -> torch.Tensor:
    """
    Преобразует FEN в канонический тензор LC0 формы (1, 112, 8, 8).
    Для одиночной позиции история (предыдущие 7 шагов) заполняется нулями.
    """
    board = chess.Board(fen)
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    
    # Флаг инверсии: если ходят черные, доска разворачивается (с точки зрения текущего игрока)
    flip = not board.turn

    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]

    # Плоскости 0..11: текущие фигуры (6 своих, 6 чужих)
    # Если ход белых: 0..5 свои (белые), 6..11 чужие (черные)
    # Если ход черных: 0..5 свои (черные), 6..11 чужие (белые)
    our_color = board.turn
    their_color = not board.turn

    for i, p_type in enumerate(piece_order):
        for sq in board.pieces(p_type, our_color):
            r, c = divmod(sq, 8)
            r = 7 - r if flip else r
            c = c if not flip else 7 - c
            planes[i, r, c] = 1.0

        for sq in board.pieces(p_type, their_color):
            r, c = divmod(sq, 8)
            r = 7 - r if flip else r
            c = c if not flip else 7 - c
            planes[i + 6, r, c] = 1.0

    # Плоскость 12: повторение текущей позиции (для одной позиции 0)
    # Плоскости 13..103: история предыдущих 7 ходов (оставляем нулями при оценке статичного FEN)

    # Вспомогательные плоскости (104..111)
    # 104: Очередь хода (1.0 если ход белых, иначе 0.0)
    if board.turn == chess.WHITE:
        planes[104, :, :] = 1.0

    # 105: Счетчик полуходов для правила 50 ходов
    planes[105, :, :] = float(board.halfmove_clock) / 100.0

    # 106-109: Права на рокировку с точки зрения игрока
    # 106: Наша рокировка на королевский фланг
    if board.has_kingside_castling_rights(our_color):
        planes[106, :, :] = 1.0
    # 107: Наша рокировка на ферзевый фланг
    if board.has_queenside_castling_rights(our_color):
        planes[107, :, :] = 1.0
    # 108: Рокировка соперника на королевский фланг
    if board.has_kingside_castling_rights(their_color):
        planes[108, :, :] = 1.0
    # 109: Рокировка соперника на ферзевый фланг
    if board.has_queenside_castling_rights(their_color):
        planes[109, :, :] = 1.0

    # 110: Поле en passant
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        r = 7 - r if flip else r
        c = c if not flip else 7 - c
        planes[110, r, c] = 1.0

    # 111: Константная плоскость единиц
    planes[111, :, :] = 1.0

    return torch.from_numpy(planes).unsqueeze(0)  # (1, 112, 8, 8)