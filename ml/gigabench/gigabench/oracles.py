"""Rule oracles use python-chess; no model/Stockfish access in gold construction."""
import chess


def mating_moves(board):
    result = []
    for move in list(board.legal_moves):
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            result.append(move.uci())
    return sorted(result)


def mate_within_two(board):
    """All first moves forcing mate in <=2 own moves; exhaustive 3-ply proof.

    Stalemate/draw replies refute a line. No assumption of best engine play.
    """
    roots = []
    for move in list(board.legal_moves):
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            roots.append(move.uci())
            continue
        replies = list(after.legal_moves)
        if not replies or after.is_game_over():
            continue
        forced = True
        for reply in replies:
            position = after.copy(stack=False)
            position.push(reply)
            if position.is_game_over() or not mating_moves(position):
                forced = False
                break
        if forced:
            roots.append(move.uci())
    return sorted(roots)


def piece_at(board, square):
    p = board.piece_at(chess.parse_square(square))
    return p.symbol() if p else '.'


def position_key(board):
    return ' '.join(board.fen().split()[:4])
