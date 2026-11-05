import chess

from backend.analysis.brilliant import brilliant_sacrifice


def test_best_material_sacrifice_is_brilliant():
    board = chess.Board("6k1/7p/8/8/8/3B4/8/6K1 w - - 0 1")
    move = chess.Move.from_uci("d3h7")
    assert move in board.legal_moves

    result = brilliant_sacrifice(
        board,
        move,
        best_uci="d3h7",
        classification="best",
    )
    assert result["is_brilliant"] is True
    assert result["sacrifice_value"] == 2


def test_good_move_is_not_brilliant_when_not_stockfish_best():
    board = chess.Board("6k1/7p/8/8/8/3B4/8/6K1 w - - 0 1")
    move = chess.Move.from_uci("d3h7")
    result = brilliant_sacrifice(
        board,
        move,
        best_uci="d3e4",
        classification="best",
    )
    assert result["is_brilliant"] is False
