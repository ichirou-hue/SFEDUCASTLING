import chess

from backend.services.adaptive_logic import select_adaptive_puzzles

KNIGHT_FEN = "k7/8/8/8/8/8/5N2/6K1 w - - 0 1"
ROOK_FEN = "k7/8/8/8/8/8/1R6/6K1 w - - 0 1"


def test_weak_piece_reserves_piece_focus_slots():
    puzzles = [
        {
            "id": f"n{i}",
            "fen": KNIGHT_FEN,
            "moves": "f2g4",
            "themes": ["fork"],
            "rating": 800,
        }
        for i in range(14)
    ] + [
        {
            "id": f"r{i}",
            "fen": ROOK_FEN,
            "moves": "b2b8",
            "themes": ["rookEndgame"],
            "rating": 800,
        }
        for i in range(14)
    ]

    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "tactics"}],
        strong_topics=[{"slug": "pieces"}],
        count=20,
        piece_stats=[{
            "key": "knight",
            "accuracy_percent": 35.0,
            "moves_scored": 12,
            "error_score": 18,
        }],
    )

    focus = [item for item in picked if item.get("adaptive_group") == "piece_focus"]
    assert len(picked) == 20
    assert len(focus) == 3
    assert all(item.get("adaptive_piece") == "knight" for item in focus)
    assert meta["piece_adjustment"]["focus_piece"] == "knight"
    assert meta["piece_adjustment"]["selected"] == 3
