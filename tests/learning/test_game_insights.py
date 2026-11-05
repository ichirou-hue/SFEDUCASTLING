from types import SimpleNamespace

import chess

from backend.services.game_insights import build_game_insights

V2 = {"model": "win_probability"}
START = chess.STARTING_FEN


def move(*, uci: str, turn: str, before: float = 0.0, after: float = 0.0, brilliant: bool = False):
    return SimpleNamespace(
        fen=START,
        uci=uci,
        turn=turn,
        ev_before=before,
        ev_after=after,
        eval_raw={**V2, "brilliant": brilliant},
        move_no=1,
        classification=None,
    )


def test_piece_profile_uses_only_player_moves_and_requires_five_samples():
    game = SimpleNamespace(
        result="1-0",
        engine="maia3",
        white="player",
        black="AI",
        moves=[
            *[move(uci="g1f3", turn="w") for _ in range(5)],
            *[move(uci="g8f6", turn="b", before=0, after=400) for _ in range(5)],
        ],
    )
    result = build_game_insights([game])
    knight = next(item for item in result["pieces"] if item["key"] == "knight")
    assert knight["moves_total"] == 5
    assert knight["moves_scored"] == 5
    assert knight["accuracy_percent"] == 100.0


def test_weak_piece_and_brilliant_counter_are_exposed():
    game = SimpleNamespace(
        result="1-0",
        engine="maia3",
        white="player",
        black="AI",
        moves=[
            *[move(uci="c1g5", turn="w", before=0, after=-400) for _ in range(5)],
            *[move(uci="g1f3", turn="w", brilliant=(i == 0)) for i in range(5)],
        ],
    )
    result = build_game_insights([game])
    assert result["weak_pieces"][0]["key"] == "bishop"
    assert result["quality"]["brilliant_moves"] == 1
