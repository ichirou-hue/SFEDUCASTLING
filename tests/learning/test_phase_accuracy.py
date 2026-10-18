from types import SimpleNamespace

import chess

from backend.services.phase_accuracy import _phase_for_position, build_phase_accuracy


START_FEN = chess.STARTING_FEN
ENDGAME_FEN = "8/8/8/8/8/8/4K3/6Rk w - - 0 11"


def move(move_no: int, turn: str, classification: str | None, fen: str = START_FEN):
    return SimpleNamespace(
        move_no=move_no,
        turn=turn,
        classification=classification,
        fen=fen,
    )


def test_phase_boundaries_use_fullmove_and_material():
    assert _phase_for_position(START_FEN, 20) == "opening"
    assert _phase_for_position(START_FEN, 21) == "middlegame"
    assert _phase_for_position(ENDGAME_FEN, 21) == "endgame"


def test_accuracy_counts_only_player_moves_and_evaluated_classifications():
    game = SimpleNamespace(
        result="1-0",
        white="AI",
        black="player",
        moves=[
            move(1, "w", "blunder"),  # opponent move
            move(2, "b", "best"),
            move(4, "b", "mistake"),
            move(6, "b", None),
        ],
    )

    result = build_phase_accuracy([game])
    opening = result["phases"][0]

    assert result["games_analyzed"] == 1
    assert opening["accuracy_percent"] == 50.0
    assert opening["moves_total"] == 3
    assert opening["moves_evaluated"] == 2
    assert opening["moves_without_eval"] == 1
    assert opening["errors"] == {"inaccuracy": 0, "mistake": 1, "blunder": 0}
    assert result["phases"][1]["accuracy_percent"] is None


def test_unfinished_games_are_ignored():
    game = SimpleNamespace(result="*", white="player", black="AI", moves=[move(1, "w", "best")])

    result = build_phase_accuracy([game])

    assert result["games_analyzed"] == 0
    assert result["phases"][0]["moves_total"] == 0
    assert result["phases"][0]["accuracy_percent"] is None
