from types import SimpleNamespace

import chess

from backend.services.phase_accuracy import (
    DECIDED_WIN_PERCENT,
    MIN_SCORED_MOVES,
    _phase_for_position,
    build_phase_accuracy,
)


START_FEN = chess.STARTING_FEN
ENDGAME_FEN = "8/8/8/8/8/8/4K3/6Rk w - - 0 11"
V2 = {"model": "win_probability"}


def move(
    move_no: int,
    turn: str,
    classification: str | None = "best",
    fen: str = START_FEN,
    ev_before: float | None = 0.0,
    ev_after: float | None = 0.0,
    eval_raw: dict | None = None,
):
    return SimpleNamespace(
        move_no=move_no,
        turn=turn,
        classification=classification,
        fen=fen,
        ev_before=ev_before,
        ev_after=ev_after,
        eval_raw=V2 if eval_raw is None else eval_raw,
    )


def test_phase_boundaries_use_fullmove_and_material():
    assert _phase_for_position(START_FEN, 20) == "opening"
    assert _phase_for_position(START_FEN, 21) == "middlegame"
    assert _phase_for_position(ENDGAME_FEN, 21) == "endgame"


def test_accuracy_uses_win_percent_not_stored_classification():
    game = SimpleNamespace(
        result="1-0",
        white="AI",
        black="player",
        moves=[
            move(1, "w"),  # ход соперника
            # Чёрные теряют больше четырёх пешек: с 50 % до ~17 %.
            move(2, "b", classification="best", ev_before=0.0, ev_after=430.0),
            move(4, "b"),
            move(6, "b"),
        ],
    )

    result = build_phase_accuracy([game])
    opening = result["phases"][0]

    assert result["games_analyzed"] == 1
    assert opening["moves_total"] == 3
    assert opening["moves_evaluated"] == 3
    assert opening["moves_scored"] == 3
    # Хранимый класс best не должен завышать точность: ход пересчитан
    # по оценкам и попал в грубые ошибки.
    assert opening["errors"]["blunder"] == 1
    assert opening["accuracy_percent"] is not None
    assert opening["accuracy_percent"] < 70


def test_decided_positions_do_not_count_towards_accuracy():
    # Белые в +9 пешек: у чёрных вероятность победы ниже предела.
    decided_before = 900.0
    game = SimpleNamespace(
        result="1-0",
        white="AI",
        black="player",
        moves=[
            move(2, "b", ev_before=decided_before, ev_after=decided_before),
            move(4, "b", ev_before=decided_before, ev_after=decided_before),
            move(6, "b", ev_before=0.0, ev_after=0.0),
        ],
    )

    opening = build_phase_accuracy([game])["phases"][0]

    assert opening["moves_evaluated"] == 3
    assert opening["moves_decided"] == 2
    assert opening["moves_scored"] == 1
    # Один ход - слишком мало для процента.
    assert opening["accuracy_percent"] is None
    assert MIN_SCORED_MOVES == 3
    assert DECIDED_WIN_PERCENT == 15.0


def test_moves_without_eval_are_reported_separately():
    game = SimpleNamespace(
        result="1-0",
        white="player",
        black="AI",
        moves=[
            move(1, "w", ev_before=None, ev_after=None),
            # Старая модель оценки: значения несопоставимы, не используем.
            move(
                3,
                "w",
                ev_before=100.0,
                ev_after=-300.0,
                eval_raw={"best_uci": "e2e4"},
            ),
            move(5, "w"),
        ],
    )

    opening = build_phase_accuracy([game])["phases"][0]

    assert opening["moves_total"] == 3
    assert opening["moves_evaluated"] == 1
    assert opening["moves_without_eval"] == 2
    assert opening["moves_scored"] == 1
    assert opening["accuracy_percent"] is None


def test_unfinished_games_are_ignored():
    game = SimpleNamespace(result="*", white="player", black="AI", moves=[move(1, "w")])

    result = build_phase_accuracy([game])

    assert result["games_analyzed"] == 0
    assert result["phases"][0]["moves_total"] == 0
    assert result["phases"][0]["accuracy_percent"] is None
