from types import SimpleNamespace

from backend.services.training_checker import check_training_task


def make_task(**overrides):
    data = {
        "id": 1,
        "task_type": "select_squares",
        "fen": "7k/8/8/8/4N3/8/8/K7 w - - 0 1",
        "source_square": "e4",
        "payload": {"mode": "all_legal_moves"},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_knight_center_all_squares():
    task = make_task()
    result = check_training_task(
        task,
        {
            "selected_squares": [
                "c3",
                "c5",
                "d2",
                "d6",
                "f2",
                "f6",
                "g3",
                "g5",
            ]
        },
    )
    assert result.correct is True
    assert result.score == 1.0


def test_knight_center_partial_answer():
    task = make_task()
    result = check_training_task(task, {"selected_squares": ["c3", "c5"]})
    assert result.correct is False
    assert 0 < result.score < 1


def test_knight_capture_mode():
    task = make_task(
        fen="7k/8/8/2r3b1/4N3/8/8/K7 w - - 0 1",
        payload={"mode": "capture_squares"},
    )
    result = check_training_task(task, {"selected_squares": ["c5", "g5"]})
    assert result.correct is True


def test_make_move_any_legal():
    task = make_task(
        task_type="make_move",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        source_square="g1",
        payload={"mode": "any_legal_move"},
    )
    result = check_training_task(task, {"move": "g1f3"})
    assert result.correct is True


def test_make_move_accepted_moves_rejects_other_legal_move():
    task = make_task(
        task_type="make_move",
        fen="7k/8/8/8/8/8/8/R6K w - - 0 1",
        source_square="a1",
        payload={"mode": "accepted_moves", "accepted_moves": ["a1a8"]},
    )

    wrong = check_training_task(task, {"move": "a1a2"})
    right = check_training_task(task, {"move": "a1a8"})

    assert wrong.correct is False
    assert right.correct is True


def test_choose_option():
    task = make_task(
        task_type="choose_option",
        source_square=None,
        payload={
            "options": [
                {"id": "mate", "label": "Мат"},
                {"id": "stalemate", "label": "Пат"},
            ],
            "correct_option": "mate",
        },
    )

    wrong = check_training_task(task, {"option": "stalemate"})
    right = check_training_task(task, {"option": "mate"})

    assert wrong.correct is False
    assert wrong.score == 0.0
    assert right.correct is True
    assert right.score == 1.0
