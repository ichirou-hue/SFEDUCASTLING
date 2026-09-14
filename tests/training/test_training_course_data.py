import chess
import pytest

from backend.services.training_course_data import LESSONS_BY_MODULE, MODULES


@pytest.mark.parametrize("module", MODULES, ids=lambda item: item["slug"])
def test_every_module_has_lessons(module):
    lessons = LESSONS_BY_MODULE.get(module["slug"])
    assert module["enabled"] is True
    assert lessons, f"У модуля {module['slug']} нет уроков"
    assert all(lesson["tasks"] for lesson in lessons)


def test_all_course_positions_and_payloads_are_valid():
    for module_slug, lessons in LESSONS_BY_MODULE.items():
        for lesson in lessons:
            for task in lesson["tasks"]:
                board = chess.Board(task["fen"])
                assert board.is_valid(), (
                    f"Некорректная позиция: {module_slug}/{lesson['slug']}/{task['title']}"
                )

                if task.get("source_square"):
                    source = chess.parse_square(task["source_square"])
                    assert board.piece_at(source) is not None, (
                        f"На source_square нет фигуры: {task['title']}"
                    )

                if task["task_type"] == "choose_option":
                    payload = task["payload"]
                    ids = {option["id"] for option in payload["options"]}
                    assert payload["correct_option"] in ids

                if task["task_type"] == "make_move" and task["payload"].get("mode") == "accepted_moves":
                    legal = {move.uci() for move in board.legal_moves}
                    for move in task["payload"]["accepted_moves"]:
                        assert move in legal, f"Эталонный ход {move} нелегален: {task['title']}"
