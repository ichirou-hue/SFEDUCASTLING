import re

import chess
import pytest

from backend.services.training_course_data import (
    LESSONS_BY_MODULE,
    LEVEL_LABELS,
    MODULES,
)

PROGRAM_MODULES = [
    ("pieces", "Шахматная азбука"),
    ("king-safety", "Мат - как цель игры"),
    ("tactics", "Тактические приёмы"),
    ("endgames", "Эндшпиль"),
    ("openings", "Дебют"),
    ("middlegame", "Миттельшпиль: стратегия"),
]

# 39 тем: в исходном файле у двух тем модуля тактики повторён номер «15»,
# поэтому строк на одну больше, чем последний номер темы.
PROGRAM_TOPICS = 39

# Каждая тема программы наполнена 12-15 заданиями: база урока плюс
# позиции из реальных партий (course_extra_tasks).
MIN_TOPIC_TASKS = 12
MAX_TOPIC_TASKS = 15


def _topics():
    """Все темы программы, кроме конкретных дебютов."""
    return [
        lesson
        for module in MODULES
        for lesson in LESSONS_BY_MODULE.get(module["slug"], [])
        if lesson.get("section_title") != "Конкретные дебюты"
    ]


def test_modules_follow_program_order():
    assert [(module["slug"], module["title"]) for module in MODULES] == PROGRAM_MODULES
    assert all(module["enabled"] for module in MODULES)
    assert sorted(module["sort_order"] for module in MODULES) == [1, 2, 3, 4, 5, 6]


def test_every_module_has_level_label():
    for module in MODULES:
        assert LEVEL_LABELS.get(module["slug"]), module["slug"]
    for module in MODULES[:3]:
        assert LEVEL_LABELS[module["slug"]] == "0-1000"
    for module in MODULES[3:]:
        assert LEVEL_LABELS[module["slug"]] == "1000-1500"


@pytest.mark.parametrize("module", MODULES, ids=lambda item: item["slug"])
def test_every_module_has_lessons(module):
    lessons = LESSONS_BY_MODULE.get(module["slug"])
    assert lessons, f"У модуля {module['slug']} нет уроков"
    assert all(lesson["tasks"] for lesson in lessons)


def test_course_has_program_topics_with_expected_task_count():
    topics = _topics()
    assert len(topics) == PROGRAM_TOPICS
    for lesson in topics:
        task_count = len(lesson["tasks"])
        assert MIN_TOPIC_TASKS <= task_count <= MAX_TOPIC_TASKS, (
            f"{lesson['slug']}: {task_count} заданий, "
            f"ожидалось {MIN_TOPIC_TASKS}-{MAX_TOPIC_TASKS}"
        )


def test_lesson_slugs_are_unique():
    slugs = [
        lesson["slug"]
        for lessons in LESSONS_BY_MODULE.values()
        for lesson in lessons
    ]
    assert len(slugs) == len(set(slugs))


def _task_signature(task):
    payload = task["payload"]
    if task["task_type"] == "choose_option":
        answer = (
            payload.get("correct_option"),
            tuple(sorted(option["id"] for option in payload["options"])),
        )
    else:
        answer = (
            payload.get("mode"),
            tuple(sorted(payload.get("accepted_moves") or ())),
        )
    return (task["fen"], task["source_square"], task["task_type"], answer)


def test_no_duplicate_tasks_inside_lesson():
    for module_slug, lessons in LESSONS_BY_MODULE.items():
        for lesson in lessons:
            signatures = [_task_signature(task) for task in lesson["tasks"]]
            assert len(signatures) == len(set(signatures)), (
                f"повторы заданий в {module_slug}/{lesson['slug']}"
            )


def test_all_course_positions_and_payloads_are_valid():
    for module_slug, lessons in LESSONS_BY_MODULE.items():
        for lesson in lessons:
            for task in lesson["tasks"]:
                board = chess.Board(task["fen"])
                assert board.is_valid(), (
                    f"Некорректная позиция: {module_slug}/{lesson['slug']}/{task['title']}"
                )
                assert 1 <= task["difficulty"] <= 3
                assert task["sort_order"] > 0

                payload = task["payload"]

                if task["task_type"] == "choose_option":
                    ids = {option["id"] for option in payload["options"]}
                    assert len(ids) >= 2, task["title"]
                    assert payload["correct_option"] in ids, task["title"]
                    assert all(option["label"] for option in payload["options"])
                    continue

                assert task["task_type"] in {"select_squares", "make_move"}
                assert task["source_square"], task["title"]
                source = chess.parse_square(task["source_square"])
                piece = board.piece_at(source)
                assert piece is not None, (
                    f"На source_square нет фигуры: {task['title']}"
                )
                assert piece.color == board.turn, (
                    f"Ходит другая сторона: {task['title']}"
                )
                assert payload.get("piece") == piece.symbol().upper(), task["title"]

                legal = [m for m in board.legal_moves if m.from_square == source]
                mode = payload.get("mode", "all_legal_moves")
                if mode in {"capture_squares", "legal_capture"}:
                    legal = [m for m in legal if board.is_capture(m)]

                if task["task_type"] == "make_move" and mode == "accepted_moves":
                    accepted = set(payload["accepted_moves"])
                    assert accepted, task["title"]
                    for move in accepted:
                        assert move in {m.uci() for m in board.legal_moves}, (
                            f"Эталонный ход {move} нелегален: {task['title']}"
                        )
                        assert chess.Move.from_uci(move).from_square == source
                    legal = [m for m in legal if m.uci() in accepted]

                assert legal, f"Нет ожидаемых ходов: {module_slug}/{lesson['slug']}/{task['title']}"


# --- Модуль «Дебют»: пять тем о принципах и 30 конкретных дебютов ---

OPENINGS_LESSONS = LESSONS_BY_MODULE["openings"]
OPENINGS_THEORY = [
    lesson
    for lesson in OPENINGS_LESSONS
    if lesson.get("section_title") != "Конкретные дебюты"
]
CONCRETE_OPENINGS = [
    lesson
    for lesson in OPENINGS_LESSONS
    if lesson.get("section_title") == "Конкретные дебюты"
]


def test_openings_module_starts_with_five_principles():
    assert len(OPENINGS_THEORY) == 5
    assert [lesson["sort_order"] for lesson in OPENINGS_THEORY] == [1, 2, 3, 4, 5]
    assert OPENINGS_THEORY[0]["title"] == "Три главных дебютных принципа"
    assert OPENINGS_THEORY[-1]["title"] == "Критические ошибки в дебюте"
    # Пять принципов сохраняют базовые задания, дополнительные идут сверху.
    assert all(len(lesson["tasks"]) >= 5 for lesson in OPENINGS_THEORY)


def test_concrete_openings_follow_principles():
    assert len(CONCRETE_OPENINGS) == 30
    assert min(lesson["sort_order"] for lesson in CONCRETE_OPENINGS) == 6
    assert max(lesson["sort_order"] for lesson in CONCRETE_OPENINGS) == 35

    theory_orders = [lesson["sort_order"] for lesson in OPENINGS_THEORY]
    assert max(theory_orders) < min(
        lesson["sort_order"] for lesson in CONCRETE_OPENINGS
    )

    slugs = [lesson["slug"] for lesson in CONCRETE_OPENINGS]
    assert len(slugs) == len(set(slugs))
    titles = [lesson["title"] for lesson in CONCRETE_OPENINGS]
    assert len(titles) == len(set(titles))


@pytest.mark.parametrize(
    "lesson", CONCRETE_OPENINGS, ids=lambda item: item["slug"]
)
def test_opening_lesson_has_varied_branches(lesson):
    variations = lesson.get("variations") or []
    assert 12 <= len(lesson["tasks"]) <= 15
    assert len(variations) >= 2, "у урока дебюта должно быть минимум две ветки"
    assert len({variation["id"] for variation in variations}) == len(variations)

    branch_ids = {variation["id"] for variation in variations}
    branch_tasks = {
        task["payload"]["branch"]
        for task in lesson["tasks"]
        if "branch" in task["payload"]
    }
    assert branch_tasks == branch_ids

    for variation in variations:
        assert variation["label"]
        assert variation["fen"]
        assert variation["intro_move"] and variation["intro_san"]
        assert variation["line"]

        board = chess.Board(variation["fen"])
        assert board.is_valid()
        intro = chess.Move.from_uci(variation["intro_move"])
        assert intro in board.legal_moves, (
            f"{lesson['slug']}/{variation['id']}: вступительный ход нелегален"
        )


@pytest.mark.parametrize(
    "lesson", CONCRETE_OPENINGS, ids=lambda item: item["slug"]
)
def test_opening_branch_tasks_follow_intro(lesson):
    common = [
        task for task in lesson["tasks"] if "branch" not in task["payload"]
    ]
    assert common, "у урока дебюта должна быть общая линия"

    board = chess.Board(common[-1]["fen"])
    board.push(chess.Move.from_uci(common[-1]["payload"]["accepted_moves"][0]))

    for variation in lesson["variations"]:
        assert variation["fen"] == board.fen(), (
            f"{lesson['slug']}: развилка стартует не после общей линии"
        )

        branch_tasks = [
            task
            for task in lesson["tasks"]
            if task["payload"].get("branch") == variation["id"]
        ]
        assert branch_tasks

        # Первое задание ветки - позиция сразу после хода соперника.
        intro_board = chess.Board(variation["fen"])
        intro_board.push(chess.Move.from_uci(variation["intro_move"]))
        assert branch_tasks[0]["fen"] == intro_board.fen(), (
            f"{lesson['slug']}/{variation['id']}: ветка начинается не с intro-хода"
        )

        # Задания ветки сгруппированы и не пересекаются с общей линией.
        assert all(task["sort_order"] >= 100 for task in branch_tasks)


# --- Матовые задачи: заголовок не должен обещать неверный мат ---

MATE_IN_TITLE = re.compile(r"мат в (один|\d+) (?:ход|хода|ходов)", re.IGNORECASE)


def _push(board, move):
    probe = board.copy()
    probe.push(move)
    return probe


def _has_forced_mate(board, moves):
    """Форсированный мат белых не длиннее `moves` своих ходов."""
    if moves == 0:
        return False
    for move in board.legal_moves:
        after = _push(board, move)
        if after.is_checkmate():
            return True
        if moves == 1 or after.is_game_over():
            continue
        if all(
            _has_forced_mate(_push(after, reply), moves - 1)
            for reply in after.legal_moves
        ):
            return True
    return False


def test_mate_titles_match_the_position():
    checked = 0
    for module_slug, lessons in LESSONS_BY_MODULE.items():
        for lesson in lessons:
            for task in lesson["tasks"]:
                if task["task_type"] != "make_move":
                    continue
                match = MATE_IN_TITLE.search(task["title"])
                if match is None:
                    continue
                checked += 1
                where = f"{module_slug}/{lesson['slug']}/{task['sort_order']}"
                board = chess.Board(task["fen"])
                distance = 1 if match.group(1).lower() == "один" else int(match.group(1))
                for uci in task["payload"]["accepted_moves"]:
                    after = _push(board, chess.Move.from_uci(uci))
                    if distance == 1:
                        assert after.is_checkmate(), (
                            f"{where}: заголовок «{task['title']}», "
                            f"а ход {uci} не мат"
                        )
                        continue
                    assert list(after.legal_moves), (
                        f"{where}: после {uci} пат, а не матовая атака"
                    )
                    # Первый ход матовой атаки: на любой ответ чёрных у белых
                    # остаётся форсированный мат за distance - 1 ход.
                    assert all(
                        _has_forced_mate(_push(after, reply), distance - 1)
                        for reply in after.legal_moves
                    ), f"{where}: после {uci} нет мата в {distance} хода"
                    # Титул не завышает длину: более короткого мата быть не должно.
                    assert not _has_forced_mate(board, distance - 1), (
                        f"{where}: мат короче, чем «{task['title']}»"
                    )
    assert checked >= 10
