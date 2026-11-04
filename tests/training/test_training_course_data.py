import chess
import pytest

from backend.services.training_course_data import (
    LEVEL_LABELS,
    LEVEL_MIN_ELO,
    LESSONS_BY_MODULE,
    MODULES,
    locked_module_slugs,
    module_required_elo,
    module_unlocked_for_elo,
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


def test_module_required_elo_matches_level_label():
    """Порог доступа - нижняя граница метки уровня модуля."""
    assert LEVEL_MIN_ELO == {
        "pieces": 0,
        "king-safety": 0,
        "tactics": 0,
        "endgames": 1000,
        "openings": 1000,
        "middlegame": 1000,
    }
    for slug, label in LEVEL_LABELS.items():
        expected = int(label.split("-")[0])
        assert module_required_elo(slug) == expected, slug


def test_module_access_by_elo():
    # Базовые модули доступны всем, включая пользователя без рейтинга.
    assert module_unlocked_for_elo("pieces", None)
    assert module_unlocked_for_elo("tactics", 0)

    # Модули 1000-1500 открываются ровно с нижней границы.
    assert not module_unlocked_for_elo("endgames", None)
    assert not module_unlocked_for_elo("openings", 999)
    assert module_unlocked_for_elo("openings", 1000)
    assert module_unlocked_for_elo("middlegame", 1500)


def test_locked_module_slugs_match_access_rule():
    """Набор закрытых слагов совпадает с построчной проверкой доступа."""
    for elo in (None, 0, 800, 999, 1000, 1500):
        expected = {
            slug
            for slug in LEVEL_MIN_ELO
            if not module_unlocked_for_elo(slug, elo)
        }
        assert locked_module_slugs(elo) == expected, elo

    assert locked_module_slugs(None) == {"endgames", "openings", "middlegame"}
    assert locked_module_slugs(999) == {"endgames", "openings", "middlegame"}
    assert locked_module_slugs(1000) == set()


@pytest.mark.parametrize("module", MODULES, ids=lambda item: item["slug"])
def test_every_module_has_lessons(module):
    lessons = LESSONS_BY_MODULE.get(module["slug"])
    assert lessons, f"У модуля {module['slug']} нет уроков"
    assert all(lesson["tasks"] for lesson in lessons)


def test_course_has_program_topics_with_four_to_six_tasks():
    topics = _topics()
    assert len(topics) == PROGRAM_TOPICS
    for lesson in topics:
        task_count = len(lesson["tasks"])
        assert 4 <= task_count <= 6, (
            f"{lesson['slug']}: {task_count} заданий, ожидалось 4-6"
        )


def test_lesson_slugs_are_unique():
    slugs = [
        lesson["slug"]
        for lessons in LESSONS_BY_MODULE.values()
        for lesson in lessons
    ]
    assert len(slugs) == len(set(slugs))


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
    assert all(len(lesson["tasks"]) == 5 for lesson in OPENINGS_THEORY)


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
