import json
from pathlib import Path

from backend.api_gateway.routes.learning import (
    COURSE_TOPIC_PUZZLE_THEMES,
    _puzzle_matches_course_topic,
)
from backend.services.course_topics import primary_course_topic_for_puzzle
from backend.services.training_course_data import MODULES


PUZZLE_FILE = Path(__file__).resolve().parents[2] / "backend" / "knowledge" / "puzzles.json"


def _puzzle_base() -> list[dict]:
    """База пазлов из состояния приложения или прямо из файла.

    В юнит-тестах state.puzzle_base обычно пуст, поэтому читаем JSON сами:
    проверки должны идти по реальному набору задач, а не скипаться.
    """
    from backend.api_gateway import state

    if state.puzzle_base:
        return state.puzzle_base.get("puzzles") or []
    with PUZZLE_FILE.open(encoding="utf-8") as f:
        return json.load(f).get("puzzles") or []


def test_course_topics_cover_all_training_modules():
    assert set(COURSE_TOPIC_PUZZLE_THEMES) == {
        "pieces",
        "king-safety",
        "endgames",
        "tactics",
        "openings",
        "middlegame",
    }
    # Любая карточка курса должна вести на свою тему, иначе кнопка
    # «Проверить на пазлах этой темы» подменит набор случайными задачами.
    assert {module["slug"] for module in MODULES} <= set(COURSE_TOPIC_PUZZLE_THEMES)


def test_every_course_topic_has_matching_puzzles():
    """Для каждого модуля есть пазлы, и все они несут темы этого модуля."""
    puzzles = _puzzle_base()
    assert puzzles, "база пазлов не загружена"

    for slug, themes in COURSE_TOPIC_PUZZLE_THEMES.items():
        matched = [p for p in puzzles if _puzzle_matches_course_topic(p, slug)]
        assert matched, f"нет пазлов для темы {slug}"
        stray = [
            p.get("id")
            for p in matched
            if not (set(p.get("themes") or []) & themes)
        ]
        assert not stray, f"тема {slug}: чужие темы у {stray[:5]}"


def test_puzzle_matches_related_course_topic():
    assert _puzzle_matches_course_topic(
        {"themes": ["crushing", "advancedPawn", "endgame"]},
        "pieces",
    )
    # Мат в один ход: тема mate* и решение действительно заканчивается матом.
    back_rank_mate = {
        "themes": ["mate", "mateIn1", "backRankMate"],
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "moves": "a1a8",
    }
    assert _puzzle_matches_course_topic(back_rank_mate, "king-safety")
    assert _puzzle_matches_course_topic(
        {"themes": ["promotion", "endgame"]},
        "endgames",
    )


def test_king_safety_requires_actual_mate():
    """Метки mate* без мата в решении в «Мат - как цель игры» не попадают."""
    non_mate = {
        "themes": ["mate", "mateIn1", "kingsideAttack"],
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "moves": "a1b1",
    }
    assert not _puzzle_matches_course_topic(non_mate, "king-safety")


def test_attack_themes_do_not_pull_puzzles_into_king_safety():
    """Атака на короля и открытый король сами по себе - не тема мата."""
    attack = {"themes": ["kingsideAttack", "exposedKing", "middlegame"]}
    assert not _puzzle_matches_course_topic(attack, "king-safety")
    assert _puzzle_matches_course_topic(attack, "middlegame")


def test_puzzle_does_not_match_unrelated_topic():
    puzzle = {"themes": ["bishopEndgame", "endgame"]}
    assert _puzzle_matches_course_topic(puzzle, "pieces")
    assert not _puzzle_matches_course_topic(puzzle, "tactics")
    assert not _puzzle_matches_course_topic(puzzle, "unknown")


def test_tactics_match_without_piece_check():
    # Фильтр «профильная фигура ходит в решении» упразднён вместе
    # со слиянием модулей: тактика относится к tactics, кем бы ни ходили.
    no_knight = {"themes": ["fork"], "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"}
    assert _puzzle_matches_course_topic(no_knight, "tactics")

    with_bishop = {"themes": ["pin"], "fen": "4k3/8/8/8/8/8/8/3BK3 w - - 0 1"}
    assert _puzzle_matches_course_topic(with_bishop, "tactics")


def test_primary_topic_needs_no_piece():
    knight_moves = {
        "themes": ["fork"],
        "fen": "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
        "moves": "b1c3",
    }
    king_moves = {
        "themes": ["fork"],
        "fen": "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
        "moves": "e1e2",
    }
    assert primary_course_topic_for_puzzle(knight_moves) == "tactics"
    assert primary_course_topic_for_puzzle(king_moves) == "tactics"


def test_openings_topic_matches_opening_puzzles():
    assert _puzzle_matches_course_topic({"themes": ["opening", "trap"]}, "openings")
    assert primary_course_topic_for_puzzle({"themes": ["opening"]}) == "openings"
    assert not _puzzle_matches_course_topic({"themes": ["fork"]}, "openings")


def test_known_attack_puzzle_not_in_king_safety():
    """Реальный пазл с kingsideAttack без мата не должен попадать в тему мата."""
    puzzles = _puzzle_base()
    assert puzzles, "база пазлов не загружена"

    by_id = {str(p.get("id")): p for p in puzzles}
    # 27qUy - размен на e8 (kingsideAttack), мата в решении нет.
    puzzle = by_id["27qUy"]

    assert "kingsideAttack" in (puzzle.get("themes") or [])
    assert not _puzzle_matches_course_topic(puzzle, "king-safety")


def test_every_king_safety_puzzle_really_ends_in_mate():
    """В теме «Мат - как цель игры» не должно быть задач без мата."""
    from backend.services.course_topics import _solution_ends_in_mate

    puzzles = _puzzle_base()
    assert puzzles, "база пазлов не загружена"

    bad = [
        p.get("id")
        for p in puzzles
        if _puzzle_matches_course_topic(p, "king-safety")
        and not _solution_ends_in_mate(p)
    ]
    assert not bad, f"без мата в решении: {bad[:5]}"
