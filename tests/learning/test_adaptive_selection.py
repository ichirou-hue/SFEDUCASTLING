from backend.services.adaptive_logic import (
    allocation_for_count,
    rank_topic_profile,
    select_adaptive_puzzles,
)
from backend.services.course_topics import primary_course_topic_for_puzzle


def _topic(slug, accuracy, attempts=10):
    return {
        "slug": slug,
        "title": slug,
        "attempts": attempts,
        "correct": 0 if accuracy is None else int(round(attempts * accuracy / 100)),
        "accuracy": accuracy,
        "training": {"attempts": attempts, "correct": 0},
        "puzzles": {"attempts": 0, "correct": 0},
    }


def test_allocation_is_70_30_for_twenty():
    assert allocation_for_count(20) == (14, 6)


def test_rank_uses_bottom_three_and_top_two():
    topics = [
        _topic("pieces", 20),
        _topic("king-safety", 30),
        _topic("endgames", 40),
        _topic("tactics", 60),
        _topic("openings", 70),
    ]
    weak, strong = rank_topic_profile(topics)
    assert [x["slug"] for x in weak] == ["pieces", "king-safety", "endgames"]
    assert [x["slug"] for x in strong] == ["openings", "tactics"]


def test_unseen_topics_are_neither_weak_nor_strong():
    topics = [
        _topic("pieces", 10),
        _topic("king-safety", None, attempts=0),
        _topic("endgames", 30),
        _topic("tactics", 60),
        _topic("openings", 80),
    ]
    weak, strong = rank_topic_profile(topics)
    # В слабые попадают только пройденные темы (низкие 3 из observed).
    assert [x["slug"] for x in weak] == ["pieces", "endgames", "tactics"]
    assert all(x["observed"] is True for x in weak)
    assert [x["slug"] for x in strong] == ["openings"]
    involved = {x["slug"] for x in weak + strong}
    assert "king-safety" not in involved


def test_zero_attempts_has_no_weak_or_strong_topics():
    topics = [
        _topic("pieces", None, attempts=0),
        _topic("king-safety", None, attempts=0),
        _topic("endgames", None, attempts=0),
    ]
    weak, strong = rank_topic_profile(topics)
    assert weak == []
    assert strong == []


def test_unseen_topic_never_enters_strong_topics():
    topics = [
        _topic("pieces", 58.3, attempts=12),
        _topic("king-safety", 0, attempts=1),
        _topic("endgames", 20, attempts=15),
        _topic("tactics", 0, attempts=2),
        _topic("openings", None, attempts=0),
    ]
    weak, strong = rank_topic_profile(topics)
    assert [x["slug"] for x in weak] == ["tactics", "king-safety", "endgames"]
    assert [x["slug"] for x in strong] == ["pieces"]
    assert "openings" not in {x["slug"] for x in strong}


def test_primary_topic_is_deterministic():
    puzzle = {"themes": ["fork", "mateIn2", "middlegame"]}
    assert primary_course_topic_for_puzzle(puzzle) == "king-safety"


def test_selection_prefers_14_weak_and_6_strong_when_pools_allow():
    weak = [
        {"slug": "pieces"},
        {"slug": "tactics"},
    ]
    strong = [
        {"slug": "king-safety"},
        {"slug": "endgames"},
    ]

    puzzles = []
    for i in range(10):
        puzzles.append({"id": f"p{i}", "themes": ["advancedPawn"]})
        puzzles.append({"id": f"n{i}", "themes": ["fork"]})
        puzzles.append({"id": f"b{i}", "themes": ["pin"]})
        puzzles.append({"id": f"q{i}", "themes": ["sacrifice"]})
        puzzles.append({"id": f"k{i}", "themes": ["exposedKing"]})
        puzzles.append({"id": f"s{i}", "themes": ["promotion"]})

    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=weak,
        strong_topics=strong,
        count=20,
    )

    assert len(picked) == 20
    assert meta["weak_selected"] == 14
    assert meta["strong_selected"] == 6
    assert meta["fallback_selected"] == 0
    assert sum(p["adaptive_group"] == "weak" for p in picked) == 14
    assert sum(p["adaptive_group"] == "strong" for p in picked) == 6


def test_adaptive_selection_uses_same_primary_topic_as_statistics():
    # Оба многотемных пазла подходят под несколько модулей при широком
    # сопоставлении, но B1 обязан использовать ту же primary-тему, что и
    # последующая агрегация puzzle_attempts.
    puzzles = [
        {
            "id": "back-rank",
            "themes": ["backRankMate", "mateIn2", "endgame"],
        },
        {
            "id": "pure-tactic",
            "themes": ["fork", "middlegame"],
        },
        {
            "id": "promotion-endgame",
            "themes": ["advancedPawn", "promotion", "endgame"],
        },
        {
            "id": "pure-pawn",
            "themes": ["advancedPawn", "endgame"],
        },
    ]

    # При count=2 квота weak=1, strong=1.
    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "tactics"}],
        strong_topics=[{"slug": "king-safety"}],
        count=2,
    )

    assert meta["weak_selected"] == 1
    assert meta["strong_selected"] == 1
    assert meta["fallback_selected"] == 0

    by_group = {item["adaptive_group"]: item for item in picked}
    assert by_group["weak"]["id"] == "pure-tactic"
    assert by_group["weak"]["adaptive_topic"] == "tactics"
    assert by_group["strong"]["id"] == "back-rank"
    assert by_group["strong"]["adaptive_topic"] == "king-safety"

    # Пазл с backRankMate+mateIn2 статистически относится к матам,
    # а advancedPawn+promotion — к эндшпилю. Они не должны
    # маскироваться под tactics/king-safety только из-за дополнительного theme.
    assert primary_course_topic_for_puzzle(puzzles[0]) == "king-safety"
    assert primary_course_topic_for_puzzle(puzzles[2]) == "endgames"


def test_every_adaptive_non_fallback_topic_matches_primary_topic():
    weak = [{"slug": "tactics"}, {"slug": "pieces"}]
    strong = [
        {"slug": "king-safety"},
        {"slug": "endgames"},
    ]
    puzzles = []
    for i in range(8):
        puzzles.extend(
            [
                {"id": f"n{i}", "themes": ["fork"]},
                {"id": f"r{i}", "themes": ["rookEndgame"]},
                {"id": f"q{i}", "themes": ["sacrifice"]},
                {"id": f"p{i}", "themes": ["advancedPawn"]},
                {"id": f"s{i}", "themes": ["promotion"]},
                {"id": f"m{i}", "themes": ["mateIn2"]},
            ]
        )

    # Добавляем неоднозначные пазлы, которые раньше могли попасть не в ту тему.
    puzzles.extend(
        [
            {"id": "amb-rook-mate", "themes": ["backRankMate", "mateIn2"]},
            {"id": "amb-pawn-promo", "themes": ["advancedPawn", "promotion"]},
            {"id": "amb-queen-bishop", "themes": ["attraction", "skewer"]},
        ]
    )

    picked, _meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=weak,
        strong_topics=strong,
        count=20,
    )

    for item in picked:
        if item["adaptive_group"] == "fallback":
            continue
        assert item["adaptive_topic"] == primary_course_topic_for_puzzle(item)


def test_easy_puzzles_from_other_topics_precede_hard_weak_topic_puzzles():
    puzzles = [
        {"id": "hard-knight", "themes": ["fork"], "rating": 2200},
        {"id": "easy-pawn", "themes": ["advancedPawn"], "rating": 700},
        {"id": "easy-rook", "themes": ["rookEndgame"], "rating": 900},
    ]
    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=3,
        topic_difficulties={"knight": 1, "pawn": 1, "rook": 1},
    )
    assert [p["id"] for p in picked[:2]] in (["easy-pawn", "easy-rook"], ["easy-rook", "easy-pawn"])
    assert picked[2]["id"] == "hard-knight"
    assert meta["difficulty_matched"] == 2


def test_weak_game_phase_gets_more_phase_tagged_tasks():
    puzzles = [
        {"id": f"{phase}-{i}", "themes": ["fork", phase], "rating": 800}
        for phase in ("opening", "middlegame", "endgame")
        for i in range(12)
    ]
    phases = [
        {"key": "opening", "accuracy_percent": 95, "moves_evaluated": 10},
        {"key": "middlegame", "accuracy_percent": 75, "moves_evaluated": 15},
        {"key": "endgame", "accuracy_percent": 40, "moves_evaluated": 20},
    ]

    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=20,
        phase_stats=phases,
    )

    phase_picks = [p for p in picked if p["adaptive_group"] == "phase_focus"]
    by_phase = {phase: sum(p["adaptive_phase"] == phase for p in phase_picks)
                for phase in ("opening", "middlegame", "endgame")}
    assert len(picked) == 20
    assert len({p["id"] for p in picked}) == 20
    assert len(phase_picks) == 4
    assert by_phase["endgame"] > by_phase["middlegame"] > by_phase["opening"]
    assert meta["phase_adjustment"]["focus_phase"] == "endgame"
    assert meta["phase_adjustment"]["selected"] == 4


def test_phase_bias_waits_for_enough_evaluated_moves():
    puzzles = [
        {"id": f"p{i}", "themes": ["fork", "endgame"], "rating": 800}
        for i in range(10)
    ]

    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=5,
        phase_stats=[{"key": "endgame", "accuracy_percent": 0, "moves_evaluated": 4}],
    )

    assert len(picked) == 5
    assert all(p["adaptive_group"] != "phase_focus" for p in picked)
    assert meta["phase_adjustment"]["reason"] == "insufficient_history"


def test_sparse_phase_pool_falls_back_without_duplicates_or_missing_tasks():
    puzzles = [
        {"id": "only-endgame", "themes": ["fork", "endgame"], "rating": 800},
        *[
            {"id": f"middle-{i}", "themes": ["fork", "middlegame"], "rating": 800}
            for i in range(25)
        ],
    ]

    picked, meta = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=20,
        phase_stats=[{"key": "endgame", "accuracy_percent": 20, "moves_evaluated": 8}],
    )

    assert len(picked) == 20
    assert len({p["id"] for p in picked}) == 20
    assert sum(p["adaptive_group"] == "phase_focus" for p in picked) == 1
    assert meta["phase_adjustment"]["selected"] == 1
