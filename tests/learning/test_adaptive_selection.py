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


def test_rank_uses_bottom_three_and_top_three():
    topics = [
        _topic("pawn", 20),
        _topic("knight", 30),
        _topic("bishop", 40),
        _topic("rook", 60),
        _topic("queen", 70),
        _topic("king", 80),
        _topic("special-rules", 90),
        _topic("check-mate-stalemate", 100),
    ]
    weak, strong = rank_topic_profile(topics)
    assert [x["slug"] for x in weak] == ["pawn", "knight", "bishop"]
    assert [x["slug"] for x in strong] == [
        "check-mate-stalemate",
        "special-rules",
        "king",
    ]


def test_unseen_topic_has_neutral_rank_but_is_not_strong():
    topics = [
        _topic("pawn", 10),
        _topic("knight", None, attempts=0),
        _topic("bishop", 90),
        _topic("rook", None, attempts=0),
    ]
    weak, strong = rank_topic_profile(topics)
    assert weak[0]["slug"] == "pawn"
    unseen = next(x for x in weak if x["slug"] == "knight")
    assert unseen["rank_accuracy"] == 50.0
    assert unseen["observed"] is False
    assert [x["slug"] for x in strong] == ["bishop"]
    assert all(x["observed"] is True for x in strong)


def test_unseen_topic_never_enters_strong_topics():
    topics = [
        _topic("pawn", 58.3, attempts=12),
        _topic("knight", 0, attempts=1),
        _topic("bishop", 20, attempts=15),
        _topic("rook", 0, attempts=2),
        _topic("queen", 0, attempts=6),
        _topic("king", None, attempts=0),
        _topic("special-rules", 50, attempts=2),
        _topic("check-mate-stalemate", 30, attempts=10),
    ]
    weak, strong = rank_topic_profile(topics)
    assert [x["slug"] for x in weak] == ["queen", "rook", "knight"]
    assert [x["slug"] for x in strong] == [
        "pawn",
        "special-rules",
        "check-mate-stalemate",
    ]
    assert "king" not in {x["slug"] for x in strong}


def test_primary_topic_is_deterministic():
    puzzle = {"themes": ["fork", "mateIn2", "middlegame"]}
    assert primary_course_topic_for_puzzle(puzzle) == "check-mate-stalemate"


def test_selection_prefers_14_weak_and_6_strong_when_pools_allow():
    weak = [
        {"slug": "pawn"},
        {"slug": "knight"},
        {"slug": "bishop"},
    ]
    strong = [
        {"slug": "queen"},
        {"slug": "king"},
        {"slug": "special-rules"},
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
            "id": "pure-rook",
            "themes": ["rookEndgame", "endgame"],
        },
        {
            "id": "promotion-pawn",
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
        weak_topics=[{"slug": "rook"}],
        strong_topics=[{"slug": "pawn"}],
        count=2,
    )

    assert meta["weak_selected"] == 1
    assert meta["strong_selected"] == 1
    assert meta["fallback_selected"] == 0

    by_group = {item["adaptive_group"]: item for item in picked}
    assert by_group["weak"]["id"] == "pure-rook"
    assert by_group["weak"]["adaptive_topic"] == "rook"
    assert by_group["strong"]["id"] == "pure-pawn"
    assert by_group["strong"]["adaptive_topic"] == "pawn"

    # Пазл с backRankMate+mateIn2 статистически относится к матам,
    # а advancedPawn+promotion — к специальным правилам. Они не должны
    # маскироваться под rook/pawn только из-за дополнительного theme.
    assert primary_course_topic_for_puzzle(puzzles[0]) == "check-mate-stalemate"
    assert primary_course_topic_for_puzzle(puzzles[2]) == "special-rules"


def test_every_adaptive_non_fallback_topic_matches_primary_topic():
    weak = [{"slug": "knight"}, {"slug": "rook"}, {"slug": "queen"}]
    strong = [
        {"slug": "pawn"},
        {"slug": "special-rules"},
        {"slug": "check-mate-stalemate"},
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
