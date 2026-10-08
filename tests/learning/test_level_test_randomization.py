"""Персональный набор level-test: seed, аллокация, fallback, слабые темы.

Старый `_level_test_puzzles` (random.sample по корзинам LEVEL_BUCKETS)
заменён на `personalized_level_test_puzzles`: набор собирается по
GROUP_ALLOCATION под rating_estimate, воспроизводим seed'ом, при нехватке
задач в корзине добирает fallback'ом, слабым темам отдаёт ~60% мест через
`_weighted_sample`.
"""

import asyncio
import random

from backend.db.session import async_session_factory
from backend.services.assessment import (
    _weighted_sample,
    personalized_level_test_puzzles,
)


def _puzzle(pid: str, rating: int, themes: tuple[str, ...] = ()) -> dict:
    return {
        "id": pid,
        "rating": rating,
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "moves": "a1a2",
        "themes": list(themes),
    }


def _wide_pool() -> list[dict]:
    """8 задач на каждую из четырёх рейтинговых групп."""
    by_group = [
        range(600, 1001, 50),       # 0-1000
        range(1050, 1501, 50),      # 1001-1500
        range(1550, 1901, 50),      # 1501-1900
        range(1950, 2301, 50),      # 1901+
    ]
    puzzles: list[dict] = []
    for group_index, ratings in enumerate(by_group):
        for offset, rating in enumerate(ratings):
            puzzles.append(_puzzle(f"g{group_index + 1}_{offset}", rating))
    return puzzles


def _personalized(
    puzzle_base: dict,
    *,
    seed: int,
    rating_estimate: int = 1200,
    total: int = 20,
    user_id: int = 424242,
) -> tuple[list[dict], dict]:
    async def _go():
        async with async_session_factory() as db:
            return await personalized_level_test_puzzles(
                db,
                user_id=user_id,
                rating_estimate=rating_estimate,
                puzzle_base=puzzle_base,
                seed=seed,
                total=total,
            )

    return asyncio.run(_go())


def test_same_seed_is_reproducible():
    base = {"puzzles": _wide_pool()}
    first, _ = _personalized(base, seed=7)
    second, _ = _personalized(base, seed=7)
    assert [p["id"] for p in first] == [p["id"] for p in second]


def test_different_seeds_give_different_selection():
    base = {"puzzles": _wide_pool()}
    first, _ = _personalized(base, seed=7)
    second, _ = _personalized(base, seed=8)
    assert [p["id"] for p in first] != [p["id"] for p in second]


def test_returns_20_unique_questions_with_metrics():
    base = {"puzzles": _wide_pool()}
    picked, metrics = _personalized(base, seed=3)
    ids = [p["id"] for p in picked]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    source_ids = {p["id"] for p in base["puzzles"]}
    assert set(ids) <= source_ids
    # rating_estimate=1200 → группа 2 → аллокация [4, 10, 4, 2].
    assert metrics["rating_group"] == "1001-1500"
    assert metrics["target_allocation"] == [4, 10, 4, 2]
    assert sum(metrics["actual_allocation"]) == 20


def test_fallback_when_pool_smaller_than_total():
    small = {"puzzles": _wide_pool()[:10]}
    picked, _ = _personalized(small, seed=5, total=20)
    ids = [p["id"] for p in picked]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_missing_puzzle_base_returns_error():
    picked, metrics = _personalized({"puzzles": []}, seed=1)
    assert picked == []
    assert metrics["error"] == "База паззлов не загружена"


def test_weighted_sample_respects_count_and_uniqueness():
    rng = random.Random(0)
    pool = [_puzzle(f"p{i}", 1000) for i in range(4)]
    picked = _weighted_sample(rng, pool, 10, set())
    assert len(picked) == 4
    assert len({p["id"] for p in picked}) == 4


def test_weighted_sample_gives_sixty_percent_to_weak_topics():
    rng = random.Random(0)
    weak = [_puzzle(f"w{i}", 1000, ("knightEndgame",)) for i in range(5)]
    normal = [_puzzle(f"n{i}", 1000) for i in range(5)]
    picked = _weighted_sample(rng, weak + normal, 5, {"knight"})
    picked_ids = {p["id"] for p in picked}
    # round(5 * 0.60) = 3 места уходят слабым темам.
    assert len(picked_ids & {p["id"] for p in weak}) == 3
    assert len(picked_ids) == 5


def test_weighted_sample_ignores_weak_topics_when_set_is_empty():
    rng = random.Random(0)
    weak = [_puzzle(f"w{i}", 1000, ("knightEndgame",)) for i in range(5)]
    picked = _weighted_sample(rng, weak, 3, set())
    assert len(picked) == 3
    assert len({p["id"] for p in picked}) == 3
