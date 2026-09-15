"""Чистая логика B1: ранжирование тем и подбор 70/30."""

from __future__ import annotations

import random
from collections.abc import Iterable

from backend.services.course_topics import primary_course_topic_for_puzzle


def allocation_for_count(count: int) -> tuple[int, int]:
    """Возвращает квоты weak/strong с целевой пропорцией 70/30."""
    count = max(0, int(count))
    weak = int(round(count * 0.70))
    weak = min(count, max(0, weak))
    return weak, count - weak


def rank_topic_profile(topics: list[dict]) -> tuple[list[dict], list[dict]]:
    """Выбирает нижние 3 и верхние 3 темы.

    Для тем без истории используются нейтральные 50% только при ранжировании
    слабых направлений. Фактическая accuracy остаётся None. В strong_topics
    попадают только темы с реальными попытками пользователя (observed=True).
    """
    enriched = []
    for index, topic in enumerate(topics):
        item = dict(topic)
        attempts = int(item.get("attempts") or 0)
        accuracy = item.get("accuracy")
        rank_accuracy = float(accuracy) if attempts > 0 and accuracy is not None else 50.0
        item["rank_accuracy"] = rank_accuracy
        item["observed"] = attempts > 0
        item["_order"] = index
        enriched.append(item)

    weak = sorted(
        enriched,
        key=lambda x: (x["rank_accuracy"], -int(x.get("attempts") or 0), x["_order"]),
    )[:3]

    weak_slugs = {item["slug"] for item in weak}
    # «Сильная» тема должна подтверждаться реальными попытками пользователя.
    # Неизученная тема (observed=False) сохраняет нейтральный rank_accuracy=50
    # для общего профиля/ранжирования, но не может считаться сильной.
    strong_candidates = [
        item
        for item in enriched
        if item["slug"] not in weak_slugs and item["observed"]
    ]
    strong = sorted(
        strong_candidates,
        key=lambda x: (-x["rank_accuracy"], -int(x.get("attempts") or 0), x["_order"]),
    )[:3]

    def public(items: Iterable[dict]) -> list[dict]:
        result = []
        for item in items:
            copy = dict(item)
            copy.pop("_order", None)
            result.append(copy)
        return result

    return public(weak), public(strong)


def _take_round_robin(
    puzzles: list[dict],
    *,
    topic_slugs: list[str],
    count: int,
    selected_ids: set[str],
    group: str,
) -> list[dict]:
    pools: dict[str, list[dict]] = {}
    for slug in topic_slugs:
        # В адаптивном режиме пазл должен быть зачислен в ту же тему,
        # по которой он был выдан. Поэтому используем не широкое
        # пересечение Lichess themes, а единственную primary-тему.
        # Это замыкает обратную связь B1: selection -> attempt -> stats.
        pool = [
            p
            for p in puzzles
            if primary_course_topic_for_puzzle(p) == slug
        ]
        random.shuffle(pool)
        pools[slug] = pool

    result: list[dict] = []
    while len(result) < count:
        progressed = False
        for slug in topic_slugs:
            pool = pools.get(slug) or []
            while pool and str(pool[-1].get("id", "")) in selected_ids:
                pool.pop()
            if not pool:
                continue
            puzzle = pool.pop()
            puzzle_id = str(puzzle.get("id", ""))
            if not puzzle_id or puzzle_id in selected_ids:
                continue
            selected_ids.add(puzzle_id)
            result.append(
                {
                    **puzzle,
                    "adaptive_group": group,
                    "adaptive_topic": slug,
                }
            )
            progressed = True
            if len(result) >= count:
                break
        if not progressed:
            break
    return result


def select_adaptive_puzzles(
    puzzles: list[dict],
    *,
    weak_topics: list[dict],
    strong_topics: list[dict],
    count: int,
) -> tuple[list[dict], dict]:
    """Формирует 70/30 выборку, по возможности сохраняя квоты."""
    count = max(1, int(count))
    weak_quota, strong_quota = allocation_for_count(count)
    selected_ids: set[str] = set()

    weak_slugs = [item["slug"] for item in weak_topics]
    strong_slugs = [item["slug"] for item in strong_topics]

    weak = _take_round_robin(
        puzzles,
        topic_slugs=weak_slugs,
        count=weak_quota,
        selected_ids=selected_ids,
        group="weak",
    )
    strong = _take_round_robin(
        puzzles,
        topic_slugs=strong_slugs,
        count=strong_quota,
        selected_ids=selected_ids,
        group="strong",
    )

    picked = weak + strong
    fallback_count = 0
    if len(picked) < count:
        remaining = [p for p in puzzles if str(p.get("id", "")) not in selected_ids]
        random.shuffle(remaining)
        for puzzle in remaining[: count - len(picked)]:
            puzzle_id = str(puzzle.get("id", ""))
            if not puzzle_id:
                continue
            selected_ids.add(puzzle_id)
            picked.append(
                {
                    **puzzle,
                    "adaptive_group": "fallback",
                    "adaptive_topic": primary_course_topic_for_puzzle(puzzle),
                }
            )
            fallback_count += 1

    random.shuffle(picked)
    return picked[:count], {
        "requested": count,
        "weak_quota": weak_quota,
        "strong_quota": strong_quota,
        "weak_selected": len(weak),
        "strong_selected": len(strong),
        "fallback_selected": fallback_count,
    }
