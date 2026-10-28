"""Чистая логика B1: ранжирование тем и подбор 70/30."""

from __future__ import annotations

import random
from collections.abc import Iterable

from backend.services.course_topics import primary_course_topic_for_puzzle
from backend.services.difficulty_logic import (
    DEFAULT_DIFFICULTY,
    puzzle_matches_difficulty,
)


def allocation_for_count(count: int) -> tuple[int, int]:
    """Возвращает квоты weak/strong с целевой пропорцией 70/30."""
    count = max(0, int(count))
    weak = int(round(count * 0.70))
    weak = min(count, max(0, weak))
    return weak, count - weak


def _phase_focus_puzzles(
    puzzles: list[dict],
    phase_stats: list[dict] | None,
    *,
    count: int,
    topic_difficulties: dict[str, int] | None,
) -> tuple[list[dict], dict]:
    """Reserve up to 20% of a set for well-observed weaker game phases."""
    count = max(0, int(count))
    phase_slots = int(round(count * 0.20))
    reliable = [
        phase for phase in (phase_stats or [])
        if phase.get("key") in {"opening", "middlegame", "endgame"}
        and phase.get("accuracy_percent") is not None
        and int(phase.get("moves_evaluated") or 0) >= 5
    ]
    reliable.sort(key=lambda phase: (float(phase["accuracy_percent"]), phase["key"]))
    metadata = {
        "enabled": False,
        "reason": "insufficient_history" if not reliable else "no_phase_slots",
        "min_evaluated_moves": 5,
        "max_share": 0.20,
        "explanation": (
            "Для учёта этапа нужно не менее 5 оценённых ходов."
            if not reliable else
            "В этом наборе слишком мало мест, чтобы выделить отдельную квоту для этапа."
        ),
        "requested_slots": phase_slots,
        "selected": 0,
        "focus_phase": reliable[0]["key"] if reliable else None,
        "phases": [
            {
                "key": phase["key"],
                "accuracy_percent": phase["accuracy_percent"],
                "moves_evaluated": phase["moves_evaluated"],
                "available": 0,
                "selected": 0,
            }
            for phase in reliable
        ],
    }
    if not reliable or phase_slots == 0:
        return [], metadata

    weights = [max(1.0, 101.0 - float(phase["accuracy_percent"])) for phase in reliable]
    weight_sum = sum(weights)
    raw_quotas = [phase_slots * weight / weight_sum for weight in weights]
    quotas = [int(value) for value in raw_quotas]
    for index in sorted(
        range(len(reliable)),
        key=lambda i: (-(raw_quotas[i] - quotas[i]), i),
    )[: phase_slots - sum(quotas)]:
        quotas[index] += 1

    phase_pools: list[list[dict]] = []
    for index, phase in enumerate(reliable):
        metadata["phases"][index]["available"] = sum(
            phase["key"] in set(puzzle.get("themes") or [])
            and bool(puzzle.get("id"))
            for puzzle in puzzles
        )
    for phase in reliable:
        matching = [
            puzzle for puzzle in puzzles
            if phase["key"] in set(puzzle.get("themes") or [])
            and puzzle.get("id")
        ]
        preferred = []
        other = []
        for puzzle in matching:
            slug = primary_course_topic_for_puzzle(puzzle)
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            (preferred if puzzle_matches_difficulty(puzzle, difficulty) else other).append(puzzle)
        random.shuffle(preferred)
        random.shuffle(other)
        phase_pools.append(other + preferred)

    selected_ids: set[str] = set()
    selected_by_phase = [0] * len(reliable)
    picked: list[dict] = []

    def add_candidate(index: int) -> bool:
        pool = phase_pools[index]
        while pool:
            puzzle = pool.pop()
            puzzle_id = str(puzzle.get("id") or "")
            if not puzzle_id or puzzle_id in selected_ids:
                continue
            selected_ids.add(puzzle_id)
            slug = primary_course_topic_for_puzzle(puzzle)
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            picked.append({
                **puzzle,
                "adaptive_group": "phase_focus",
                "adaptive_topic": slug,
                "adaptive_phase": reliable[index]["key"],
                "adaptive_difficulty": difficulty,
                "difficulty_match": puzzle_matches_difficulty(puzzle, difficulty),
            })
            selected_by_phase[index] += 1
            return True
        return False

    for index, quota in enumerate(quotas):
        for _ in range(quota):
            add_candidate(index)

    # If a phase pool is small, spend the unused phase slots on the next
    # weakest phase with available tasks; remaining slots use the normal selector.
    while len(picked) < phase_slots:
        progressed = False
        for index in range(len(reliable)):
            if add_candidate(index):
                progressed = True
                break
        if not progressed:
            break

    for index, phase in enumerate(reliable):
        metadata["phases"][index]["selected"] = selected_by_phase[index]
    metadata.update({
        "enabled": bool(picked),
        "reason": "applied" if picked else "no_matching_puzzles",
        "explanation": (
            "Этапы с более низкой точностью получают большую долю задач; остальные места распределяются по темам курса и сложности."
            if picked else
            "Для этапов с надёжной статистикой не нашлось подходящих задач; набор сформирован обычным способом."
        ),
        "selected": len(picked),
        "focus_phase": reliable[0]["key"],
    })
    return picked, metadata


def rank_topic_profile(topics: list[dict]) -> tuple[list[dict], list[dict]]:
    """Выбирает нижние 3 и верхние 3 темы среди реально пройденных.

    Тема без попыток (observed=False) не может считаться ни слабой, ни
    сильной: по ней ещё нет данных. Поэтому в weak_topics/strong_topics
    попадают только темы с attempts > 0. При нулевом прогрессе оба списка
    пусты — иначе витрина/админка показывала бы «слабые: пешка, конь, слон»
    у пользователя без единой попытки.
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

    observed = [item for item in enriched if item["observed"]]

    weak = sorted(
        observed,
        key=lambda x: (x["rank_accuracy"], -int(x.get("attempts") or 0), x["_order"]),
    )[:3]

    weak_slugs = {item["slug"] for item in weak}
    strong = sorted(
        (item for item in observed if item["slug"] not in weak_slugs),
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
    matching_difficulty: bool,
    topic_difficulties: dict[str, int] | None = None,
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
        difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
        candidates = [p for p in pool if puzzle_matches_difficulty(p, difficulty) == matching_difficulty]
        random.shuffle(candidates)
        pools[slug] = candidates

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
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            result.append(
                {
                    **puzzle,
                    "adaptive_group": group,
                    "adaptive_topic": slug,
                    "adaptive_difficulty": difficulty,
                    "difficulty_match": puzzle_matches_difficulty(puzzle, difficulty),
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
    topic_difficulties: dict[str, int] | None = None,
    phase_stats: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Формирует 70/30 выборку, по возможности сохраняя квоты."""
    count = max(1, int(count))
    phase_picked, phase_meta = _phase_focus_puzzles(
        puzzles,
        phase_stats,
        count=count,
        topic_difficulties=topic_difficulties,
    )
    if phase_picked:
        selected_ids = {str(puzzle["id"]) for puzzle in phase_picked}
        remaining = [puzzle for puzzle in puzzles if str(puzzle.get("id") or "") not in selected_ids]
        topic_picked, topic_meta = select_adaptive_puzzles(
            remaining,
            weak_topics=weak_topics,
            strong_topics=strong_topics,
            count=count - len(phase_picked),
            topic_difficulties=topic_difficulties,
        ) if len(phase_picked) < count else ([], {
            "weak_quota": 0,
            "strong_quota": 0,
            "weak_selected": 0,
            "strong_selected": 0,
            "fallback_selected": 0,
            "difficulty_matched": 0,
            "difficulty_fallback": 0,
        })
        picked = phase_picked + topic_picked
        random.shuffle(picked)
        picked.sort(key=lambda puzzle: puzzle.get("difficulty_match") is not True)
        phase_meta["selected"] = len(phase_picked)
        return picked[:count], {
            **topic_meta,
            "requested": count,
            "topic_requested": count - len(phase_picked),
            "phase_adjustment": phase_meta,
        }

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
        matching_difficulty=True,
        topic_difficulties=topic_difficulties,
    )
    strong = _take_round_robin(
        puzzles,
        topic_slugs=strong_slugs,
        count=strong_quota,
        selected_ids=selected_ids,
        group="strong",
        matching_difficulty=True,
        topic_difficulties=topic_difficulties,
    )

    # Если целевые темы бедны, добираем подходящую сложность из всей базы.
    # Для новичка это важнее строгого соблюдения квоты по темам.
    picked = weak + strong
    fallback_count = 0
    if len(picked) < count:
        preferred = []
        for puzzle in puzzles:
            puzzle_id = str(puzzle.get("id", ""))
            if not puzzle_id or puzzle_id in selected_ids:
                continue
            slug = primary_course_topic_for_puzzle(puzzle)
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            if puzzle_matches_difficulty(puzzle, difficulty):
                preferred.append(puzzle)
        random.shuffle(preferred)
        for puzzle in preferred[: count - len(picked)]:
            puzzle_id = str(puzzle["id"])
            selected_ids.add(puzzle_id)
            slug = primary_course_topic_for_puzzle(puzzle)
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            picked.append(
                {
                    **puzzle,
                    "adaptive_group": "fallback",
                    "adaptive_topic": slug,
                    "adaptive_difficulty": difficulty,
                    "difficulty_match": True,
                }
            )
            fallback_count += 1

    # Лишь после исчерпания подходящих по сложности задач берём соседние
    # диапазоны в целевых темах, а затем из всей базы.
    if len(weak) < weak_quota:
        extra = _take_round_robin(
            puzzles,
            topic_slugs=weak_slugs,
            count=min(weak_quota - len(weak), count - len(picked)),
            selected_ids=selected_ids,
            group="weak",
            matching_difficulty=False,
            topic_difficulties=topic_difficulties,
        )
        weak += extra
        picked += extra
    if len(strong) < strong_quota:
        extra = _take_round_robin(
            puzzles,
            topic_slugs=strong_slugs,
            count=min(strong_quota - len(strong), count - len(picked)),
            selected_ids=selected_ids,
            group="strong",
            matching_difficulty=False,
            topic_difficulties=topic_difficulties,
        )
        strong += extra
        picked += extra

    if len(picked) < count:
        remaining = [p for p in puzzles if str(p.get("id", "")) not in selected_ids]
        random.shuffle(remaining)
        for puzzle in remaining[: count - len(picked)]:
            puzzle_id = str(puzzle.get("id", ""))
            if not puzzle_id:
                continue
            selected_ids.add(puzzle_id)
            slug = primary_course_topic_for_puzzle(puzzle)
            difficulty = int((topic_difficulties or {}).get(slug, DEFAULT_DIFFICULTY))
            picked.append(
                {
                    **puzzle,
                    "adaptive_group": "fallback",
                    "adaptive_topic": slug,
                    "adaptive_difficulty": difficulty,
                    "difficulty_match": puzzle_matches_difficulty(puzzle, difficulty),
                }
            )
            fallback_count += 1

    random.shuffle(picked)
    # В начале набора всегда показываем задачи подходящей сложности. Это
    # особенно важно для новичка, у которого пока нет истории решений.
    picked.sort(key=lambda puzzle: puzzle.get("difficulty_match") is not True)
    final = picked[:count]
    difficulty_matched = sum(1 for p in final if p.get("difficulty_match") is True)
    return final, {
        "requested": count,
        "weak_quota": weak_quota,
        "strong_quota": strong_quota,
        "weak_selected": len(weak),
        "strong_selected": len(strong),
        "fallback_selected": fallback_count,
        "difficulty_matched": difficulty_matched,
        "difficulty_fallback": len(final) - difficulty_matched,
        "phase_adjustment": phase_meta,
    }
