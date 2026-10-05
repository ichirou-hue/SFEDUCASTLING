"""Логика стартовой оценки уровня SFEDUCASTLING.

Онбординг Q1-Q8 даёт предварительный rating_estimate. Затем level-test
получает персональный набор из 20 паззлов: основная часть соответствует
этому рейтинговому диапазону, соседние диапазоны проверяют границы,
а накопленные попытки пользователя смещают выборку в сторону слабых тем.
"""

from __future__ import annotations

import math
import random
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.services.adaptive_training import build_weakness_profile
from backend.services.course_topics import COURSE_TOPIC_META, primary_course_topic_for_puzzle


RATING_GROUPS: list[dict[str, Any]] = [
    {"id": 1, "key": "0-1000", "title": "Начальный", "min": 0, "max": 1000},
    {"id": 2, "key": "1001-1500", "title": "Любительский", "min": 1001, "max": 1500},
    {"id": 3, "key": "1501-1900", "title": "Клубный", "min": 1501, "max": 1900},
    {"id": 4, "key": "1901+", "title": "Продвинутый", "min": 1901, "max": 9999},
]

# Распределение 20 задач по диапазонам для каждого стартового уровня.
# Оно даёт много задач своего уровня, но обязательно проверяет соседние границы.
GROUP_ALLOCATION: dict[int, list[int]] = {
    1: [12, 5, 2, 1],
    2: [4, 10, 4, 2],
    3: [2, 4, 10, 4],
    4: [1, 2, 5, 12],
}

Q1_ESTIMATES = {
    "never": 500,
    "know_moves": 800,
    "sometimes": 1200,
    "regularly": 1650,
    "tournaments": 2050,
}

Q1_LABELS = {
    "never": "Никогда",
    "know_moves": "Знаю ходы",
    "sometimes": "Играю иногда",
    "regularly": "Играю регулярно",
    "tournaments": "Играю в турнирах",
}

GOAL_LABELS = {
    "friends_family": "игра с друзьями и семьёй",
    "online_rating": "рост онлайн-рейтинга",
    "tournaments": "турнирная подготовка",
    "child": "занятия с ребёнком",
}

WEEKLY_LABELS = {
    "lt1": "до 1 часа в неделю",
    "1_3": "1–3 часа в неделю",
    "3_5": "3–5 часов в неделю",
    "5plus": "5+ часов в неделю",
}

FORMAT_LABELS = {
    "puzzles": "задачи",
    "games": "партии",
    "lessons": "уроки",
}

HINT_LABELS = {
    "yes": "подсказки включены",
    "stuck": "подсказки только при затруднении",
    "no": "без подсказок",
}


def rating_group_for(rating: int | float | None) -> dict[str, Any]:
    value = int(rating or 0)
    for group in RATING_GROUPS:
        if group["min"] <= value <= group["max"]:
            return group
    return RATING_GROUPS[-1]


def onboarding_rating(answers: dict[str, Any]) -> tuple[int, str, int]:
    """Возвращает (rating_estimate, rating_scale, prior_band)."""
    q2 = answers.get("q2") or {}
    external_rating = q2.get("rating")
    rating_usable = q2.get("rating_usable", True)
    if q2.get("has_rating") and rating_usable and external_rating is not None:
        rating = max(0, min(3500, int(external_rating)))
        platform = str(q2.get("platform") or "external")
        rating_type = str(q2.get("rating_type") or "unknown")
        scale = str(q2.get("rating_scale") or f"{platform}_{rating_type}")
    else:
        rating = Q1_ESTIMATES.get(str(answers.get("q1")), 800)
        scale = "self_report"

    group = rating_group_for(rating)
    return rating, scale, int(group["id"])


def build_onboarding_feedback(answers: dict[str, Any], rating: int) -> list[str]:
    group = rating_group_for(rating)
    feedback = [
        f"Предварительная оценка: около {rating} пунктов — диапазон «{group['key']}» ({group['title']}).",
        "Основная часть входного теста будет подобрана под этот диапазон, а часть задач будет проще и сложнее для проверки границ.",
    ]

    q1 = Q1_LABELS.get(str(answers.get("q1")))
    if q1:
        feedback.append(f"Самооценка опыта: «{q1}». Она используется как резервная оценка, если нет внешнего рейтинга.")

    q2 = answers.get("q2") or {}
    if q2.get("linked_account") and q2.get("rating_usable") and q2.get("rating") is not None:
        feedback.append(
            f"Для старта учтён рейтинг связанного аккаунта {q2.get('username')}: "
            f"{q2.get('rating')} ({q2.get('platform')}, {q2.get('rating_type')})."
        )
    elif q2.get("linked_account"):
        feedback.append(
            f"Аккаунт {q2.get('username')} привязан, но его рейтинг не прошёл критерии надёжности; "
            "для стартовой оценки использована самооценка Q1."
        )
    elif q2.get("has_rating"):
        feedback.append(
            f"Для старта учтён внешний рейтинг {q2.get('rating')} ({q2.get('platform')}, {q2.get('rating_type')})."
        )

    goals = [GOAL_LABELS.get(x, x) for x in (answers.get("q3") or [])]
    if goals:
        feedback.append("Основные цели: " + ", ".join(goals[:2]) + ".")

    weekly = WEEKLY_LABELS.get(str(answers.get("q4")))
    if weekly:
        feedback.append(f"Рекомендуемый темп обучения будет рассчитан под нагрузку: {weekly}.")

    ranked = answers.get("q5") or []
    if ranked:
        first = FORMAT_LABELS.get(str(ranked[0]), str(ranked[0]))
        feedback.append(f"В персональных рекомендациях первым приоритетом будут «{first}».")

    hints = HINT_LABELS.get(str(answers.get("q6")))
    if hints:
        feedback.append(f"Режим помощи: {hints}.")

    age = str(answers.get("q7") or "")
    if age == "under10":
        feedback.append("Для младшей возрастной группы рекомендуется короткий темп занятий и упрощённая подача.")
    elif age == "10_16":
        feedback.append("Для возрастной группы 10–16 лет рекомендации будут формулироваться короче и практичнее.")
    elif age == "17plus":
        feedback.append("Для группы 17+ используется стандартный режим объяснений и рекомендаций.")

    if answers.get("q8"):
        feedback.append("Ответ Q8 сохранён только для аналитики источников знакомства с платформой.")

    return feedback


def puzzle_rating_group(puzzle: dict[str, Any]) -> int:
    return int(rating_group_for(int(puzzle.get("rating") or 0))["id"])


def _weighted_sample(
    rng: random.Random,
    pool: list[dict[str, Any]],
    count: int,
    weak_topics: set[str],
) -> list[dict[str, Any]]:
    """Выбирает без повторов, отдавая ~60% мест слабым темам."""
    if count <= 0 or not pool:
        return []

    weak_pool = [
        p for p in pool if primary_course_topic_for_puzzle(p) in weak_topics
    ]
    normal_pool = [p for p in pool if p not in weak_pool]

    weak_target = min(len(weak_pool), round(count * 0.60)) if weak_topics else 0
    picked: list[dict[str, Any]] = []
    if weak_target:
        picked.extend(rng.sample(weak_pool, weak_target))

    remaining = count - len(picked)
    rest_pool = [p for p in normal_pool if p not in picked]
    if len(rest_pool) < remaining:
        rest_pool.extend(p for p in weak_pool if p not in picked and p not in rest_pool)
    if rest_pool and remaining:
        picked.extend(rng.sample(rest_pool, min(remaining, len(rest_pool))))
    return picked


async def personalized_level_test_puzzles(
    db: AsyncSession,
    *,
    user_id: int,
    rating_estimate: int,
    puzzle_base: dict[str, Any] | None,
    seed: int,
    total: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Формирует персональный воспроизводимый набор задач.

    Учитывает:
    * стартовый rating_estimate;
    * слабые темы из training_attempts + puzzle_attempts;
    * уже встречавшиеся пользователю пазлы;
    * seed для воспроизводимости конкретной попытки.
    """
    all_puzzles = [
        p
        for p in (puzzle_base or {}).get("puzzles", [])
        if p.get("id") and p.get("fen") and p.get("moves")
    ]
    if not all_puzzles:
        return [], {"error": "База паззлов не загружена"}

    rng = random.Random(seed)
    group = rating_group_for(rating_estimate)
    allocation = list(GROUP_ALLOCATION[int(group["id"])])

    profile = await build_weakness_profile(
        db, user_id=user_id, puzzle_base=puzzle_base
    )
    weak_topics = {
        item.get("slug")
        for item in profile.get("weak_topics", [])
        if item.get("slug") and int(item.get("attempts") or 0) > 0
    }

    seen_ids = set(
        await db.scalars(
            select(PuzzleAttempt.puzzle_id).where(PuzzleAttempt.user_id == user_id)
        )
    )
    previous_question_sets = (
        await db.scalars(
            select(LevelTest.question_ids).where(LevelTest.user_id == user_id)
        )
    ).all()
    for question_ids in previous_question_sets:
        seen_ids.update(str(x) for x in (question_ids or []))

    fresh = [p for p in all_puzzles if str(p.get("id")) not in seen_ids]
    source = fresh if len(fresh) >= total else all_puzzles

    picked: list[dict[str, Any]] = []
    picked_ids: set[str] = set()
    actual_allocation = [0, 0, 0, 0]

    for group_index, wanted in enumerate(allocation, start=1):
        pool = [
            p
            for p in source
            if puzzle_rating_group(p) == group_index
            and str(p.get("id")) not in picked_ids
        ]
        sample = _weighted_sample(rng, pool, wanted, weak_topics)
        for p in sample:
            pid = str(p.get("id"))
            if pid in picked_ids:
                continue
            picked.append(p)
            picked_ids.add(pid)
            actual_allocation[group_index - 1] += 1

    # Если в конкретной корзине мало задач, добираем из всего пула,
    # сохраняя уникальность. Так endpoint всегда стремится вернуть 20 задач.
    if len(picked) < total:
        fallback = [p for p in source if str(p.get("id")) not in picked_ids]
        rng.shuffle(fallback)
        for p in fallback:
            if len(picked) >= total:
                break
            picked.append(p)
            picked_ids.add(str(p.get("id")))
            actual_allocation[puzzle_rating_group(p) - 1] += 1

    rng.shuffle(picked)
    picked = picked[:total]

    metrics = {
        "rating_estimate": int(rating_estimate),
        "rating_group": group["key"],
        "target_allocation": allocation,
        "actual_allocation": actual_allocation,
        "weak_topics": sorted(weak_topics),
        "weak_topic_titles": [
            item.get("title")
            for item in profile.get("weak_topics", [])
            if item.get("slug") in weak_topics
        ],
        "prior_attempts": sum(int(item.get("attempts") or 0) for item in profile.get("topics", [])),
        "fresh_pool_used": source is fresh,
        "seen_puzzles": len(seen_ids),
    }
    return picked, metrics


def calculate_performance_rating(
    *,
    initial_rating: int,
    answers: list[dict[str, Any]],
    puzzle_lookup: dict[str, dict[str, Any]],
    k_factor: float = 72.0,
) -> int:
    """Elo-подобная оценка после 20 задач.

    Каждая задача выступает как «соперник» с её puzzle rating. Это даёт
    плавный итоговый рейтинг вместо жёсткого скачка только по числу верных.
    """
    rating = float(initial_rating)
    for answer in answers:
        puzzle = puzzle_lookup.get(str(answer.get("puzzle_id")))
        if not puzzle:
            continue
        puzzle_rating = float(puzzle.get("rating") or rating)
        expected = 1.0 / (1.0 + math.pow(10.0, (puzzle_rating - rating) / 400.0))
        actual = 1.0 if answer.get("correct") else 0.0
        rating += k_factor * (actual - expected)
    return max(0, min(3500, int(round(rating))))


def build_test_feedback(
    final_rating: int,
    answers: list[dict[str, Any]],
    puzzle_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    group = rating_group_for(final_rating)
    correct = sum(1 for a in answers if a.get("correct"))
    total = len(answers)
    accuracy = round(correct * 100 / total) if total else 0

    theme_stats: dict[str, list[int]] = {}
    for answer in answers:
        puzzle = puzzle_lookup.get(str(answer.get("puzzle_id")))
        topic = primary_course_topic_for_puzzle(puzzle) or "other"
        row = theme_stats.setdefault(topic, [0, 0])
        row[0] += 1
        if answer.get("correct"):
            row[1] += 1

    weak = sorted(
        (
            (topic, vals[1] / vals[0])
            for topic, vals in theme_stats.items()
            if vals[0] >= 2 and topic != "other"
        ),
        key=lambda x: x[1],
    )[:2]

    feedback = [
        f"Итоговая оценка: около {final_rating} — диапазон «{group['key']}» ({group['title']}).",
        f"Точность в тесте: {correct} из {total} ({accuracy}%).",
    ]
    if weak:
        feedback.append(
            "Для ближайших тренировок стоит уделить больше внимания темам: "
            + ", ".join(COURSE_TOPIC_META.get(topic, {}).get("title", topic) for topic, _ in weak)
            + "."
        )
    return feedback
