from backend.services.assessment import (
    GROUP_ALLOCATION,
    calculate_performance_rating,
    onboarding_rating,
    rating_group_for,
)


def test_rating_group_boundaries():
    assert rating_group_for(0)["key"] == "0-1000"
    assert rating_group_for(1000)["key"] == "0-1000"
    assert rating_group_for(1001)["key"] == "1001-1500"
    assert rating_group_for(1500)["key"] == "1001-1500"
    assert rating_group_for(1501)["key"] == "1501-1900"
    assert rating_group_for(1900)["key"] == "1501-1900"
    assert rating_group_for(1901)["key"] == "1901+"


def test_each_personalized_allocation_contains_20_questions():
    assert set(GROUP_ALLOCATION) == {1, 2, 3, 4}
    assert all(sum(allocation) == 20 for allocation in GROUP_ALLOCATION.values())


def test_q1_is_fallback_when_external_rating_is_absent():
    rating, scale, band = onboarding_rating(
        {"q1": "sometimes", "q2": {"has_rating": False}}
    )
    assert rating == 1200
    assert scale == "self_report"
    assert band == 2


def test_external_rating_overrides_q1():
    rating, scale, band = onboarding_rating(
        {
            "q1": "never",
            "q2": {
                "has_rating": True,
                "platform": "lichess",
                "rating_type": "blitz",
                "rating": 1725,
            },
        }
    )
    assert rating == 1725
    assert scale == "lichess_blitz"
    assert band == 3


def test_performance_rating_moves_in_expected_direction():
    puzzles = {
        f"p{i}": {"id": f"p{i}", "rating": 1500}
        for i in range(10)
    }
    correct = [{"puzzle_id": f"p{i}", "correct": True} for i in range(10)]
    wrong = [{"puzzle_id": f"p{i}", "correct": False} for i in range(10)]

    assert calculate_performance_rating(
        initial_rating=1200, answers=correct, puzzle_lookup=puzzles
    ) > 1200
    assert calculate_performance_rating(
        initial_rating=1200, answers=wrong, puzzle_lookup=puzzles
    ) < 1200
