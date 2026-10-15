from backend.services.assessment import (
    GROUP_ALLOCATION,
    allowed_rating_groups,
    calculate_performance_rating,
    max_puzzle_rating_for_user,
    onboarding_rating,
    rating_for_selection,
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


def test_q1_self_report_is_capped_without_linked_account():
    # q1="sometimes" -> самооценка 1200, но без привязки оценка ограничена 800.
    rating, scale, band = onboarding_rating(
        {"q1": "sometimes", "q2": {"has_rating": False}}
    )
    assert rating == 800
    assert scale == "self_report"
    # prior_band по-прежнему отражает заявленный уровень (2).
    assert band == 2


def test_linked_external_rating_overrides_q1():
    rating, scale, band = onboarding_rating(
        {
            "q1": "never",
            "q2": {"platform": "lichess", "login": "user", "time_control": "blitz"},
        },
        linked={"rating": 1725, "scale": "lichess_blitz", "usable": True},
    )
    assert rating == 1725
    assert scale == "lichess_blitz"
    assert band == 3


def test_unusable_linked_rating_falls_back_to_capped_self_report():
    rating, scale, band = onboarding_rating(
        {"q1": "tournaments", "q2": None},
        linked={"rating": 2100, "scale": "lichess_blitz", "usable": False},
    )
    # Рейтинг не подтверждён -> ограничение самооценки, заявка сохраняется в band.
    assert rating == 800
    assert scale == "self_report"
    assert band == 4


def test_q1_regular_matches_frontend_code():
    # Фронт отправляет код "regular" (OnboardingModal), а не "regularly".
    rating, scale, band = onboarding_rating(
        {"q1": "regular", "q2": {"has_rating": False}}
    )
    assert rating == 800
    assert scale == "self_report"
    assert band == 3


def test_rating_for_selection_priority():
    assert rating_for_selection(2100, 1200, 2) == (2100, "level_test")
    assert rating_for_selection(None, 1200, 2) == (1200, "onboarding")
    assert rating_for_selection(None, None, 2) == (1250, "prior_band")
    assert rating_for_selection(None, None, None) == (None, "none")


def test_allowed_rating_groups_keeps_own_and_neighbors():
    assert allowed_rating_groups(1) == {1, 2}
    assert allowed_rating_groups(2) == {1, 2, 3}
    assert allowed_rating_groups(4) == {3, 4}


def test_max_puzzle_rating_is_rating_plus_200():
    assert max_puzzle_rating_for_user(None) is None
    assert max_puzzle_rating_for_user(1200) == 1400
    assert max_puzzle_rating_for_user(0) == 200
    assert max_puzzle_rating_for_user(1200, margin=0) == 1200


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
