from backend.api_gateway.routes.chess_profile import select_profile_rating


def test_lichess_prefers_reliable_blitz():
    profile = {
        "platform": "lichess",
        "perfs": {
            "blitz": {"rating": 1420, "games": 80, "rd": 72, "provisional": False},
            "rapid": {"rating": 1510, "games": 120, "rd": 65, "provisional": False},
            "bullet": {"rating": 1370, "games": 300, "rd": 60, "provisional": False},
        },
    }
    result = select_profile_rating(profile)
    assert result["usable"] is True
    assert result["rating_type"] == "blitz"
    assert result["rating"] == 1420
    assert result["scale"] == "lichess_blitz"


def test_lichess_falls_back_to_rapid_when_blitz_is_unreliable():
    profile = {
        "platform": "lichess",
        "perfs": {
            "blitz": {"rating": 1440, "games": 12, "rd": 80, "provisional": False},
            "rapid": {"rating": 1560, "games": 55, "rd": 90, "provisional": False},
            "bullet": {"rating": 1400, "games": 100, "rd": 75, "provisional": False},
        },
    }
    result = select_profile_rating(profile)
    assert result["usable"] is True
    assert result["rating_type"] == "rapid"
    assert result["rating"] == 1560


def test_unreliable_account_can_be_linked_but_not_used_for_assessment():
    profile = {
        "platform": "lichess",
        "perfs": {
            "blitz": {"rating": 1200, "games": 8, "rd": 180, "provisional": True},
        },
    }
    result = select_profile_rating(profile)
    assert result["rating"] == 1200
    assert result["usable"] is False
    assert result["warnings"]
