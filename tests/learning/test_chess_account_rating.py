from backend.api_gateway.routes import chess_profile
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


def test_zero_game_account_rating_capped_and_unusable():
    profile = {
        "platform": "chesscom",
        "perfs": {
            "blitz": {"rating": 1400, "games": 0, "rd": None, "provisional": False},
        },
    }
    result = select_profile_rating(profile)
    assert result["rating"] == chess_profile.NO_GAMES_RATING_CAP
    assert result["usable"] is False
    assert any("0 партий" in w for w in result["warnings"])


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_chesscom_zero_games_profile_has_no_rating(monkeypatch):
    def fake_get(url, *args, **kwargs):  # noqa: ARG001
        if url.endswith("/stats"):
            return _FakeResponse(
                {"chess_blitz": {"last": {"rating": 1400}, "record": {}}}
            )
        return _FakeResponse({"username": "brandnew", "last_online": 0})

    monkeypatch.setattr(chess_profile.requests, "get", fake_get)
    profile, status = chess_profile._chesscom_profile("brandnew")

    assert status == 200
    assert profile["perfs"]["blitz"]["games"] == 0
    assert profile["perfs"]["blitz"]["rating"] is None
    result = select_profile_rating(profile)
    assert result["rating"] is None
    assert result["usable"] is False
