"""Run a real engine in CI; unit mocks cannot validate server dependencies."""
import os

import pytest

from backend.api_gateway.routes.health import stockfish_ready
from backend.services.game_review import review_game


@pytest.mark.skipif(os.getenv("CI", "").lower() != "true", reason="Real Stockfish is installed by CI")
def test_real_stockfish_reviews_moves():
    assert stockfish_ready()
    result = review_game(["e2e4", "e7e5", "g1f3", "b8c6"], player_color="w")
    assert result["moves_scored"] > 0
    assert len(result["plies"]) == 4
