"""Тесты игровых endpoint'ов: /api/legal-moves, /api/move, /api/stockfish-move."""

import time

import chess
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- /api/legal-moves ---

class TestLegalMoves:
    def test_valid_start_position(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "e2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "moves" in data
        assert len(data["moves"]) == 2  # e3, e4
        assert "e3" in data["moves"]
        assert "e4" in data["moves"]

    def test_knight_moves(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "g1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "f3" in data["moves"]
        assert "h3" in data["moves"]

    def test_invalid_fen(self):
        resp = client.post("/api/legal-moves", json={"fen": "invalid", "square": "e2"})
        assert resp.status_code == 422

    def test_invalid_square(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "z9"})
        assert resp.status_code == 422


# --- /api/move ---

class TestMakeMove:
    def test_valid_pawn_push(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "e2", "to_sq": "e4"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["san"] == "e4"
        assert data["status"] == "playing"
        assert data["turn"] == "b"

    def test_illegal_move(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "e2", "to_sq": "e6"
        })
        # Модель MoveRequest отклоняет нелегальный ход → 422
        assert resp.status_code == 422

    def test_checkmate(self):
        # Scholar's mate: после Qh5, Nc6, Bc4, Nf6 - Qxf7#
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        resp = client.post("/api/move", json={
            "fen": fen, "from_sq": "h5", "to_sq": "f7"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "checkmate"

    def test_invalid_fen(self):
        resp = client.post("/api/move", json={
            "fen": "bad", "from_sq": "e2", "to_sq": "e4"
        })
        assert resp.status_code == 422

    def test_invalid_square(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "z9", "to_sq": "e4"
        })
        assert resp.status_code == 422


# --- /api/stockfish-move ---

class TestAiMove:
    def test_game_over(self):
        # Scholar's mate - чёрные получили мат, игра окончена
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        resp = client.post("/api/stockfish-move", json={"fen": fen, "elo": 1500})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "Партия окончена"

    def test_valid_request(self, monkeypatch):
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._stockfish_move",
            lambda board: chess.Move.from_uci("e2e4"),
        )
        resp = client.post("/api/stockfish-move", json={"fen": START_FEN, "elo": 1500, "engine": "stockfish"})
        assert resp.status_code == 200
        data = resp.json()
        assert "fen" in data
        assert "san" in data
        assert "from" in data
        assert "to" in data
        assert data.get("engine") == "stockfish"

    def test_unavailable_maia_does_not_fall_back_to_stockfish(self, monkeypatch):
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._maia3_move",
            lambda req, board: None,
        )
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._stockfish_move",
            lambda board: pytest.fail("Maia3 must not be replaced by Stockfish"),
        )

        resp = client.post(
            "/api/maia-move",
            json={"fen": START_FEN, "elo": 1500, "engine": "maia3"},
        )

        assert resp.status_code == 503

    def test_levels_list(self):
        resp = client.get("/api/levels")
        assert resp.status_code == 200
        data = resp.json()
        assert "levels" in data
        assert len(data["levels"]) >= 7
        assert data["levels"][0]["engine"] == "maia3"
        assert data["levels"][-1]["engine"] == "stockfish"

    def test_invalid_fen(self):
        resp = client.post("/api/stockfish-move", json={"fen": "bad", "elo": 1500})
        assert resp.status_code == 422


# --- история партии для Maia3 ---

class _RecordingMaiaEngine:
    """Заглушка Maia3: запоминает доску, которую ей передали на ход."""

    def __init__(self, move: chess.Move):
        self.move = move
        self.boards: list[chess.Board] = []

    def configure(self, options: dict) -> None:
        pass

    def play(self, board: chess.Board, limit=None):
        self.boards.append(board.copy())

        class _Result:
            def __init__(self, move: chess.Move):
                self.move = move

        return _Result(self.move)


class _BrokenMaiaEngine:
    """Заглушка Maia3, которая падает на каждом ходе."""

    def __init__(self) -> None:
        self.quitted = False

    def configure(self, options: dict) -> None:
        pass

    def play(self, board: chess.Board, limit=None):
        raise chess.engine.EngineTerminatedError("process died")

    def quit(self) -> None:
        self.quitted = True


class _HangingMaiaEngine:
    """Заглушка Maia3, которая не отвечает (зависший процесс)."""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay

    def configure(self, options: dict) -> None:
        pass

    def play(self, board: chess.Board, limit=None):
        time.sleep(self.delay)
        raise chess.engine.EngineTerminatedError("too late")


class TestMaiaHistory:
    """Maia3 запущена с --use-uci-history и без истории играет вслепую.

    python-chess передаёт историю только тогда, когда у доски есть стек
    ходов: ``position fen ... moves ...``. Поэтому сервер обязан собрать
    доску заново из присланных ходов, а не отдавать голый FEN.
    """

    def test_matching_history_reaches_the_engine(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        # Роут строит доску из FEN, поэтому стека ходов у неё нет:
        # историю может дать только собранный заново replay.
        replay = chess.Board()
        for uci in ("e2e4", "e7e5"):
            replay.push_uci(uci)
        board = chess.Board(replay.fen())

        engine = _RecordingMaiaEngine(chess.Move.from_uci("g1f3"))
        monkeypatch.setattr(game_routes, "get_maia3", lambda: engine)

        request = FenRequest(
            fen=board.fen(),
            elo=1500,
            moves=["e2e4", "e7e5"],
            engine="maia3",
        )
        move = game_routes._maia3_move(request, board)

        assert move == chess.Move.from_uci("g1f3")
        assert len(engine.boards) == 1
        # Именно стек ходов несёт историю: python-chess превращает его
        # в "position fen <старт> moves e2e4 e7e5".
        assert len(engine.boards[0].move_stack) == 2

    def test_mismatched_history_falls_back_to_current_position(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        replay = chess.Board()
        for uci in ("e2e4", "e7e5"):
            replay.push_uci(uci)
        board = chess.Board(replay.fen())

        engine = _RecordingMaiaEngine(chess.Move.from_uci("g1f3"))
        monkeypatch.setattr(game_routes, "get_maia3", lambda: engine)

        # Историю прислали от другой партии - доверять ей нельзя,
        # но ход всё равно должен быть сделан по текущей позиции.
        request = FenRequest(
            fen=board.fen(),
            elo=1500,
            moves=["d2d4"],
            engine="maia3",
        )
        move = game_routes._maia3_move(request, board)

        assert move == chess.Move.from_uci("g1f3")
        assert len(engine.boards[0].move_stack) == 0
        assert engine.boards[0].fen() == board.fen()

    def test_dead_engine_is_reset(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        engine = _BrokenMaiaEngine()
        monkeypatch.setattr(game_routes, "get_maia3", lambda: engine)

        reset_calls = []
        monkeypatch.setattr(
            game_routes,
            "reset_maia3",
            lambda: reset_calls.append(True),
        )

        request = FenRequest(fen=START_FEN, elo=1500, moves=[], engine="maia3")
        move = game_routes._maia3_move(request, chess.Board())

        assert move is None
        # Без сброса мёртвый процесс остался бы в переменной модуля
        # и все следующие ходы падали бы до перезапуска сервера.
        # Попыток две, поэтому и сбросов два.
        assert reset_calls == [True, True]

    def test_dead_engine_is_retried_with_a_fresh_process(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        broken = _BrokenMaiaEngine()
        fresh = _RecordingMaiaEngine(chess.Move.from_uci("e2e4"))
        engines = [broken, fresh]

        monkeypatch.setattr(
            game_routes,
            "get_maia3",
            lambda: engines.pop(0) if engines else fresh,
        )

        reset_calls = []
        monkeypatch.setattr(
            game_routes,
            "reset_maia3",
            lambda: reset_calls.append(True),
        )

        request = FenRequest(fen=START_FEN, elo=1500, moves=[], engine="maia3")
        move = game_routes._maia3_move(request, chess.Board())

        # Один упавший ход раньше превращался в 503 и «висящую» партию:
        # теперь сервер поднимает новый процесс и доигрывает ход сам.
        assert move == chess.Move.from_uci("e2e4")
        assert reset_calls == [True]

    def test_hanging_engine_is_aborted_by_timeout(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        engine = _HangingMaiaEngine(delay=0.5)
        monkeypatch.setattr(game_routes, "get_maia3", lambda: engine)
        monkeypatch.setattr(game_routes, "MAIA3_PLAY_TIMEOUT", 0.05)

        reset_calls = []
        monkeypatch.setattr(
            game_routes,
            "reset_maia3",
            lambda: reset_calls.append(True),
        )

        request = FenRequest(fen=START_FEN, elo=1500, moves=[], engine="maia3")
        move = game_routes._maia3_move(request, chess.Board())

        # Зависший движок держал мьютекс, и следующие ходы вставали за ним.
        assert move is None
        assert reset_calls == [True, True]

    def test_fen_counters_do_not_discard_history(self, monkeypatch):
        from backend.api_gateway.models import FenRequest
        from backend.api_gateway.routes import game as game_routes

        replay = chess.Board()
        for uci in ("e2e4", "e7e5"):
            replay.push_uci(uci)

        # Тот же ход, но с другими счётчиками: клиентские библиотеки
        # считают полуходы и номер хода по-своему, а на игру это не влияет.
        placement, turn, castling, en_passant, _halfmove, _fullmove = (
            replay.fen().split()
        )
        board = chess.Board(
            f"{placement} {turn} {castling} {en_passant} 7 42"
        )

        engine = _RecordingMaiaEngine(chess.Move.from_uci("g1f3"))
        monkeypatch.setattr(game_routes, "get_maia3", lambda: engine)

        request = FenRequest(
            fen=board.fen(),
            elo=1500,
            moves=["e2e4", "e7e5"],
            engine="maia3",
        )
        move = game_routes._maia3_move(request, board)

        assert move == chess.Move.from_uci("g1f3")
        assert len(engine.boards[0].move_stack) == 2


# --- /api/game/new ---

class TestNewGame:
    def test_default_is_white(self):
        resp = client.post("/api/game/new", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN
        assert data["player_color"] == "w"
        assert data["ai_color"] == "b"
        assert data["turn"] == "w"
        assert data["ai_to_move"] is False

    def test_playing_black_ai_moves_first(self):
        resp = client.post("/api/game/new", json={"player_color": "b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_color"] == "b"
        assert data["ai_color"] == "w"
        assert data["ai_to_move"] is True

    def test_accepts_human_readable_color(self):
        resp = client.post("/api/game/new", json={"player_color": "черные"})
        assert resp.status_code == 200
        assert resp.json()["player_color"] == "b"

    def test_invalid_color(self):
        resp = client.post("/api/game/new", json={"player_color": "green"})
        assert resp.status_code == 422


# --- /api/game/captured-material ---

class TestCapturedMaterial:
    def test_empty_history(self):
        resp = client.post("/api/game/captured-material", json={"moves": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN
        assert data["captured_by_white"] == []
        assert data["captured_by_black"] == []
        assert data["material"]["diff"] == 0
        assert data["material"]["advantage"] == "equal"

    def test_normal_capture(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "d7d5", "e4d5"], "player_color": "w"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["captured_by_black"] == []
        assert data["counts"]["white"]["p"] == 1
        assert data["material"]["diff"] == 1
        assert data["material"]["advantage"] == "white"
        assert data["player_advantage"] == 1

    def test_player_advantage_for_black(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "d7d5", "e4d5"], "player_color": "b"},
        )
        assert resp.status_code == 200
        assert resp.json()["player_advantage"] == -1

    def test_en_passant_capture(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["counts"]["white"]["p"] == 1

    def test_promotion_capture_is_not_a_capture_of_own_piece(self):
        # 1.d4 h5 2.c4 h4 3.Qd3 h3 4.Qxh3 - ферзь берёт пешку.
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["d2d4", "h7h5", "c2c4", "h5h4", "d1d3", "h4h3", "d3h3"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["material"]["white"] == 1

    def test_illegal_move_returns_400(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e5"]},
        )
        assert resp.status_code == 400

    def test_invalid_fen(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": [], "fen": "bad"},
        )
        assert resp.status_code == 422


# --- /api/game/finish ---

class TestFinishGame:
    def test_accepts_player_color(self):
        # Пустой список ходов: фоновый record_game выходит без обращения к БД.
        resp = client.post(
            "/api/game/finish",
            json={"moves": [], "player_color": "b"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_invalid_player_color(self):
        resp = client.post(
            "/api/game/finish",
            json={"moves": [], "player_color": "green"},
        )
        assert resp.status_code == 422


# --- /api/game/review ---

class TestGameReview:
    def test_empty_history_returns_400(self):
        resp = client.post("/api/game/review", json={"moves": []})
        assert resp.status_code == 400

    def test_engine_unavailable_returns_503(self, monkeypatch):
        import backend.api_gateway.routes.game as game_routes
        from backend.services.game_review import EngineUnavailable

        def boom(*args, **kwargs):
            raise EngineUnavailable("нет движка")

        monkeypatch.setattr(game_routes, "review_game", boom)

        resp = client.post("/api/game/review", json={"moves": ["e2e4"]})
        assert resp.status_code == 503
        assert "Stockfish" in resp.json()["detail"]

    def test_invalid_move_returns_400(self, monkeypatch):
        import backend.api_gateway.routes.game as game_routes

        def boom(*args, **kwargs):
            raise ValueError("Недопустимый ход на позиции 1: e2e5")

        monkeypatch.setattr(game_routes, "review_game", boom)

        resp = client.post("/api/game/review", json={"moves": ["e2e5"]})
        assert resp.status_code == 400
        assert "e2e5" in resp.json()["detail"]

    def test_returns_review_payload(self, monkeypatch):
        import backend.api_gateway.routes.game as game_routes

        seen: dict = {}

        def fake_review(moves, player_color="w"):
            seen["moves"] = moves
            seen["player_color"] = player_color
            return {
                "player_color": player_color,
                "moves_analyzed": 1,
                "truncated": False,
                "accuracy": 87.5,
                "stars": 3,
                "thresholds": {"3": 85.0, "2": 70.0, "1": 50.0},
                "average_centipawn_loss": 5.0,
                "counts": {"best": 1},
                "moves": [],
            }

        monkeypatch.setattr(game_routes, "review_game", fake_review)

        resp = client.post(
            "/api/game/review",
            json={"moves": ["e2e4", "e7e5"], "player_color": "b"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["accuracy"] == 87.5
        assert data["stars"] == 3
        assert seen == {"moves": ["e2e4", "e7e5"], "player_color": "b"}

    def test_invalid_player_color_returns_422(self, monkeypatch):
        import backend.api_gateway.routes.game as game_routes

        monkeypatch.setattr(
            game_routes, "review_game", lambda *a, **k: {"stars": 0}
        )

        resp = client.post(
            "/api/game/review",
            json={"moves": ["e2e4"], "player_color": "green"},
        )
        assert resp.status_code == 422

