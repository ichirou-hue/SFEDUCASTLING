"""Тесты разбора партии: точность, звёзды, результат и оценки ходов."""

import chess
import pytest

import backend.services.game_review as game_review


class FakeEngine:
    """Минимальная заглушка Stockfish с заданным сценарием оценок.

    ``evals`` - очередь оценок в сантипешках от лица белых: каждый
    вызов ``get_evaluation`` забирает следующее значение. Когда очередь
    пуста, отдаётся 0 (равная позиция).
    """

    def __init__(self, evals: list[int] | None = None):
        self.evals = list(evals or [])
        self.searched_fens: list[str] = []

    def set_fen_position(self, fen: str) -> None:
        self.searched_fens.append(fen)

    def get_evaluation(self, searchtime: int | None = None) -> dict:
        value = self.evals.pop(0) if self.evals else 0
        return {"type": "cp", "value": value}


# --- win_percent / move_accuracy ---

class TestWinPercent:
    def test_equal_position_is_fifty(self):
        assert game_review.win_percent(0) == pytest.approx(50.0)

    def test_positive_eval_above_fifty(self):
        assert game_review.win_percent(100) > 50.0
        assert game_review.win_percent(500) > game_review.win_percent(100)

    def test_symmetric(self):
        assert game_review.win_percent(150) + game_review.win_percent(-150) == pytest.approx(100.0)

    def test_bounds(self):
        assert 0.0 <= game_review.win_percent(-10000) <= 100.0
        assert 0.0 <= game_review.win_percent(10000) <= 100.0


class TestMoveAccuracy:
    def test_no_drop_is_full_accuracy(self):
        assert game_review.move_accuracy(50.0, 50.0) == pytest.approx(100.0, abs=1e-3)

    def test_better_move_is_not_penalised(self):
        # Рост вероятности победы (win_after > win_before) - не потеря.
        assert game_review.move_accuracy(40.0, 60.0) == pytest.approx(100.0, abs=1e-3)

    def test_big_drop_is_low_accuracy(self):
        assert game_review.move_accuracy(90.0, 10.0) < 10.0

    def test_accuracy_never_negative(self):
        assert game_review.move_accuracy(100.0, 0.0) >= 0.0


# --- stars_for_accuracy / потолок по результату ---

class TestStars:
    @pytest.mark.parametrize(
        ("accuracy", "stars"),
        [
            (None, 0),
            (0.0, 0),
            (49.9, 0),
            (50.0, 1),
            (69.9, 1),
            (70.0, 2),
            (84.9, 2),
            (85.0, 3),
            (100.0, 3),
        ],
    )
    def test_thresholds(self, accuracy, stars):
        assert game_review.stars_for_accuracy(accuracy) == stars

    @pytest.mark.parametrize(
        ("outcome", "cap"),
        [("win", 3), ("draw", 2), ("loss", 1), (None, 3)],
    )
    def test_cap_by_outcome(self, outcome, cap):
        assert game_review.star_cap_for_outcome(outcome) == cap


# --- game_outcome ---

class TestGameOutcome:
    def test_checkmate_against_player_is_loss(self):
        # Мат белому: 1. f3 e5 2. g4 Qh4#
        board = chess.Board()
        for token in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push(chess.Move.from_uci(token))

        assert game_review.game_outcome(board, chess.WHITE) == "loss"
        assert game_review.game_outcome(board, chess.BLACK) == "win"

    def test_checkmate_delivered_by_player_is_win(self):
        # Мат чёрному: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
        board = chess.Board()
        for token in ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"):
            board.push(chess.Move.from_uci(token))

        assert game_review.game_outcome(board, chess.WHITE) == "win"
        assert game_review.game_outcome(board, chess.BLACK) == "loss"

    def test_stalemate_is_draw(self):
        board = chess.Board()
        for token in (
            "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5",
            "a5c7", "a6h6", "h2h4", "f7f6", "c7d7", "e8f7",
            "d7b7", "d8d3", "b7b8", "d3h7", "b8c8", "f7g6", "c8e6",
        ):
            board.push(chess.Move.from_uci(token))

        assert board.is_stalemate()
        assert game_review.game_outcome(board, chess.WHITE) == "draw"

    def test_unfinished_game_has_no_outcome(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))

        assert game_review.game_outcome(board, chess.WHITE) is None


# --- _position_eval ---

class TestPositionEval:
    def test_white_to_move_keeps_engine_value(self):
        engine = FakeEngine([120])
        cp, mate = game_review._position_eval(engine, chess.Board())
        assert cp == pytest.approx(120.0)
        assert mate is None

    def test_black_to_move_flips_engine_value(self):
        engine = FakeEngine([120])
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        cp, mate = game_review._position_eval(engine, board)
        assert cp == pytest.approx(-120.0)
        assert mate is None

    def test_mate_is_reported_and_not_searched(self):
        # Мат на доске: движок не вызываем, отдаём мат 0 и крайнюю оценку.
        board = chess.Board()
        for token in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push(chess.Move.from_uci(token))

        engine = FakeEngine()
        cp, mate = game_review._position_eval(engine, board)

        assert cp == pytest.approx(-game_review.MATE_CP)
        assert mate == 0
        assert engine.searched_fens == []

    def test_mate_on_board_for_black_is_positive(self):
        # Мат чёрному (белые поставили мат): очередь хода у чёрных.
        board = chess.Board()
        for token in ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"):
            board.push(chess.Move.from_uci(token))

        cp, mate = game_review._position_eval(FakeEngine(), board)

        assert cp == pytest.approx(game_review.MATE_CP)
        assert mate == 0

    def test_draw_is_zero(self):
        board = chess.Board()
        for token in (
            "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5",
            "a5c7", "a6h6", "h2h4", "f7f6", "c7d7", "e8f7",
            "d7b7", "d8d3", "b7b8", "d3h7", "b8c8", "f7g6", "c8e6",
        ):
            board.push(chess.Move.from_uci(token))

        assert game_review._position_eval(FakeEngine(), board) == (0.0, None)

    def test_engine_mate_is_capped_at_mate_cp(self):
        class MateEngine(FakeEngine):
            def get_evaluation(self, searchtime: int | None = None) -> dict:
                return {"type": "mate", "value": 2}

        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        # Ход чёрных: мат в 2 ставит сторона, чья очередь (чёрные).
        cp, mate = game_review._position_eval(MateEngine(), board)

        assert mate == -2
        # Для расчётов мат - это предел, а не «десять тысяч пешек».
        assert cp == pytest.approx(-game_review.MATE_CP)


# --- classify ---

class TestClassify:
    def test_no_drop_is_best(self):
        assert game_review.classify(50.0, 50.0) == "best"
        assert game_review.classify(50.0, 55.0) == "best"

    def test_thresholds_by_win_percent_drop(self):
        # Ступени по падению вероятности победы, в процентных пунктах:
        # 2 / 5 / 10 / 20 / 30.
        assert game_review.classify(60.0, 59.0) == "best"
        assert game_review.classify(60.0, 56.0) == "excellent"
        assert game_review.classify(60.0, 52.0) == "good"
        assert game_review.classify(60.0, 45.0) == "inaccuracy"
        assert game_review.classify(60.0, 35.0) == "mistake"
        assert game_review.classify(60.0, 20.0) == "blunder"

    def test_same_cp_drop_differs_by_position(self):
        # Потеря 300 cp в равной позиции - ошибка, а в выигранной -
        # почти ничего: победа и так близка.
        equal_drop = game_review.classify(
            game_review.win_percent(300), game_review.win_percent(0)
        )
        winning_drop = game_review.classify(
            game_review.win_percent(900), game_review.win_percent(600)
        )

        assert equal_drop == "mistake"
        assert winning_drop == "good"


# --- game_accuracy ---

class TestGameAccuracy:
    def test_perfect_game_is_hundred(self):
        # Вероятность победы не менялась: все ходы идеальны.
        assert game_review.game_accuracy([50.0] * 9, chess.WHITE) == 100.0

    def test_no_player_moves(self):
        assert game_review.game_accuracy([50.0, 50.0], chess.BLACK) is None

    def test_accuracy_is_measured_from_player_side(self):
        # Белые упускают перевес, чёрные возвращают его своим ходом:
        # каждый оценивается только по собственным ходам.
        values = [70.0, 70.0, 40.0, 40.0, 70.0]

        assert game_review.game_accuracy(values, chess.WHITE) == 100.0

        black = game_review.game_accuracy(values, chess.BLACK)
        assert black is not None and black < 100.0

    def test_black_mistake_lowers_black_accuracy(self):
        # Ход чёрных поднял вероятность победы белых с 30 до 50 процентов.
        # Для чёрных это потеря: точность должна упасть, а не стать 100%.
        values = [30.0, 30.0, 50.0]

        assert game_review.game_accuracy(values, chess.WHITE) == 100.0

        black = game_review.game_accuracy(values, chess.BLACK)
        assert black is not None and black < 60.0

    def test_white_mistake_keeps_black_accuracy_high(self):
        # Обратный случай: ошибся белый, чёрный ход был сильным.
        values = [30.0, 30.0, 20.0]

        assert game_review.game_accuracy(values, chess.BLACK) == 100.0

    def test_blunders_pull_accuracy_down(self):
        # Игрок белыми: первый ход ничего не потерял, потом два зевка.
        values = [50.0, 50.0, 50.0, 20.0, 20.0, 2.0, 2.0]
        accuracy = game_review.game_accuracy(values, chess.WHITE)

        assert accuracy is not None
        assert accuracy < 50.0

    def test_harmonic_mean_penalises_single_blunder(self):
        # Один провал должен тянуть итог вниз заметнее, чем арифметика.
        values = [50.0, 50.0, 50.0, 2.0] + [2.0] + [50.0] * 15
        accuracy = game_review.game_accuracy(values, chess.WHITE)

        assert accuracy is not None
        assert accuracy < 90.0

    def test_decided_positions_are_not_scored(self):
        # У игрока белыми живой только первый ход, дальше партия решена:
        # ходы в безнадёжной позиции не должны тянуть точность вверх.
        values = [50.0, 30.0, 10.0, 8.0, 5.0, 3.0]

        accuracy = game_review.game_accuracy(values, chess.WHITE)

        assert accuracy == pytest.approx(
            game_review.move_accuracy(50.0, 30.0), abs=0.1
        )

    def test_all_decided_positions_give_no_accuracy(self):
        # Игрок получил безнадёжную позицию с самого начала: оценивать
        # его ходы бессмысленно, честнее показать прочерк.
        assert game_review.game_accuracy([5.0, 5.0, 3.0, 2.0], chess.WHITE) is None


# --- веса позиций ---

class TestPositionWeights:
    def test_weights_follow_win_values(self):
        calm = [50.0] * 10
        sharp = [50.0, 30.0, 70.0, 20.0, 80.0, 50.0, 90.0, 10.0, 50.0, 50.0]

        calm_weights = game_review._position_weights(calm)
        sharp_weights = game_review._position_weights(sharp)

        assert max(calm_weights) < max(sharp_weights)

    def test_weights_are_clamped(self):
        weights = game_review._position_weights([0.0] * 20 + [100.0] * 20)

        assert all(0.5 <= weight <= 12.0 for weight in weights)


# --- MultiPV ---

class TestEnsureSingleLine:
    def test_multipv_is_reset_to_one(self):
        class ParameterEngine:
            def __init__(self):
                self.parameters = {"MultiPV": 5}
                self.updated = None

            def get_engine_parameters(self):
                return dict(self.parameters)

            def update_engine_parameters(self, values):
                self.updated = values
                self.parameters.update(values)

        engine = ParameterEngine()
        game_review._ensure_single_line(engine)

        assert engine.updated == {"MultiPV": 1}

    def test_single_line_is_left_alone(self):
        class ParameterEngine:
            def __init__(self):
                self.updated = None

            def get_engine_parameters(self):
                return {"MultiPV": 1}

            def update_engine_parameters(self, values):
                self.updated = values

        engine = ParameterEngine()
        game_review._ensure_single_line(engine)

        assert engine.updated is None


# --- review_game ---

FORCED_LOSS = ("f2f3", "e7e5", "g2g4", "d8h4")  # мат чёрных после 2. g4


class TestReviewGame:
    def test_engine_unavailable(self, monkeypatch):
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: None)

        with pytest.raises(game_review.EngineUnavailable):
            game_review.review_game(["e2e4"])

    def test_invalid_move(self, monkeypatch):
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: FakeEngine())

        with pytest.raises(ValueError):
            game_review.review_game(["e2e5"])

    def test_aggregates_only_player_moves(self, monkeypatch):
        engine = FakeEngine([0])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5"], player_color="w", search_time_ms=1
        )

        assert result["player_color"] == "w"
        assert result["moves_analyzed"] == 1
        assert result["accuracy"] == 100.0
        assert result["stars"] == 3
        assert result["counts"]["best"] == 1
        assert result["truncated"] is False
        assert [move["san"] for move in result["moves"]] == ["e4"]

    def test_plies_cover_both_sides_with_eval(self, monkeypatch):
        # Заглушка отдаёт оценку от лица ходящей стороны, как Stockfish.
        # После e4 ход чёрных: -40 для чёрных = +40 для белых.
        engine = FakeEngine([0, -40, -15])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5"], player_color="w", search_time_ms=1
        )

        plies = result["plies"]
        assert len(plies) == 2
        # Оценка после хода - от лица белых, как на шахматных сайтах.
        assert plies[0]["eval_cp"] == pytest.approx(40.0)
        assert plies[1]["eval_cp"] == pytest.approx(-15.0)
        assert plies[0]["by_player"] is True
        assert plies[1]["by_player"] is False
        assert plies[1]["classification"] is None
        assert plies[1]["move_no"] == 1

    def test_no_initial_search_for_black(self, monkeypatch):
        engine = FakeEngine([0, 0])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        game_review.review_game(["e2e4", "e7e5"], player_color="b", search_time_ms=1)

        # Стартовая позиция не оценивается: у чёрных «до» - позиция после
        # хода белых, она уже посчитана.
        assert len(engine.searched_fens) == 2

    def test_loss_is_measured_from_player_side(self, monkeypatch):
        # Старт 0, после хода белых +250: перевес белых вырос, потери нет.
        # Заглушка отвечает от лица чёрных: -250 = белым +250.
        engine = FakeEngine([0, -250, -250])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5"], player_color="w", search_time_ms=1
        )

        assert result["moves"][0]["loss_cp"] == 0.0
        assert result["moves"][0]["accuracy"] == 100.0
        assert result["moves"][0]["classification"] == "best"

    def test_white_drop_is_counted_as_loss(self, monkeypatch):
        # Белые были лучше (+300), после их хода стало -100: потеря 400.
        # Заглушка: до хода оценка белых +300, после - оценка чёрных +100.
        engine = FakeEngine([300, 100])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["f2f3", "e7e5"], player_color="w", search_time_ms=1
        )

        move = result["moves"][0]
        assert move["loss_cp"] == pytest.approx(400.0)
        # Падение вероятности победы с 75 % до 41 % - это зевок.
        assert move["classification"] == "blunder"
        assert move["accuracy"] < 50.0

    def test_black_loss_is_not_zeroed_by_sign(self, monkeypatch):
        # После e4 белым +200 (для чёрных -200), чёрные ошиблись: белым
        # +600 (ход белых, оценка от лица белых), потеря чёрных 400.
        engine = FakeEngine([-200, 600])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "c7c5"], player_color="b", search_time_ms=1
        )

        move = result["moves"][0]
        assert move["move_no"] == 1
        assert move["loss_cp"] == pytest.approx(400.0)
        assert move["classification"] == "mistake"

    def test_moves_in_decided_position_are_not_scored(self, monkeypatch):
        # Игрок чёрными: соперник ушёл на +8 пешек, дальше чёрные просто
        # ходят. Такие ходы не должны получать 100 % и вытягивать итог.
        engine = FakeEngine([-50, 800, 800, 800])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5", "g1f3", "b8c6"], player_color="b", search_time_ms=1
        )

        assert result["moves_analyzed"] == 2
        assert result["moves_scored"] == 1
        assert result["moves_decided"] == 1
        assert result["accuracy"] is not None and result["accuracy"] < 50.0

        shuffled = result["moves"][1]
        assert shuffled["decided"] is True
        assert shuffled["classification"] in {"best", "excellent", "good", "inaccuracy", "mistake", "blunder"}
        assert result["counts"]["best"] == 0

    def test_loss_caps_stars(self, monkeypatch):
        # Идеальная точность, но игрок получил мат: максимум одна звезда.
        engine = FakeEngine()
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            list(FORCED_LOSS), player_color="w", search_time_ms=1
        )

        assert result["outcome"] == "loss"
        assert result["accuracy"] == 100.0
        assert result["stars"] == 1
        assert result["star_cap"] == 1

    def test_win_allows_three_stars(self, monkeypatch):
        engine = FakeEngine()
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
            player_color="w",
            search_time_ms=1,
        )

        assert result["outcome"] == "win"
        assert result["stars"] == 3
        assert result["star_cap"] == 3
        # Последний ход - мат: оценка крайняя, мат 0.
        assert result["plies"][-1]["mate"] == 0
        assert result["plies"][-1]["eval_cp"] == pytest.approx(game_review.MATE_CP)

    def test_draw_caps_stars_at_two(self, monkeypatch):
        engine = FakeEngine()
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            [
                "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5",
                "a5c7", "a6h6", "h2h4", "f7f6", "c7d7", "e8f7",
                "d7b7", "d8d3", "b7b8", "d3h7", "b8c8", "f7g6", "c8e6",
            ],
            player_color="w",
            search_time_ms=1,
        )

        assert result["outcome"] == "draw"
        assert result["stars"] == 2
        assert result["star_cap"] == 2

    def test_limit_truncates_analysis(self, monkeypatch):
        engine = FakeEngine([0, 0])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5", "g1f3", "b8c6"],
            player_color="w",
            search_time_ms=1,
            max_player_moves=1,
        )

        assert result["moves_analyzed"] == 1
        assert result["truncated"] is True

    def test_plies_limit_truncates_eval(self, monkeypatch):
        engine = FakeEngine([0, 0])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(
            ["e2e4", "e7e5", "g1f3"],
            player_color="w",
            search_time_ms=1,
            max_plies=1,
        )

        assert len(result["plies"]) == 3
        assert result["plies"][0]["eval_cp"] is not None
        assert result["plies"][1]["eval_cp"] is None
        assert result["truncated"] is True

    def test_no_player_moves(self, monkeypatch):
        engine = FakeEngine([0])
        monkeypatch.setattr(game_review, "ensure_stockfish", lambda: engine)

        result = game_review.review_game(["e2e4"], player_color="b", search_time_ms=1)

        assert result["moves_analyzed"] == 0
        assert result["accuracy"] is None
        assert result["stars"] == 0
