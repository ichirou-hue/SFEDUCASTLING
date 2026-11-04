"""Разбор завершённой партии: точность, звёзды и оценка каждого хода.

Аналог game review на chess.com, но без хранения: эндпоинт получает
историю ходов (UCI) и сторону игрока, прогоняет позиции через Stockfish
и считает, насколько каждый ход хуже лучшего.

Оценка позиции берётся у движка в сантипешках, мат переводится
в предельную оценку ``±MATE_CP``: точное число ходов до мата на таком
поиске недостоверно, а для шкалы и точности важно только то, что
позиция решена.

Точность считается по модели Lichess. Оценка переводится в вероятность
победы, падение этой вероятности между позицией до хода и после даёт
точность хода. Итоговая точность - среднее двух величин: взвешенного
среднего (вес растёт там, где оценка сильно менялась, то есть в острых
позициях) и среднего гармонического (оно сильно проседает из-за
отдельных грубых ошибок). Одно взвешенное среднее сглаживало бы зевки,
одно гармоническое - наоборот, слишком зависело бы от одного хода.

Ходы из уже решённых позиций не учитываются (то же правило, что
в точности по этапам). Когда партия выиграна или проиграна, вероятность
победы упирается в предел, и любой ход выглядит безупречным: король,
который гулял по полю в проигранной партии, получал 100 % за каждый ход
и вытягивал итог к 70-100 %, хотя партия была потеряна раньше.

Классификация ходов тоже считается по падению вероятности победы,
а не по сантипешкам: потеря 100 cp в равной позиции и потеря 100 cp
в безнадёжной - это разные по значению события.

Звёзды (3/2/1) - пороги по итоговой точности, но результат партии
ограничивает максимум: за победу доступны все три, за ничью - две,
за поражение - одна.

Кроме агрегатов разбор отдаёт ``plies`` - оценку после каждого полухода.
По ним фронтенд рисует историю ходов с подписью и даёт листать партию.
"""

from __future__ import annotations

import math

import chess

from backend.analysis.engine_eval import (
    DEFAULT_SEARCH_MS,
    ensure_single_line,
    position_eval,
)
from backend.analysis.win_probability import (
    DECIDED_WIN_PERCENT,
    MATE_CP,
    classify,
    is_decided,
    move_accuracy,
    player_win_percent,
    win_percent,
)
from backend.api_gateway.state import ensure_stockfish, stockfish_lock

# Псевдонимы для тестов и старого кода: сама реализация живёт в
# backend.analysis.engine_eval.
_position_eval = position_eval
_ensure_single_line = ensure_single_line

# Предел числа разбираемых ходов игрока (для точности и классификации).
MAX_PLAYER_MOVES = 40

# Предел числа оцениваемых полуходов (для истории с оценками). Дальше
# оценки не считаются: разбор не должен висеть минутами на длинной партии.
MAX_PLIES = 120

# Пороги звёзд по итоговой точности (проценты).
STAR_THRESHOLDS: tuple[tuple[int, float], ...] = (
    (3, 85.0),
    (2, 70.0),
    (1, 50.0),
)

# Потолок звёзд по результату партии.
STAR_CAP_BY_OUTCOME: dict[str, int] = {
    "win": 3,
    "draw": 2,
    "loss": 1,
}

# Окно для оценки «остроты» позиции: чем сильнее гуляла оценка рядом,
# тем весомее ход. Как в Lichess, от 2 до 8 полуходов.
_WINDOW_MIN = 2
_WINDOW_MAX = 8
_WEIGHT_MIN = 0.5
_WEIGHT_MAX = 12.0


class EngineUnavailable(RuntimeError):
    """Stockfish не загружен - разбор партии невозможен."""


def stars_for_accuracy(accuracy: float | None) -> int:
    """Сколько звёзд заслуживает партия с такой точностью."""
    if accuracy is None:
        return 0
    for stars, threshold in STAR_THRESHOLDS:
        if accuracy >= threshold:
            return stars
    return 0


def star_cap_for_outcome(outcome: str | None) -> int:
    """Максимум звёзд при таком результате партии."""
    return STAR_CAP_BY_OUTCOME.get(outcome or "", 3)


def game_outcome(board: chess.Board, player_turn: chess.Color) -> str | None:
    """Результат партии для игрока: win / draw / loss или None.

    ``board`` - позиция после последнего хода. На мате очередь хода
    у проигравшей стороны, поэтому сравниваем её со стороной игрока.
    """
    if board.is_checkmate():
        return "loss" if board.turn == player_turn else "win"
    if board.is_game_over(claim_draw=True):
        return "draw"
    return None


def _standard_deviation(values: list[float]) -> float:
    """Выборочное стандартное отклонение; для одного значения - ноль."""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _harmonic_mean(values: list[float]) -> float:
    """Среднее гармоническое: проседает от одного низкого значения."""
    if not values:
        return 0.0

    return len(values) / sum(1.0 / max(value, 0.001) for value in values)


def _position_weights(win_values: list[float]) -> list[float]:
    """Вес каждого полухода по «остроте» позиции, как в Lichess.

    Вес - стандартное отклонение вероятности победы в окне вокруг хода.
    В спокойной позиции оценка почти не меняется, и ход мало что решает;
    в остром положении любая ошибка резко меняет оценку. Вес ограничен
    снизу, чтобы спокойные ходы не выпадали из расчёта совсем, и сверху,
    чтобы один резкий отрезок не перекрыл всю партию.
    """
    count = len(win_values)
    window = min(_WINDOW_MAX, max(_WINDOW_MIN, (count - 1) // 10))

    weights: list[float] = []

    for index in range(max(0, count - 1)):
        start = min(max(0, index - window // 2), max(0, count - window))
        spread = _standard_deviation(win_values[start : start + window])
        weights.append(min(_WEIGHT_MAX, max(_WEIGHT_MIN, spread)))

    return weights


def game_accuracy(
    win_values: list[float],
    player_color: chess.Color,
    decided_win_percent: float = DECIDED_WIN_PERCENT,
) -> float | None:
    """Итоговая точность игрока по оценкам всех полуходов.

    Args:
        win_values: Вероятность победы (с точки зрения белых) по порядку:
            стартовая позиция, затем позиция после каждого полухода.
        player_color: Сторона игрока.
        decided_win_percent: Порог решённой позиции: ходы, в которых
            вероятность победы игрока уже ниже порога (или выше
            ``100 - порог``), не учитываются.

    Returns:
        Точность в процентах или ``None``, если засчитывать нечего.
    """
    weights = _position_weights(win_values)
    weighted: list[tuple[float, float]] = []
    plain: list[float] = []

    for index in range(len(win_values) - 1):
        white_moves = index % 2 == 0

        if white_moves != (player_color == chess.WHITE):
            continue

        # win_values хранятся с точки зрения белых. Для чёрных рост
        # вероятности белых - это потеря, поэтому оценку надо перевернуть:
        # иначе плохой ход чёрных выглядел бы как стопроцентно точный
        # (вероятность белых выросла, а не упала).
        win_before = win_values[index]
        win_after = win_values[index + 1]
        if player_color == chess.BLACK:
            win_before = 100.0 - win_before
            win_after = 100.0 - win_after

        # В решённой позиции вероятность победы упирается в предел (0 или
        # 100 %), и любой ход выглядит безупречным: король, который
        # гуляет по полю в проигранной партии, получал 100 % точности
        # за каждый ход и вытягивал итог к 70-100 %. Такие ходы не
        # оцениваем - качество хода в них ни о чём не говорит.
        if is_decided(win_before, decided_win_percent):
            continue

        accuracy = move_accuracy(win_before, win_after)
        plain.append(accuracy)

        if index < len(weights):
            weighted.append((accuracy, weights[index]))

    if not plain:
        return None

    total_weight = sum(weight for _, weight in weighted)

    if total_weight <= 0:
        return round(sum(plain) / len(plain), 1)

    weighted_mean = sum(accuracy * weight for accuracy, weight in weighted) / total_weight
    harmonic = _harmonic_mean([accuracy for accuracy, _ in weighted])

    return round((weighted_mean + harmonic) / 2.0, 1)


def review_game(
    moves: list[str],
    player_color: str = "w",
    search_time_ms: int = DEFAULT_SEARCH_MS,
    max_player_moves: int = MAX_PLAYER_MOVES,
    max_plies: int = MAX_PLIES,
) -> dict:
    """Разбирает партию и возвращает точность, звёзды и оценки ходов.

    Args:
        moves: Полная история партии в UCI от начальной позиции.
        player_color: Сторона игрока ("w" или "b").
        search_time_ms: Время поиска движка на одну позицию.
        max_player_moves: Сколько ходов игрока разбирать.
        max_plies: Сколько полуходов оценивать для истории.

    Raises:
        EngineUnavailable: Stockfish не загружен.
        ValueError: История содержит недопустимый ход.
    """
    engine = ensure_stockfish()
    if engine is None:
        raise EngineUnavailable("Stockfish недоступен, разбор невозможен")

    player_turn = chess.WHITE if player_color == "w" else chess.BLACK
    sign = 1.0 if player_turn == chess.WHITE else -1.0

    with stockfish_lock:
        _ensure_single_line(engine)

    # Оценка стартовой позиции нужна как «до» для первого хода белых.
    # У чёрных первый ход делает соперник, и «до» для их первого хода
    # придёт из следующего поиска, поэтому старт не ищем зря.
    if player_turn == chess.WHITE:
        prev_cp, _ = _position_eval(engine, chess.Board(), search_time_ms)
    else:
        prev_cp = 0.0
    win_values: list[float] = [win_percent(prev_cp)]

    board = chess.Board()
    plies: list[dict] = []
    player_moves: list[dict] = []
    analyzed = 0
    truncated = False

    for index, token in enumerate(moves, start=1):
        try:
            move = chess.Move.from_uci(token)
        except ValueError as exc:
            raise ValueError(f"Недопустимый ход на позиции {index}: {token}") from exc
        if move not in board.legal_moves:
            raise ValueError(f"Недопустимый ход на позиции {index}: {token}")

        is_player = board.turn == player_turn
        fen_before = board.fen()
        san = board.san(move)
        move_no = board.fullmove_number

        board.push(move)

        entry: dict = {
            "ply": index,
            "move_no": move_no,
            "san": san,
            "uci": token,
            "fen": fen_before,
            "fen_after": board.fen(),
            "by_player": is_player,
            "eval_cp": None,
            "mate": None,
            "loss_cp": None,
            "accuracy": None,
            "classification": None,
            "decided": False,
        }

        if index <= max_plies:
            cp, mate = _position_eval(engine, board, search_time_ms)
            entry["eval_cp"] = round(cp, 1)
            entry["mate"] = mate
            win_values.append(win_percent(cp))

            if is_player and analyzed < max_player_moves:
                # Потеря игрока: падение оценки от его лица. Для чёрных
                # рост оценки белых - это его потеря, поэтому знак.
                loss_cp = max(0.0, (prev_cp - cp) * sign)
                win_before = player_win_percent(prev_cp, player_turn)
                win_after = player_win_percent(cp, player_turn)
                entry["loss_cp"] = round(loss_cp, 1)

                if is_decided(win_before):
                    # Позиция уже решена: ход её не меняет, и оценивать
                    # его качество бессмысленно - в безнадёжном положении
                    # любой ход получает 100 % точности.
                    entry["decided"] = True
                else:
                    accuracy = move_accuracy(win_before, win_after)
                    entry["accuracy"] = round(accuracy, 1)
                    entry["classification"] = classify(win_before, win_after)

                player_moves.append(entry)
                analyzed += 1
            else:
                if is_player:
                    truncated = True

            prev_cp = cp
        else:
            truncated = True
            # Оценки нет, но порядок полуходов сохраняем: при расчёте
            # точности важна чётность индекса.
            win_values.append(win_values[-1])

        plies.append(entry)

    accuracy = game_accuracy(win_values, player_turn)

    scored = [m for m in player_moves if m["classification"] is not None]
    decided = [m for m in player_moves if m["decided"]]

    losses = [m["loss_cp"] for m in scored if m["loss_cp"] is not None]
    average_loss = round(sum(losses) / len(losses), 1) if losses else None

    counts = {
        "best": 0,
        "excellent": 0,
        "good": 0,
        "inaccuracy": 0,
        "mistake": 0,
        "blunder": 0,
    }
    for item in scored:
        classification = item["classification"]
        if classification in counts:
            counts[classification] += 1

    outcome = game_outcome(board, player_turn)
    star_cap = star_cap_for_outcome(outcome)
    stars = min(stars_for_accuracy(accuracy), star_cap)

    return {
        "player_color": "w" if player_turn == chess.WHITE else "b",
        "outcome": outcome,
        "moves_analyzed": len(player_moves),
        "moves_scored": len(scored),
        "moves_decided": len(decided),
        "truncated": truncated,
        "accuracy": accuracy,
        "stars": stars,
        "star_cap": star_cap,
        "thresholds": {str(stars): threshold for stars, threshold in STAR_THRESHOLDS},
        "average_centipawn_loss": average_loss,
        "counts": counts,
        "moves": player_moves,
        "plies": plies,
    }
