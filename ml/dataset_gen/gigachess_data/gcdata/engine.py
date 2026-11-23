"""Оценки Stockfish 16+ и NAG-классификация хода."""
from __future__ import annotations

import math

import chess
import chess.engine
import logging

# «Not transmitting history with null moves» — штатно для поиска угроз нулевым ходом.
logging.getLogger("chess.engine").setLevel(logging.ERROR)

from .motifs import capture_kind, line_info, move_info, static_motifs

MATE_CP = 10000  # мат переводим в большие сантипешки только для CPL/шансов, в тексте он остаётся «мат в N»
CPL_CAP = 1000
THREAT_CP = 150        # тактическая угроза (взятие/шах): второй ход подряд даёт >= 1.5 пешки
QUIET_THREAT_CP = 300  # тихая угроза засчитывается, только если она сильная
PV_KEEP = 12  # сколько полуходов главного варианта сохранять для ключей терминов

# NAG по потере шансов на выигрыш (формула lichess), а не по сырому CPL:
# 150 сантипешек в равной позиции и при +8 — разные по смыслу ошибки.
NAG_TABLE = {
    1: ("!", "сильный ход"),
    2: ("?", "ошибка"),
    4: ("??", "грубая ошибка (зевок)"),
    6: ("?!", "неточность"),
}


def win_chance(cp: int) -> float:
    """[-1, 1] — шансы стороны, для которой дан cp."""
    return 2 / (1 + math.exp(-0.00368208 * cp)) - 1


def _score(info: dict, pov: bool) -> tuple[int | None, int | None, int]:
    """(cp, mate, cp_for_math) с точки зрения pov."""
    s = info["score"].pov(pov)
    if s.is_mate():
        return None, s.mate(), s.score(mate_score=MATE_CP)
    return s.score(), None, s.score()


def classify(wc_drop: float, played_best: bool, only_move_gap: float, forced: bool = False) -> int | None:
    if wc_drop >= 0.30:
        return 4
    if wc_drop >= 0.20:
        return 2
    if wc_drop >= 0.10:
        return 6
    # «!» — единственный ход, который не очевиден: взятие в ответ и уход от шаха не в счёт.
    if played_best and only_move_gap >= 0.25 and not forced:
        return 1
    return None


def verdict_without_nag(cpl: int) -> str:
    return "точный ход" if cpl <= 20 else "приемлемый ход"


class Analyzer:
    def __init__(self, path: str, depth: int = 18, movetime: float | None = None,
                 threads: int = 1, hash_mb: int = 128, threat_depth: int = 12, max_time: float = 20.0):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({k: v for k, v in {"Threads": threads, "Hash": hash_mb}.items()
                               if k in self.engine.options})
        # Потолок по времени: в некоторых позициях (почти пустая доска, форсированный мат)
        # Stockfish не завершает «go depth N» и рабочий процесс висит бесконечно.
        self.limit = chess.engine.Limit(time=movetime) if movetime else chess.engine.Limit(depth=depth, time=max_time)
        # Поиск угроз — вспомогательный, ему хватает меньшей глубины.
        self.threat_limit = chess.engine.Limit(depth=min(threat_depth, depth), time=max_time / 2)
        self.engine_name = self.engine.id.get("name", "unknown")

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _threat(self, board: chess.Board, side: bool, base_math: int, tactical_only: bool) -> dict | None:
        """Угроза стороны side в позиции board, где сейчас ходит соперник: даём side второй ход подряд.

        base_math — оценка позиции board для side при нормальной очерёдности.
        Угрозой считаем выигрыш >= THREAT_CP или мат.
        """
        nb = board.copy(stack=False)
        nb.push(chess.Move.null())
        if nb.is_game_over() or nb.was_into_check():
            return None
        info = self.engine.analyse(nb, self.threat_limit)
        pv = info.get("pv") or []
        if not pv:
            return None
        s = info["score"].pov(side)
        value = s.score(mate_score=MATE_CP)
        mate = s.mate()
        plies, summary = line_info(nb, pv, 3)
        first = plies[0]
        tactical = bool(first["captured"] or first["check"] or first["mate"])
        mates = mate is not None and mate > 0
        gain = value - base_math
        if not (mates or (tactical and gain >= THREAT_CP) or (not tactical_only and gain >= QUIET_THREAT_CP)):
            return None
        return {"move": plies[0], "line": plies, "line_summary": summary,
                "mate_in": mate if mate and mate > 0 else None,
                "gain_cp": int(min(value - base_math, CPL_CAP)), "value_after_threat": int(value)}

    def analyse_move(self, fen_before: str, move_uci: str) -> dict:
        before = chess.Board(fen_before)
        move = chess.Move.from_uci(move_uci)
        mover = before.turn
        after = before.copy(stack=False)
        after.push(move)

        infos = self.engine.analyse(before, self.limit, multipv=2)
        best = infos[0]
        best_move = best["pv"][0]
        cp_b_w, mate_b_w, _ = _score(best, chess.WHITE)
        _, _, best_math = _score(best, mover)
        second_math = _score(infos[1], mover)[2] if len(infos) > 1 else None

        # Позиция после хода.
        if after.is_checkmate():
            cp_a_w, mate_a_w, after_math, pv = None, 0, MATE_CP, []
        elif after.is_stalemate() or after.is_insufficient_material():
            cp_a_w, mate_a_w, after_math, pv = 0, None, 0, []
        else:
            info_a = self.engine.analyse(after, self.limit)
            cp_a_w, mate_a_w, _ = _score(info_a, chess.WHITE)
            after_math = _score(info_a, mover)[2]
            pv = info_a.get("pv", [])

        played_best = move == best_move
        if played_best:
            # Лучший ход по определению: расхождение двух поисков не превращаем в «потерю».
            after_math = max(after_math, best_math)
        cpl = max(0, min(CPL_CAP, best_math - after_math))
        wc_drop = max(0.0, win_chance(best_math) - win_chance(after_math))
        gap = (win_chance(best_math) - win_chance(second_math)) if second_math is not None else 0.0
        nag = classify(wc_drop, played_best, gap, forced=before.is_capture(move) or before.is_check())

        # Мат: CPL упирается в потолок и ничего не говорит, а падение шансов в уже решённой позиции
        # мало. Классифицируем по смыслу: допустил мат / упустил мат.
        mate_b_mover = best["score"].pov(mover).mate()
        mate_a_mover = None if after.is_checkmate() else (
            None if after.is_stalemate() or after.is_insufficient_material() else info_a["score"].pov(mover).mate())
        allows_mate = mate_a_mover is not None and mate_a_mover < 0 and not (mate_b_mover is not None and mate_b_mover < 0)
        misses_mate = mate_b_mover is not None and mate_b_mover > 0 and not after.is_checkmate() and \
            not (mate_a_mover is not None and mate_a_mover > 0)
        if allows_mate:
            nag = 4 if best_math > -700 else 2
        elif misses_mate:
            nag = nag if nag in (2, 4) else 2
        mate_involved = mate_b_mover is not None or mate_a_mover is not None or after.is_checkmate()

        # Угроза хода: что сыграла бы сторона, если бы соперник пропустил ход (нулевой ход).
        own_threat = None
        if not after.is_game_over() and not after.is_check():
            own_threat = self._threat(after, mover, after_math, tactical_only=True)
        # Угроза соперника до хода и что с ней стало.
        opp_threat = None
        if not before.is_check():
            opp_threat = self._threat(before, not mover, -best_math, tactical_only=False)
            if opp_threat:
                # Отражена, если после хода соперник больше не получает того, что давала угроза.
                # Не отражена, если угрожающий ход — всё равно лучший ответ соперника.
                # Для угрозы мата — отражена, если мата больше нет, даже если ход соперника тот же.
                if opp_threat["mate_in"]:
                    opp_threat["parried"] = not (mate_a_mover is not None and mate_a_mover < 0)
                else:
                    still_best = bool(pv) and pv[0].uci() == opp_threat["move"]["uci"]
                    opp_threat["parried"] = not still_best and (-after_math) <= opp_threat["value_after_threat"] - 100

        reply = None
        if pv:
            plies, summary = line_info(after, pv, 3)  # итог — ровно по показанным ходам
            reply = {"line": plies, "summary": summary,
                     "mates": mate_a_mover is not None and mate_a_mover < 0,
                     # Полный вариант — для ключей терминов (gcdata/keys.py): им нужны ходы дальше трёх.
                     "pv_uci": [m.uci() for m in pv[:PV_KEEP]]}

        best_pv = best.get("pv", [best_move])
        best_plies, best_summary = line_info(before, best_pv, 3)

        return {
            "engine": self.engine_name,
            "limit": {"depth": self.limit.depth, "time": self.limit.time},
            "score_before": {"cp": cp_b_w, "mate": mate_b_w},  # с точки зрения белых
            "score_after": {"cp": cp_a_w, "mate": mate_a_w},
            "best_move_before": {**move_info(before, best_move), "line": best_plies, "line_summary": best_summary,
                                 "pv_uci": [m.uci() for m in best_pv[:PV_KEEP]],
                                 "capture_kind": capture_kind(before, best_move),
                                 "mate_in": mate_b_mover if mate_b_mover and mate_b_mover > 0 else None},
            "played_is_best": played_best,
            "mate_involved": mate_involved,
            "allows_mate_in": -mate_a_mover if allows_mate else None,
            "misses_mate_in": mate_b_mover if misses_mate else None,
            "own_threat": own_threat,
            "opp_threat_before": opp_threat,
            "motifs": static_motifs(before, move),
            "cpl": int(cpl),
            "win_chance_drop": round(wc_drop, 4),
            "only_move_gap": round(gap, 4),
            "nag": nag,
            "nag_glyph": NAG_TABLE[nag][0] if nag else None,
            "verdict": NAG_TABLE[nag][1] if nag else verdict_without_nag(int(cpl)),
            "best_reply": reply,
        }
