from typing import Dict, Any, List, Optional
import shutil
import os
import chess
import chess.engine


def get_default_stockfish_path() -> str:
    env_path = os.getenv("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    found = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if found:
        return found

    candidates = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish"
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    return "stockfish"


class StockfishAnalyzer:
    def __init__(
        self,
        engine_path: Optional[str] = None,
        depth: int = 14,
        multipv: int = 3,
        threads: int = 4,
        hash_size_mb: int = 256
    ):
        self.engine_path = engine_path or get_default_stockfish_path()
        self.depth = depth
        self.multipv = multipv

        # Инициализируем движок один раз для всего батча
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.engine.configure({
            "Threads": threads,
            "Hash": hash_size_mb
        })

    def close(self):
        """Корректное завершение процесса Stockfish."""
        if hasattr(self, "engine") and self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def analyze(self, board: chess.Board, played_move: chess.Move) -> Dict[str, Any]:
        legal_count = len(list(board.legal_moves))
        actual_multipv = min(self.multipv, max(1, legal_count))

        # Оценка позиции ДО совершения хода
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=actual_multipv
        )

        top_lines: List[Dict[str, Any]] = []
        for line in info:
            pv = line.get("pv", [])
            if not pv:
                continue

            score_pov = line["score"].pov(board.turn)
            score_white = line["score"].white()

            temp_b = board.copy()
            pv_san = []
            for m in pv[:5]:
                pv_san.append(temp_b.san(m))
                temp_b.push(m)

            top_lines.append({
                "move_uci": pv[0].uci(),
                "move_san": board.san(pv[0]),
                "score_cp_pov": score_pov.score(),
                "score_cp_white": score_white.score(),
                "mate_in": score_pov.mate(),
                "pv_san": pv_san
            })

        is_engine_top_move = bool(
            top_lines and top_lines[0]["move_uci"] == played_move.uci()
        )
        played_eval = next(
            (l for l in top_lines if l["move_uci"] == played_move.uci()),
            None
        )

        # Если сделанный ход не в топе MultiPV, оцениваем позицию после него
        if not played_eval:
            temp_b = board.copy()
            temp_b.push(played_move)
            sub_info = self.engine.analyse(
                temp_b,
                chess.engine.Limit(depth=max(8, self.depth - 2))
            )
            sub_score_pov = sub_info["score"].pov(board.turn)
            sub_score_white = sub_info["score"].white()

            played_eval = {
                "move_uci": played_move.uci(),
                "move_san": board.san(played_move),
                "score_cp_pov": sub_score_pov.score(),
                "score_cp_white": sub_score_white.score(),
                "mate_in": sub_score_pov.mate(),
                "pv_san": []
            }

        best_cp = top_lines[0]["score_cp_pov"] if top_lines and top_lines[0]["score_cp_pov"] is not None else 0
        curr_cp = played_eval["score_cp_pov"] if played_eval["score_cp_pov"] is not None else 0
        cp_loss = max(0, best_cp - curr_cp)

        had_mate = bool(top_lines and top_lines[0]["mate_in"] is not None and top_lines[0]["mate_in"] > 0)
        still_has_mate = bool(played_eval["mate_in"] is not None and played_eval["mate_in"] > 0)
        mate_missed = had_mate and not still_has_mate

        return {
            "played_move_eval": played_eval,
            "best_alternative": top_lines[0] if top_lines else None,
            "all_alternatives": top_lines,
            "centipawn_loss": cp_loss,
            "prev_score_cp": best_cp,
            "is_engine_top_move": is_engine_top_move,
            "mate_missed": mate_missed
        }