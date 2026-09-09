import chess
import chess.engine
from typing import Dict, Any, List, Optional
from loguru import logger
from src.schemas import MultiPVLine, AnalyzedPosition, RawPosition

class StockfishAnalyzer:
    def __init__(self, stockfish_path: str = "/usr/games/stockfish", threads: int = 8, hash_mb: int = 2048):
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine = None
        self._start_engine()

    def _start_engine(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb
            })
        except Exception as e:
            logger.error(f"Не удалось запустить Stockfish по пути '{self.stockfish_path}': {e}")
            raise e

    def analyze_position(self, raw_pos: RawPosition, depth: int = 22, multipv_count: int = 3) -> Optional[AnalyzedPosition]:
        board = chess.Board(raw_pos.fen)
        
        # 1. Анализ MultiPV для поиска альтернатив и лучшего хода
        analysis_info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=multipv_count
        )

        if not analysis_info:
            return None

        multipv_lines: List[MultiPVLine] = []
        best_move_san = ""
        best_eval_cp = None

        for idx, entry in enumerate(analysis_info):
            pv = entry.get("pv", [])
            if not pv:
                continue

            first_move = pv[0]
            move_san = board.san(first_move)
            
            # Собираем PV-линию в SAN нотации
            temp_board = board.copy()
            pv_san_list = []
            for m in pv[:6]:  # До 6 полуходов
                pv_san_list.append(temp_board.san(m))
                temp_board.push(m)

            score = entry.get("score")
            pov_score = score.pov(board.turn)
            eval_cp = pov_score.score(mate_score=10000)
            eval_mate = pov_score.mate()

            if idx == 0:
                best_move_san = move_san
                best_eval_cp = eval_cp

            multipv_lines.append(MultiPVLine(
                rank=idx + 1,
                move_uci=first_move.uci(),
                move_san=move_san,
                eval_cp=eval_cp,
                eval_mate=eval_mate,
                pv_san=pv_san_list
            ))

        # 2. Оценка сыгранного хода
        try:
            played_move_obj = board.parse_san(raw_pos.played_move)
        except Exception:
            played_move_obj = chess.Move.from_uci(raw_pos.played_move)

        played_eval_cp = best_eval_cp
        if raw_pos.played_move != best_move_san:
            board.push(played_move_obj)
            played_info = self.engine.analyse(board, chess.engine.Limit(depth=max(16, depth - 4)))
            board.pop()
            
            played_pov = played_info.get("score").pov(board.turn)
            played_eval_cp = played_pov.score(mate_score=10000)

        cp_loss = max(0, (best_eval_cp or 0) - (played_eval_cp or 0))

        return AnalyzedPosition(
            id=raw_pos.id,
            source_type=raw_pos.source_type,
            game_id=raw_pos.game_id,
            fen=raw_pos.fen,
            played_move=raw_pos.played_move,
            move_number=raw_pos.move_number,
            turn=raw_pos.turn,
            phase=raw_pos.phase,
            best_move=best_move_san,
            played_move_eval_cp=played_eval_cp,
            best_move_eval_cp=best_eval_cp,
            centipawn_loss=cp_loss,
            depth=depth,
            multipv=multipv_lines,
            board_ascii=str(board)
        )

    def close(self):
        if self.engine:
            self.engine.quit()