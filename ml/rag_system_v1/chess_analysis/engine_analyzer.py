import os
import chess
import chess.engine
from typing import Dict, Any, List, Optional

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")

PIECE_NAMES_RU = {
    chess.PAWN: "Пешка",
    chess.KNIGHT: "Конь",
    chess.BISHOP: "Слон",
    chess.ROOK: "Ладья",
    chess.QUEEN: "Ферзь",
    chess.KING: "Король"
}


class ChessEngineAnalyzer:
    """Глубокий анализатор позиции на базе python-chess и Stockfish."""

    def __init__(self, stockfish_path: str = STOCKFISH_PATH, depth: int = 18):
        self.depth = depth
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            print(f"[ERROR] Не удалось подключить Stockfish ({stockfish_path}): {e}")
            self.engine = None

    def close(self):
        if self.engine:
            self.engine.quit()

    @staticmethod
    def detect_stage(board: chess.Board) -> str:
        """Определяет стадию партии строго по материалу, исключая ошибки счетчика FEN."""
        piece_values = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        material = sum(
            len(board.pieces(pt, chess.WHITE)) * val + len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in piece_values.items()
        )
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        
        # Если фигур мало или нет ферзей при небольшом материале — это эндшпиль
        if material <= 14 or (queens == 0 and material <= 20):
            return "endgame"
        if material >= 30 and queens == 2:
            return "opening"
        return "middlegame"

    @staticmethod
    def describe_move_verbally(board: chess.Board, move: chess.Move) -> str:
        """Создает недвусмысленное русскоязычное описание хода для блокировки галлюцинаций."""
        if board.is_castling(move):
            side = "короткую сторону (O-O)" if chess.square_file(move.to_square) == 6 else "длинную сторону (O-O-O)"
            return f"Рокировка в {side}"

        piece = board.piece_at(move.from_square)
        p_name = PIECE_NAMES_RU.get(piece.piece_type, "Фигура") if piece else "Фигура"
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        san = board.san(move)
        is_cap = board.is_capture(move)
        action = "берет на" if is_cap else "идет на"

        return f"{p_name} с {from_sq} {action} {to_sq} ({san})"

    @staticmethod
    def inspect_position_features(board: chess.Board) -> Dict[str, List[str]]:
        features = {
            "white_tactics": [],
            "black_tactics": [],
            "structure": [],
            "king_safety": [],
            "key_squares": []
        }

        # Анализ линий
        for f in range(8):
            w_pawns = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[f])
            b_pawns = len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[f])
            f_name = chess.FILE_NAMES[f]

            if w_pawns == 0 and b_pawns == 0:
                features["structure"].append(f"открытая вертикаль {f_name}")
            elif (w_pawns == 0 and b_pawns > 0) or (w_pawns > 0 and b_pawns == 0):
                features["structure"].append(f"полуоткрытая вертикаль {f_name}")

        # Проходные пешки
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                color = piece.color
                passed = True
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                front_ranks = range(r + 1, 8) if color == chess.WHITE else range(0, r)
                adj_files = [f_idx for f_idx in [f - 1, f, f + 1] if 0 <= f_idx <= 7]

                for fr in front_ranks:
                    for af in adj_files:
                        target_sq = chess.square(af, fr)
                        enemy_p = board.piece_at(target_sq)
                        if enemy_p and enemy_p.piece_type == chess.PAWN and enemy_p.color != color:
                            passed = False
                            break

                if passed:
                    c_name = "белая" if color == chess.WHITE else "черная"
                    features["structure"].append(f"проходная пешка {c_name} на {chess.square_name(sq)}")

        return features

    def evaluate_position(self, fen: str, played_uci_move: Optional[str] = None) -> Dict[str, Any]:
        board = chess.Board(fen)
        turn = board.turn
        turn_str = "Белые" if turn == chess.WHITE else "Черные"
        stage = self.detect_stage(board)
        features = self.inspect_position_features(board)

        # 1. Расчет позиции до хода
        info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        best_move = info_before["pv"][0]
        best_move_verbal = self.describe_move_verbally(board, best_move)
        
        score_before_obj = info_before["score"].white()
        eval_before = score_before_obj.score(mate_score=10000) / 100.0

        # 2. Анализ сделанного хода
        eval_after = eval_before
        eval_loss = 0.0
        played_move_verbal = "Ход не указан (анализ позиции)"
        mistake_type = "анализ исходной позиции"
        rag_query_terms = []

        if played_uci_move:
            played_move = chess.Move.from_uci(played_uci_move)
            if played_move in board.legal_moves:
                played_move_verbal = self.describe_move_verbally(board, played_move)
                board_after = board.copy()
                board_after.push(played_move)

                info_after = self.engine.analyse(board_after, chess.engine.Limit(depth=self.depth))
                eval_after = info_after["score"].white().score(mate_score=10000) / 100.0

                turn_factor = 1.0 if turn == chess.WHITE else -1.0
                eval_loss = max(0.0, (eval_before - eval_after) * turn_factor)

                if eval_loss >= 2.0:
                    mistake_type = "грубый тактический зевок"
                    rag_query_terms.extend(["тактический зевок", "потеря фигуры", "связка", "вилка"])
                elif eval_loss >= 0.5:
                    mistake_type = "позиционная ошибка и уступка инициативы"
                    rag_query_terms.extend(["позиционная ошибка", "активность ладьи", "открытая линия", "форпост"])
                elif eval_loss >= 0.15:
                    mistake_type = "неточность и потеря темпа"
                    rag_query_terms.extend(["потеря темпа", "борьба за центр", "развитие"])
                else:
                    mistake_type = "хороший практический ход"
                    rag_query_terms.extend(["план игры", "активность фигур"])

        # Уточнение запроса для RAG
        if stage == "endgame":
            rooks_count = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
            if rooks_count > 0:
                rag_query_terms.extend(["ладейный эндшпиль", "активность ладьи", "отсечение короля", "правило тарраша"])
            else:
                rag_query_terms.extend(["пешечный эндшпиль", "оппозиция", "ключевые поля", "цугцванг"])

        return {
            "fen": fen,
            "turn": turn_str,
            "stage": stage,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "eval_loss": eval_loss,
            "best_move_uci": best_move.uci(),
            "best_move_verbal": best_move_verbal,
            "played_move_verbal": played_move_verbal,
            "mistake_type": mistake_type,
            "rag_query": " ".join(rag_query_terms) if rag_query_terms else f"шахматная стратегия {stage}"
        }