from typing import Any, Dict, List, Optional
import chess


class BoardFlagsExtractor:
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    PIECE_NAMES_EN = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    @classmethod
    def check_sacrifice(cls, board: chess.Board, move: chess.Move) -> bool:
        piece = board.piece_at(move.from_square)
        if not piece:
            return False

        piece_val = cls.PIECE_VALUES.get(piece.piece_type, 0)
        to_sq = move.to_square
        opp_color = not piece.color

        attackers = board.attackers(opp_color, to_sq)
        for a_sq in attackers:
            att_p = board.piece_at(a_sq)
            if att_p and cls.PIECE_VALUES.get(att_p.piece_type, 0) < piece_val:
                return True

        defenders = board.attackers(piece.color, to_sq)
        if attackers and not defenders and piece_val >= 3:
            return True

        return False

    @classmethod
    def find_pins(cls, board: chess.Board) -> List[str]:
        """Identifies absolute pins (to King) and relative pins (to Queen)."""
        pins_found = []
        turn = board.turn
        opp_turn = not turn

        # 1. Absolute pins (to King)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == turn and board.is_pinned(turn, sq):
                p_name = cls.PIECE_NAMES_EN.get(p.piece_type, "Piece")
                pins_found.append(f"{p_name} on {chess.square_name(sq)} pinned to King")

        # 2. Relative pins (to Queen)
        queens = board.pieces(chess.QUEEN, turn)
        sliding_attackers = (
            board.pieces(chess.BISHOP, opp_turn)
            | board.pieces(chess.ROOK, opp_turn)
            | board.pieces(chess.QUEEN, opp_turn)
        )

        for q_sq in queens:
            for att_sq in sliding_attackers:
                att_piece = board.piece_at(att_sq)
                if not att_piece:
                    continue

                between_bb = chess.between(att_sq, q_sq)
                if not between_bb:
                    continue

                ray_squares = list(chess.SquareSet(between_bb))
                pieces_between = [sq for sq in ray_squares if board.piece_at(sq) is not None]

                if len(pieces_between) == 1:
                    pinned_sq = pieces_between[0]
                    pinned_p = board.piece_at(pinned_sq)
                    if pinned_p and pinned_p.color == turn and pinned_p.piece_type != chess.QUEEN:
                        diff_rank = abs(chess.square_rank(att_sq) - chess.square_rank(q_sq))
                        diff_file = abs(chess.square_file(att_sq) - chess.square_file(q_sq))
                        is_diagonal = diff_rank == diff_file
                        is_orthogonal = (diff_rank == 0) or (diff_file == 0)

                        can_attack = (
                            att_piece.piece_type in [chess.BISHOP, chess.QUEEN] and is_diagonal
                        ) or (
                            att_piece.piece_type in [chess.ROOK, chess.QUEEN] and is_orthogonal
                        )

                        if can_attack:
                            pinned_name = cls.PIECE_NAMES_EN.get(pinned_p.piece_type, "Piece")
                            att_name = cls.PIECE_NAMES_EN.get(att_piece.piece_type, "Piece")
                            pins_found.append(
                                f"{pinned_name} on {chess.square_name(pinned_sq)} pinned to Queen on {chess.square_name(q_sq)} by {att_name} on {chess.square_name(att_sq)}"
                            )

        return list(set(pins_found))

    @classmethod
    def find_hanging_pieces(cls, board: chess.Board) -> List[str]:
        hanging = []
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                opp_c = not p.color
                if board.is_attacked_by(opp_c, sq) and not board.is_attacked_by(p.color, sq):
                    p_name = cls.PIECE_NAMES_EN.get(p.piece_type, "Piece")
                    hanging.append(f"{p_name} on {chess.square_name(sq)}")
        return hanging

    @classmethod
    def extract(cls, board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        legal_moves = list(board.legal_moves)
        is_forced = len(legal_moves) == 1

        from_sq, to_sq = move.from_square, move.to_square
        piece = board.piece_at(from_sq)
        p_type = piece.piece_type if piece else None

        f_rank, f_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        t_rank, t_file = chess.square_rank(to_sq), chess.square_file(to_sq)

        if p_type == chess.KNIGHT:
            m_type = "knight_jump"
        elif p_type == chess.PAWN:
            m_type = "pawn_push"
        elif p_type == chess.KING:
            m_type = "king_step"
        elif f_rank == t_rank or f_file == t_file:
            m_type = "orthogonal"
        else:
            m_type = "diagonal"

        pins_before = set(cls.find_pins(board))

        sim_board = board.copy()
        sim_board.push(move)
        pins_after = set(cls.find_pins(sim_board))
        created_pins = list(pins_after - pins_before)

        center_balance = {}
        for c_sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
            sq_name = chess.square_name(c_sq)
            w_att = len(board.attackers(chess.WHITE, c_sq))
            b_att = len(board.attackers(chess.BLACK, c_sq))
            center_balance[sq_name] = w_att - b_att

        open_files = []
        for f_idx in range(8):
            f_mask = chess.BB_FILES[f_idx]
            pawns = board.pieces_mask(chess.PAWN, chess.WHITE) | board.pieces_mask(chess.PAWN, chess.BLACK)
            if not (f_mask & pawns):
                open_files.append(chess.FILE_NAMES[f_idx])

        return {
            "is_check": board.is_check(),
            "gives_check": board.gives_check(move),
            "is_capture": board.is_capture(move),
            "is_castling": board.is_castling(move),
            "is_forced_move": is_forced,
            "legal_moves_count": len(legal_moves),
            "movement_type": m_type,
            "is_sacrifice": cls.check_sacrifice(board, move),
            "active_pins": list(pins_before),
            "pins_created_by_move": created_pins,
            "hanging_pieces": cls.find_hanging_pieces(board),
            "center_balance": center_balance,
            "open_files": open_files,
        }