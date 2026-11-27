import chess
from typing import Dict, Any, List

PIECE_NAMES = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король"
}

def format_piece(piece: chess.Piece) -> str:
    color_str = "белый" if piece.color == chess.WHITE else "черный"
    name = PIECE_NAMES.get(piece.piece_type, "фигура")
    return f"{color_str} {name}"

def extract_position_facts(fen: str, best_move_san: str) -> Dict[str, Any]:
    board = chess.Board(fen)
    
    # 1. Поиск незащищенных и атакуемых фигур
    hanging_pieces = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        color = piece.color
        attackers = list(board.attackers(not color, square))
        defenders = list(board.attackers(color, square))
        sq_name = chess.square_name(square)
        p_desc = f"{format_piece(piece)} на {sq_name}"

        if attackers and not defenders:
            hanging_pieces.append(f"{p_desc} (под боем без защиты)")
        elif not defenders and not attackers:
            pass  # Просто стоит без защиты, но пока не атакована

    # 2. Поиск абсолютных и относительных связок
    pins = []
    for square, piece in board.piece_map().items():
        if piece.piece_type in (chess.KING, chess.PAWN):
            continue
        if board.is_pinned(piece.color, square):
            pins.append(f"{format_piece(piece)} на {chess.square_name(square)} связан")

    # 3. Детальный разбор лучшего хода (best_move)
    best_move_effects = []
    try:
        move_obj = board.parse_san(best_move_san)
        to_sq = chess.square_name(move_obj.to_square)
        
        # Взятие
        if board.is_capture(move_obj):
            captured = board.piece_at(move_obj.to_square)
            if captured:
                best_move_effects.append(f"взятие: {format_piece(captured)} на {to_sq}")
            else:
                best_move_effects.append(f"взятие на проходе на {to_sq}")

        # Превращение пешки
        if move_obj.promotion:
            promo_name = PIECE_NAMES.get(move_obj.promotion, "фигуру")
            best_move_effects.append(f"превращение в {promo_name}")

        # Выполняем ход для проверки нападений
        board.push(move_obj)

        if board.is_check():
            best_move_effects.append("объявлен шах")
            
        if board.is_checkmate():
            best_move_effects.append("мат")

        # Прямые нападения фигуры после того, как она встала на целевое поле
        piece_moved = board.piece_at(move_obj.to_square)
        if piece_moved and piece_moved.piece_type != chess.KING:
            attacks = board.attacks(move_obj.to_square)
            enemy_attacked = []
            for attacked_sq in attacks:
                target_piece = board.piece_at(attacked_sq)
                # Если на поле стоит фигура соперника
                if target_piece and target_piece.color == board.turn:
                    enemy_attacked.append(f"{format_piece(target_piece)} на {chess.square_name(attacked_sq)}")
            
            if enemy_attacked:
                best_move_effects.append(f"нападение на: {', '.join(enemy_attacked)}")

    except Exception:
        pass

    return {
        "hanging_pieces": hanging_pieces,
        "pins": pins,
        "best_move_effects": best_move_effects
    }