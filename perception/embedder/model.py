"""
Модуль эмбеддера шахматных позиций для SFEDUCASTLING.

Преобразует FEN-позиции в 512-мерные эмбеддинги для RAG и векторного поиска.
"""

import logging
from typing import Optional

import chess
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fen_to_tensor(fen: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Преобразует FEN в тензор формы (1, 13, 8, 8).

    Каналы:
        0-5   → белые фигуры (P N B R Q K)
        6-11  → чёрные фигуры (p n b r q k)
        12    → чей ход (1.0 = белые)
    """
    if not isinstance(fen, str):
        raise TypeError(f"Ожидалась строка, получено {type(fen)}")

    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Некорректная FEN: {fen}") from e

    tensor = torch.zeros(1, 13, 8, 8, dtype=torch.float32)

    piece_map = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_map[piece.symbol()]
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            tensor[0, channel, rank, file] = 1.0

    tensor[0, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    if device:
        tensor = tensor.to(device)

    return tensor


class ChessEmbedder(nn.Module):
    """CNN эмбеддер позиций (8x8 → 512)."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def embed_fen(
        self, fen: str, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Удобный метод: FEN → embedding"""
        if device is None:
            device = next(self.parameters()).device

        tensor = fen_to_tensor(fen, device)
        self.eval()
        embedding = self(tensor)
        return embedding.squeeze(0)


# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Добавляем корень проекта
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    logging.basicConfig(level=logging.INFO)

    from training_data.collectors.lichess_api import LichessClient

    # Скачиваем 5 партий
    client = LichessClient()
    games = client.fetch_games("alireza2003", max_games=5, perf_type="blitz")

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessEmbedder().to(device)

    print("\n" + "=" * 70)
    print("SFEDUCASTLING · Lichess партии → Embeddings")
    print("=" * 70)

    for i, game in enumerate(games, 1):
        # Lichess API: players.white.user.name
        white = (
            game.get("players", {}).get("white", {}).get("user", {}).get("name", "?")
        )
        black = (
            game.get("players", {}).get("black", {}).get("user", {}).get("name", "?")
        )
        moves = game.get("moves", "").split()

        print(f"\n▸ Партия {i}: {white} vs {black}")
        print(f"  Ходов: {len(moves)}")
        print("  " + "-" * 60)

        board = chess.Board()
        for j, move in enumerate(moves[:10], 1):
            try:
                board.push_san(move)
                emb = model.embed_fen(board.fen(), device)
                print(
                    f"  {j:2d}. {move:5s} → norm={emb.norm():.4f}  mean={emb.mean():+.4f}  std={emb.std():.4f}"
                )
            except ValueError:
                print(f"  {j:2d}. {move:5s} → НЕКОРРЕКТНЫЙ ХОД")

    print("\n" + "=" * 70)
    print(f"Готово · {len(games)} партий обработано")
    print("=" * 70 + "\n")
