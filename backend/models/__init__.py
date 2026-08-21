"""SQLAlchemy-модели SFEDUCASTLING.

Пакет собирает модели всех подсистем в одном месте для Alembic
и переиспользования в сервисах.
"""

from backend.models.chat_message import ChatMessage
from backend.models.dataset_move import DatasetMove
from backend.models.game import Game, GameMove

__all__ = ["ChatMessage", "DatasetMove", "Game", "GameMove"]
