"""SQLAlchemy-модели SFEDUCASTLING.

Пакет собирает модели всех подсистем в одном месте для Alembic
и переиспользования в сервисах.
"""

from backend.models.chat_message import ChatMessage
from backend.models.dataset_move import DatasetMove
from backend.models.game import Game, GameMove
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.training_attempt import TrainingAttempt
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.models.user import RefreshToken, User

__all__ = [
    "ChatMessage",
    "DatasetMove",
    "Game",
    "GameMove",
    "LevelTest",
    "PuzzleAttempt",
    "RefreshToken",
    "TrainingAttempt",
    "TrainingLesson",
    "TrainingModule",
    "TrainingTask",
    "User",
]
