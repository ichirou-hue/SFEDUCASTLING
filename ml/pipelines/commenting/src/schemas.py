from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class RawPosition(BaseModel):
    id: str
    source_type: Literal["master_game", "amateur_blunder", "puzzle"]
    game_id: Optional[str] = None
    fen: str
    played_move: str
    move_number: int
    turn: Literal["white", "black"]
    event: Optional[str] = None
    white_player: Optional[str] = None
    black_player: Optional[str] = None
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    phase: Literal["opening", "middlegame", "endgame"]


class MultiPVLine(BaseModel):
    rank: int
    move_uci: str
    move_san: str
    eval_cp: Optional[int] = None
    eval_mate: Optional[int] = None
    pv_san: List[str]


class AnalyzedPosition(BaseModel):
    id: str
    source_type: str
    game_id: Optional[str] = None
    fen: str
    played_move: str
    move_number: int
    turn: str
    phase: str
    best_move: str
    played_move_eval_cp: Optional[int] = None
    best_move_eval_cp: Optional[int] = None
    centipawn_loss: int = 0
    depth: int
    multipv: List[MultiPVLine]
    board_ascii: str


class CoachExplanation(BaseModel):
    """Схема выходных данных от LLM"""
    position_summary: str = Field(description="Краткое описание расстановки сил: кто атакует, где напряжение на доске.")
    root_problem: str = Field(description="Главная проблема или скрытая угроза в позиции.")
    player_mistake: Optional[str] = Field(default=None, description="В чем конкретно заключается ошибка сделанного хода (null, если ход хороший/лучший).")
    best_move: str = Field(description="Лучший ход (строго в нотации SAN, как передал Stockfish).")
    why_best: str = Field(description="Почему этот ход объективно лучший. Объяснение тактики или стратегии.")
    strategic_concept: str = Field(description="Глубокая шахматная идея (концепция), стоящая за этим ходом.")
    mistake_consequences: Optional[str] = Field(default=None, description="К чему приведет ошибка игрока (null, если ошибки не было).")
    main_line: str = Field(description="Словесное объяснение главного варианта (Principal Variation). Как будут развиваться события.")
    practical_advice: str = Field(description="Практический совет: как человеку замечать подобные идеи в своих партиях.")


class GeneratedExample(BaseModel):
    """Полная запись, сохраняемая в data/03_generated/"""
    id: str
    source_type: str
    fen: str
    played_move: str
    best_move: str
    centipawn_loss: int
    phase: str
    turn: str
    coach_explanation: CoachExplanation