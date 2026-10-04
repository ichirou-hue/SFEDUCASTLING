"""Реэкспорт функционала бывшего backend/llm/chess_explainer.py.

Сервис разбит на модули пакета; этот __init__ переэкспортирует все общедоступные
имена, чтобы вызывающий код (backend.api_gateway.routes.explanation) работал
без изменений.
"""

from backend.llm.explainer.constants import (
    GIGACHESS_INPUT_TYPE,
    GIGACHESS_TRANSPORT_ATTEMPTS,
    GIGACHESS_TRANSPORT_BACKOFF_SECONDS,
    MAX_GIGACHESS_ATTEMPTS,
)
from backend.llm.explainer.derived_facts import (
    _build_derived_explanation_facts,
    _line_opening_facts,
)
from backend.llm.explainer.deterministic import _deterministic_explanation
from backend.llm.explainer.grounding import _build_gigachess_grounding
from backend.llm.explainer.move_facts import (
    _join_squares,
    _move_facts,
    _move_info,
    _piece_movement_type,
)
from backend.llm.explainer.names import (
    COLOR_NAMES,
    PIECE_NAMES,
    PIECE_VALUES,
    _capture_object_case_patterns,
    _piece_name,
    _piece_name_accusative,
    _piece_name_capitalized,
    _piece_word_accusative,
)
from backend.llm.explainer.position_facts import _position_change_facts
from backend.llm.explainer.prompts import (
    _build_prompt,
    _build_verified_explanation_context,
    _compact_retry_focus,
    _render_compact_prompt,
)
from backend.llm.explainer.repair import (
    _build_repair_prompt,
    _materialize_protected_capture_promotion_answer,
    _materialize_protected_castling_answer,
    _materialize_protected_en_passant_answer,
    _materialize_protected_promotion_answer,
    _render_protected_capture_promotion_repair_prompt,
    _render_protected_castling_repair_prompt,
    _render_protected_en_passant_repair_prompt,
    _render_protected_promotion_repair_prompt,
    _render_strict_special_repair_prompt,
)
from backend.llm.explainer.service import explain_move
from backend.llm.explainer.stockfish import (
    _normalise_stockfish_result,
    _stockfish_analysis,
)
from backend.llm.explainer.validation import (
    _is_negated_claim,
    _piece_word_present,
    _validate_llm_explanation,
)

__all__ = [
    "COLOR_NAMES",
    "GIGACHESS_INPUT_TYPE",
    "GIGACHESS_TRANSPORT_ATTEMPTS",
    "GIGACHESS_TRANSPORT_BACKOFF_SECONDS",
    "MAX_GIGACHESS_ATTEMPTS",
    "PIECE_NAMES",
    "PIECE_VALUES",
    "_build_derived_explanation_facts",
    "_build_gigachess_grounding",
    "_build_prompt",
    "_build_repair_prompt",
    "_build_verified_explanation_context",
    "_capture_object_case_patterns",
    "_compact_retry_focus",
    "_deterministic_explanation",
    "_is_negated_claim",
    "_join_squares",
    "_line_opening_facts",
    "_materialize_protected_capture_promotion_answer",
    "_materialize_protected_castling_answer",
    "_materialize_protected_en_passant_answer",
    "_materialize_protected_promotion_answer",
    "_move_facts",
    "_move_info",
    "_normalise_stockfish_result",
    "_piece_movement_type",
    "_piece_name",
    "_piece_name_accusative",
    "_piece_name_capitalized",
    "_piece_word_accusative",
    "_piece_word_present",
    "_position_change_facts",
    "_render_compact_prompt",
    "_render_protected_capture_promotion_repair_prompt",
    "_render_protected_castling_repair_prompt",
    "_render_protected_en_passant_repair_prompt",
    "_render_protected_promotion_repair_prompt",
    "_render_strict_special_repair_prompt",
    "_stockfish_analysis",
    "_validate_llm_explanation",
    "explain_move",
]