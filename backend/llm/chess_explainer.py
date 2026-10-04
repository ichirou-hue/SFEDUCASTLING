"""Фасад для реэкспорта функций, ранее живших в этом монолите.

Весь код перенесён без изменений в пакет backend.llm.explainer.
Этот файл сохранён для обратной совместимости импортов:
    from backend.llm.chess_explainer import explain_move
"""

from backend.llm.explainer import *  # noqa: F401,F403
from backend.llm.explainer import __all__  # noqa: F401