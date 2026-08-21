"""Утилиты санитизации пользовательского ввода."""

import re

_TAG_RE = re.compile(r"<[^>]*>")

MAX_TEXT_LENGTH = 4000


def sanitize_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Убирает HTML-теги и экранирует спецсимволы.

    Args:
        text: Исходный текст.
        max_length: Максимальная длина результата.

    Returns:
        Очищенный и экранированный текст.
    """
    text = _TAG_RE.sub("", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text[:max_length]
