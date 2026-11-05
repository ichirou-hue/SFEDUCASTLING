"""Правила смены логина (он же отображаемое имя) и цена в GigaCoin.

Смена меняет и `users.login`, и `users.nickname`: вход после неё выполняется
по новому логину, а прежний логин освобождается для новой регистрации.

Смена платная, цена растёт экспоненциально: 10, 20, 40, 80, ...
Первая смена стоит 10 GigaCoin. Верхняя граница нужна только для того,
чтобы цена оставалась в пределах int32 при большом числе смен.
"""

from __future__ import annotations

NICKNAME_MIN_LENGTH = 3
NICKNAME_MAX_LENGTH = 32
NICKNAME_CHANGE_BASE_PRICE = 10
# 2 ** 20 * 10 ~ 10.5 млн - с запасом помещается в Integer.
NICKNAME_CHANGE_MAX_EXPONENT = 20

# Латиница и кириллица, цифры, пробел и безопасные разделители.
NICKNAME_ALLOWED_SEPARATORS = frozenset(" _.-")


def nickname_change_price(changes: int | None) -> int:
    """Стоимость следующей смены для уже сделанного числа смен."""
    exponent = min(max(int(changes or 0), 0), NICKNAME_CHANGE_MAX_EXPONENT)
    return NICKNAME_CHANGE_BASE_PRICE * (2**exponent)


def normalize_nickname(raw: str) -> str:
    """Убирает лишние пробелы по краям и внутри строки."""
    return " ".join(str(raw or "").split())


def validate_nickname(raw: str) -> str:
    """Проверяет и нормализует новый логин. Бросает ValueError для API."""
    value = normalize_nickname(raw)
    if len(value) < NICKNAME_MIN_LENGTH:
        raise ValueError(f"Логин: минимум {NICKNAME_MIN_LENGTH} символа")
    if len(value) > NICKNAME_MAX_LENGTH:
        raise ValueError(f"Логин: максимум {NICKNAME_MAX_LENGTH} символов")
    if not any(char.isalnum() for char in value):
        raise ValueError("Логин: нужны хотя бы одна буква или цифра")
    if any(not (char.isalnum() or char in NICKNAME_ALLOWED_SEPARATORS) for char in value):
        raise ValueError(
            "Логин: только буквы, цифры, пробел и символы _ - ."
        )
    return value
