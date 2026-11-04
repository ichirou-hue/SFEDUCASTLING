"""Экспоненциальная задержка после серии неудачных попыток входа.

Задача — не дать перебирать пароль даже при высокой частоте запросов.
После ``after_failures`` промахов включается ожидание:

    1 c, 2 c, 4 c, 8 c, ... (но не больше ``max_delay``)

Сервер при этом НЕ спит: запрос сразу отклоняется с ``429`` и заголовком
``Retry-After``, а клиент ждёт столько, сколько ему разрешено. Именно такой
контракт позволяет держать один рабочий процесс без подвисших корутин.

Состояние хранится в памяти процесса. Для одного инстанса этого достаточно;
при горизонтальном масштабировании счётчики нужно вынести в общий store
(Redis), иначе у каждого процесса будет свой лимит.

Часы подменяются через ``clock`` — тесты двигают время вручную и не спят.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _AttemptState:
    """Счётчик промахов для одного ключа (IP или IP+логин)."""

    failures: int = 0
    last_failure: float = 0.0
    blocked_until: float = 0.0


class LoginThrottle:
    """Счётчик неудачных входов с экспоненциальной задержкой."""

    def __init__(
        self,
        *,
        after_failures: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        ttl: float = 900.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.after_failures = max(1, int(after_failures))
        self.base_delay = max(0.0, float(base_delay))
        self.max_delay = max(self.base_delay, float(max_delay))
        self.ttl = max(0.0, float(ttl))
        self.max_keys = max(1, int(max_keys))
        self._clock = clock
        self._states: dict[str, _AttemptState] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Не даёт словарю расти бесконечно при переборе разных логинов.

        Вызывается под локом. Сначала убираем записи старше TTL, затем —
        если всё ещё переполнено — самые давние по последнему промаху.
        """
        if len(self._states) < self.max_keys:
            return
        stale = [
            key
            for key, state in self._states.items()
            if now - state.last_failure > self.ttl
        ]
        for key in stale:
            del self._states[key]
        overflow = len(self._states) - self.max_keys + 1
        if overflow > 0:
            # Удаляем пакет, а не одну запись: иначе полный проход по словарю
            # выполнялся бы на каждом новом ключе и сам стал бы DoS-вектором.
            batch = max(overflow, max(1, self.max_keys // 10))
            oldest = sorted(
                self._states.items(), key=lambda item: item[1].last_failure
            )
            for key, _ in oldest[:batch]:
                del self._states[key]

    def _fresh_state(self, key: str, now: float) -> _AttemptState:
        """Возвращает состояние ключа, сбрасывая совсем старые промахи."""
        state = self._states.get(key)
        if state is None:
            state = _AttemptState()
            self._states[key] = state
            return state
        if state.failures and now - state.last_failure > self.ttl:
            state.failures = 0
            state.blocked_until = 0.0
        return state

    @staticmethod
    def _delay_for(failures: int, after: int, base: float, cap: float) -> float:
        """Задержка для текущего числа промахов (0 — блокировки ещё нет)."""
        if failures < after:
            return 0.0
        exponent = failures - after
        # Ограничиваем степень до возведения в неё: иначе 2 ** 10000
        # не помещается во float. Дальше всё равно действует потолок.
        if exponent >= 32:
            return cap
        return min(cap, base * (2 ** exponent))

    def retry_after(self, key: str) -> float:
        """Сколько секунд осталось до конца текущей блокировки (0 — можно)."""
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return 0.0
            # Давно не было промахов: запись устарела и больше не нужна.
            if state.failures and now - state.last_failure > self.ttl:
                del self._states[key]
                return 0.0
            return max(0.0, state.blocked_until - now)

    def record_failure(self, key: str) -> float:
        """Учитывает промах.

        Возвращает задержку в секундах: ``0``, пока порог не достигнут,
        затем ``base * 2 ** (failures - after)`` с потолком ``max_delay``.
        """
        now = self._clock()
        with self._lock:
            self._prune(now)
            state = self._fresh_state(key, now)
            state.failures += 1
            state.last_failure = now
            delay = self._delay_for(
                state.failures,
                self.after_failures,
                self.base_delay,
                self.max_delay,
            )
            if delay:
                state.blocked_until = now + delay
            return delay

    def record_success(self, key: str) -> None:
        """Успешный вход обнуляет счётчик именно этого ключа."""
        with self._lock:
            self._states.pop(key, None)

    def failures(self, key: str) -> int:
        """Текущее число промахов — для диагностики и тестов."""
        with self._lock:
            state = self._states.get(key)
            return state.failures if state else 0

    def reset(self) -> None:
        """Полный сброс состояния (тесты, перезапуск политики)."""
        with self._lock:
            self._states.clear()
