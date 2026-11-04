"""Общий slowapi-лимитер для тяжёлых и чувствительных endpoint'ов.

Жёсткий лимит частоты — первая линия защиты: он отсекает шквал запросов
до обращения к БД и проверки пароля. Экспоненциальная задержка после серии
неудачных входов живёт отдельно (``backend.services.login_throttle``),
потому что slowapi умеет только фиксированное окно.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# headers_enabled=True → slowapi добавляет Retry-After и X-RateLimit-* в 429,
# чтобы клиент знал, сколько ждать, а не долбил сервер повторно.
limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
