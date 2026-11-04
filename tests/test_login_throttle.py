"""Защита входа от перебора пароля.

Две проверяемые линии:
* ``LoginThrottle`` — экспоненциальная задержка после серии промахов
  (юнит-тесты с подменяемыми часами, без реальных sleep);
* ``POST /api/auth/login`` — 429 + Retry-After, сброс счётчика успешным
  входом и отдельный порог на IP.
"""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.api_gateway.rate_limit import limiter
from backend.api_gateway.routes import auth as auth_routes
from backend.app import app
from backend.db.session import async_session_factory
from backend.models.user import User
from backend.services.login_throttle import LoginThrottle

client = TestClient(app)

PASSWORD = "Sup3rSecret!"


class FakeClock:
    """Управляемые часы: тесты не спят по-настоящему."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------
# Юнит-тесты самого счётчика
# --------------------------------------------------------------------------


def test_no_delay_before_threshold():
    throttle = LoginThrottle(
        after_failures=10, base_delay=1.0, max_delay=300.0, clock=FakeClock()
    )

    for _ in range(9):
        assert throttle.record_failure("key") == 0.0
    assert throttle.retry_after("key") == 0.0


def test_backoff_doubles_at_each_failure_and_is_capped():
    clock = FakeClock()
    throttle = LoginThrottle(
        after_failures=3, base_delay=1.0, max_delay=8.0, clock=clock
    )

    assert throttle.record_failure("k") == 0.0  # 1-й промах
    assert throttle.record_failure("k") == 0.0  # 2-й
    assert throttle.record_failure("k") == 1.0  # 3-й → 1 c

    delays = []
    for _ in range(4):
        clock.advance(10)  # переждали текущую блокировку
        delays.append(throttle.record_failure("k"))

    # 4-й → 2 c, 5-й → 4 c, 6-й и 7-й упираются в потолок 8 c.
    assert delays == [2.0, 4.0, 8.0, 8.0]


def test_retry_after_counts_down_to_zero():
    clock = FakeClock()
    throttle = LoginThrottle(
        after_failures=1, base_delay=4.0, max_delay=60.0, clock=clock
    )

    assert throttle.record_failure("k") == 4.0
    clock.advance(1.5)
    assert throttle.retry_after("k") == pytest.approx(2.5)
    clock.advance(3.0)
    assert throttle.retry_after("k") == 0.0


def test_success_resets_failure_counter():
    clock = FakeClock()
    throttle = LoginThrottle(
        after_failures=3, base_delay=1.0, max_delay=60.0, clock=clock
    )

    for _ in range(3):
        throttle.record_failure("k")
    assert throttle.retry_after("k") == 1.0

    throttle.record_success("k")

    assert throttle.failures("k") == 0
    assert throttle.retry_after("k") == 0.0
    clock.advance(2)
    assert throttle.record_failure("k") == 0.0


def test_old_failures_decay():
    clock = FakeClock()
    throttle = LoginThrottle(
        after_failures=2, base_delay=1.0, max_delay=60.0, ttl=60.0, clock=clock
    )

    throttle.record_failure("k")
    throttle.record_failure("k")
    assert throttle.retry_after("k") == 1.0

    clock.advance(61)
    assert throttle.retry_after("k") == 0.0
    assert throttle.failures("k") == 0
    # После паузы счёт начинается заново, а не с накопленных промахов.
    assert throttle.record_failure("k") == 0.0


def test_many_distinct_keys_do_not_grow_state_unbounded():
    """Перебор разных логинов не должен раздувать память бесконечно."""
    clock = FakeClock()
    throttle = LoginThrottle(
        after_failures=10,
        base_delay=1.0,
        max_delay=60.0,
        ttl=60.0,
        max_keys=10,
        clock=clock,
    )

    for index in range(100):
        clock.advance(0.01)
        throttle.record_failure(f"k{index}")

    # Самые давние ключи вытеснены, свежие — на месте.
    assert throttle.failures("k0") == 0
    assert throttle.failures("k99") == 1
    assert len(throttle._states) <= 10


# --------------------------------------------------------------------------
# Интеграционные тесты endpoint'а
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_limits():
    """Чистое состояние лимитеров на каждый тест."""
    for throttle in (auth_routes._login_throttle, auth_routes._ip_throttle):
        throttle.reset()
    limiter.reset()
    yield
    for throttle in (auth_routes._login_throttle, auth_routes._ip_throttle):
        throttle.reset()
    limiter.reset()


@pytest.fixture()
def clock(monkeypatch):
    fake = FakeClock()
    # Фиксируем политику в тестах, чтобы .env с другими значениями
    # (например, LOGIN_BACKOFF_BASE_SEC) не ломал ожидания.
    monkeypatch.setattr(auth_routes._login_throttle, "after_failures", 10)
    monkeypatch.setattr(auth_routes._login_throttle, "base_delay", 1.0)
    monkeypatch.setattr(auth_routes._login_throttle, "max_delay", 300.0)
    monkeypatch.setattr(auth_routes._ip_throttle, "after_failures", 30)
    monkeypatch.setattr(auth_routes._ip_throttle, "base_delay", 1.0)
    monkeypatch.setattr(auth_routes._ip_throttle, "max_delay", 300.0)
    for throttle in (auth_routes._login_throttle, auth_routes._ip_throttle):
        monkeypatch.setattr(throttle, "_clock", fake)
    return fake


@pytest.fixture()
def cleanup():
    logins: list[str] = []

    def track(login: str) -> str:
        logins.append(login)
        return login

    yield track

    async def _clean():
        if not logins:
            return
        async with async_session_factory() as db:
            await db.execute(delete(User).where(User.login.in_(logins)))
            await db.commit()

    asyncio.run(_clean())


def _register(cleanup) -> str:
    login = cleanup(f"throttle_{uuid.uuid4().hex[:8]}")
    resp = client.post(
        "/api/auth/register", json={"login": login, "password": PASSWORD}
    )
    assert resp.status_code == 201, resp.text
    return login


def _wrong(login: str):
    return client.post(
        "/api/auth/login", json={"login": login, "password": "WrongPass1"}
    )


def _right(login: str):
    return client.post(
        "/api/auth/login", json={"login": login, "password": PASSWORD}
    )


def test_ten_wrong_attempts_start_exponential_backoff(cleanup, clock):
    login = _register(cleanup)

    # Первые 9 промахов — обычный 401 без задержки.
    for _ in range(9):
        assert _wrong(login).status_code == 401

    # 10-й промах включает ожидание 1 секунду.
    tenth = _wrong(login)
    assert tenth.status_code == 429, tenth.text
    assert tenth.headers["Retry-After"] == "1"

    # Мгновенный повтор (даже с верным паролем) — всё ещё 429.
    during = _right(login)
    assert during.status_code == 429
    assert during.headers["Retry-After"] == "1"

    # Переждали 1 секунду и снова промахнулись — уже 2 секунды.
    clock.advance(1.1)
    limiter.reset()  # изолируем slowapi-окно от проверки нашей экспоненты
    eleventh = _wrong(login)
    assert eleventh.status_code == 429
    assert eleventh.headers["Retry-After"] == "2"

    # И следующий промах после паузы — 4 секунды.
    clock.advance(2.1)
    limiter.reset()
    twelfth = _wrong(login)
    assert twelfth.status_code == 429
    assert twelfth.headers["Retry-After"] == "4"


def test_correct_password_works_after_wait(cleanup, clock):
    login = _register(cleanup)

    for _ in range(10):
        _wrong(login)
    assert _right(login).status_code == 429

    clock.advance(1.1)
    limiter.reset()
    ok = _right(login)
    assert ok.status_code == 200, ok.text
    assert ok.json()["access_token"]


def test_success_before_threshold_resets_counter(cleanup, clock):
    login = _register(cleanup)

    for _ in range(5):
        assert _wrong(login).status_code == 401

    assert _right(login).status_code == 200

    # Счётчик логина обнулён: следующие 9 промахов снова дают 401.
    for _ in range(9):
        assert _wrong(login).status_code == 401


def test_per_ip_threshold_blocks_other_logins(cleanup, clock):
    """Перебор разных логинов с одного адреса тоже упирается в лимит."""
    login = _register(cleanup)

    # Наполняем IP-счётчик напрямую: 30 промахов с адреса TestClient.
    host = "ip:testclient"
    for _ in range(auth_routes._ip_throttle.after_failures):
        auth_routes._ip_throttle.record_failure(host)

    blocked = _right(login)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "1"


def test_unknown_login_is_throttled_identically(cleanup, clock):
    """Нет перебора по разнице ответов: неизвестный логин ведёт себя так же."""
    ghost = cleanup(f"ghost_{uuid.uuid4().hex[:8]}")

    for _ in range(9):
        assert _wrong(ghost).status_code == 401

    tenth = _wrong(ghost)
    assert tenth.status_code == 429
    assert tenth.headers["Retry-After"] == "1"
