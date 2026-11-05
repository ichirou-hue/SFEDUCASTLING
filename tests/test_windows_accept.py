"""Тесты защиты accept на Windows (backend/api_gateway/windows_accept.py).

WinError 64 во время accept - это клиент, закрывший соединение раньше,
чем сервер успел его принять. Раньше asyncio в этом случае закрывал
слушающий сокет, и сервер переставал принимать подключения до
перезапуска.
"""

import pytest

from backend.api_gateway import windows_accept


class _Sock:
    def __init__(self, fd=7):
        self._fd = fd

    def fileno(self):
        return self._fd

    def close(self):
        self._fd = -1


class _Future:
    def __init__(self):
        self._error = None
        self._result = None
        self._callbacks = []

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def add_done_callback(self, callback):
        self._callbacks.append(callback)

    def fail(self, error):
        self._error = error
        for callback in self._callbacks:
            callback(self)


class _Proactor:
    def __init__(self):
        self.accepts = 0
        self.pending = None

    def accept(self, sock):
        self.accepts += 1
        self.pending = _Future()
        return self.pending


class _Loop:
    def __init__(self):
        self._proactor = _Proactor()
        self._accept_futures = {}
        self.scheduled = []
        self.delayed = []
        self.exceptions = []
        self.transports = []
        self.closed = False

    def call_soon(self, callback, *args):
        self.scheduled.append((callback, args))

    def call_later(self, delay, callback, *args):
        self.delayed.append((delay, callback, args))

    def is_closed(self):
        return self.closed

    def call_exception_handler(self, context):
        self.exceptions.append(context)

    def _make_socket_transport(self, *args, **kwargs):
        self.transports.append((args, kwargs))

    def _make_ssl_transport(self, *args, **kwargs):
        raise AssertionError("тест без TLS")


def _start(loop, sock):
    """Запускает подменённый _start_serving и первый шаг accept."""

    windows_accept._start_serving(loop, lambda: object(), sock)

    callback, args = loop.scheduled.pop(0)
    callback(*args)


def _abort_error(winerror=64):
    # WinError 64 приходит обычным OSError: errno 22, winerror 64.
    # На POSIX четвёртый аргумент OSError игнорируется, поэтому атрибут
    # winerror выставляем явно: CI гоняет эти тесты на Linux.
    error = OSError(22, "netname deleted")
    error.winerror = winerror
    return error


class TestAbortDoesNotKillListener:

    def test_accept_error_keeps_listener_open(self):
        loop = _Loop()
        sock = _Sock()

        _start(loop, sock)
        assert loop._proactor.accepts == 1

        loop._proactor.pending.fail(_abort_error())

        # Главное: слушающий сокет жив, accept поставлен заново.
        assert sock.fileno() != -1
        assert loop.scheduled, "accept должен быть возобновлён"

        # Обрыв повторяется сразу, без паузы: иначе флуд оборванных
        # соединений превращался в секунды недоступности сервера.
        callback, args = loop.scheduled.pop(0)
        callback(*args)

        assert loop._proactor.accepts == 2
        assert loop.delayed == []

    def test_abort_is_not_reported_as_error(self):
        loop = _Loop()
        sock = _Sock()

        _start(loop, sock)
        loop._proactor.pending.fail(_abort_error())

        # Обрыв клиента - не ошибка сервера, шума в логах быть не должно.
        assert loop.exceptions == []

    def test_other_error_is_reported_but_listener_survives(self):
        loop = _Loop()
        sock = _Sock()

        _start(loop, sock)
        loop._proactor.pending.fail(OSError(24, "too many open files"))

        assert len(loop.exceptions) == 1
        assert loop.exceptions[0]["message"] == "Accept failed on a socket"
        # Даже настоящую ошибку не превращаем в мёртвый сервер.
        assert sock.fileno() != -1

        # А вот тут пауза нужна: иначе на EMFILE будет горячий цикл.
        assert loop.delayed
        delay, retry, args = loop.delayed.pop(0)
        assert delay == windows_accept.RETRY_DELAY
        retry(*args)
        assert loop._proactor.accepts == 2

    def test_closed_listener_stops_the_loop(self):
        loop = _Loop()
        sock = _Sock()
        sock.close()

        _start(loop, sock)
        loop._proactor.pending.fail(_abort_error())

        # Сокет закрыли снаружи (остановка сервера) - повтор не нужен.
        assert loop.scheduled == []
        assert loop.delayed == []

    def test_closed_event_loop_stops_the_loop(self):
        loop = _Loop()
        sock = _Sock()

        _start(loop, sock)
        loop.scheduled.clear()
        loop.closed = True
        loop._proactor.pending.fail(_abort_error())

        assert loop.scheduled == []
        assert loop.delayed == []


class TestAbortDetection:

    @pytest.mark.parametrize("winerror", [64, 995, 10053, 10054])
    def test_connection_abort_codes(self, winerror):
        error = OSError(22, "abort")
        error.winerror = winerror
        assert windows_accept.is_connection_abort(error) is True

    def test_other_errors_are_not_aborts(self):
        assert windows_accept.is_connection_abort(OSError(24, "emfile")) is False
        assert windows_accept.is_connection_abort(None) is False
        assert windows_accept.is_connection_abort(RuntimeError("x")) is False


class TestQuietHandler:

    class _Loop:
        def __init__(self):
            self.defaulted = []

        def default_exception_handler(self, context):
            self.defaulted.append(context)

    def test_client_abort_is_silent(self):
        loop = self._Loop()
        windows_accept._quiet_exception_handler(loop, {
            "message": "Accept failed on a socket",
            "exception": _abort_error(),
        })

        assert loop.defaulted == []

    def test_task_noise_is_silent(self):
        loop = self._Loop()
        windows_accept._quiet_exception_handler(loop, {
            "message": "Task exception was never retrieved",
            "exception": _abort_error(),
        })

        assert loop.defaulted == []

    def test_real_error_goes_to_default_handler(self):
        loop = self._Loop()
        windows_accept._quiet_exception_handler(loop, {
            "message": "Task exception was never retrieved",
            "exception": RuntimeError("настоящая ошибка"),
        })

        assert len(loop.defaulted) == 1


class TestInstall:

    def test_install_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(windows_accept, "_installed", False)

        if windows_accept.os.name != "nt":
            assert windows_accept.install() is False
            return

        assert windows_accept.install() is True
        assert windows_accept.install() is True

    def test_install_skips_non_windows(self, monkeypatch):
        monkeypatch.setattr(windows_accept.os, "name", "posix")
        monkeypatch.setattr(windows_accept, "_installed", False)

        assert windows_accept.install() is False
