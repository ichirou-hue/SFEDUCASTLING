"""Обход бага asyncio на Windows: оборванный клиент убивает сервер.

Проблема
--------
На Windows uvicorn работает на ProactorEventLoop. Клиент может оборвать
соединение раньше, чем сервер его принял: обновление страницы, закрытая
вкладка, отменённый axios-запрос. Тогда AcceptEx завершается ошибкой
WinError 64 "Указанное сетевое имя более недоступно".

В Python 3.14 такая ошибка приходит обычным OSError, а не
ConnectionResetError, поэтому в asyncio/proactor_events.py попадает в
общую ветку ``except OSError``:

    self.call_exception_handler(...)
    sock.close()      # закрывается СЛУШАЮЩИЙ сокет

Дальше процесс жив, uvicorn считает, что сервер работает
(``server.is_serving()`` - True), но принимать новые подключения уже
некому: браузер получает ERR_CONNECTION_REFUSED или запросы висят до
таймаута. Снаружи это выглядит так, будто API "дёргается": открытые
keep-alive соединения ещё отвечают, новые - нет. Лечится только
перезапуском сервера.

Что делает модуль
-----------------
Подменяет ``BaseProactorEventLoop._start_serving`` версией, которая при
ошибке accept не закрывает слушающий сокет, а повторяет accept: сразу,
если клиент просто отвалился, и с небольшой паузой, если ошибка похожа
на нехватку ресурсов. Штатная остановка не меняется: если сокет закрыли
снаружи (``fileno() == -1``) или закрыт сам цикл событий, цикл accept
завершается как раньше.

Отдельно глушится шум в логах: "Accept failed on a socket" и
"Task exception was never retrieved" для WinError 64 - это не ошибка
приложения, а описанный выше обрыв клиента.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# Пауза перед повтором accept после ошибки, похожей на нехватку
# ресурсов (EMFILE и т.п.). Обрыв клиента повторяется сразу.
RETRY_DELAY = 0.05

# Коды "клиент пропал", а не "сервер сломался":
# 64 - ERROR_NETNAME_DELETED, 995 - ERROR_OPERATION_ABORTED,
# 10053/10054 - WSAECONNABORTED/WSAECONNRESET.
_ABORT_WINERRORS = frozenset({64, 995, 10053, 10054})

_NOISY_MESSAGES = (
    "Accept failed on a socket",
    "Task exception was never retrieved",
    "Future exception was never retrieved",
)

_installed = False


def is_connection_abort(exc: BaseException | None) -> bool:
    """Пропал ли клиент (а не сломался ли сервер)."""

    return (
        isinstance(exc, OSError)
        and getattr(exc, "winerror", None) in _ABORT_WINERRORS
    )


def _start_serving(
    self,
    protocol_factory,
    sock,
    sslcontext=None,
    server=None,
    backlog=100,
    ssl_handshake_timeout=None,
    ssl_shutdown_timeout=None,
    **_ignored,
):
    """Копия BaseProactorEventLoop._start_serving без закрытия сокета.

    Отличие от стандартной реализации одно: в ветке ``except OSError``
    слушающий сокет не закрывается, а accept повторяется. Всё остальное,
    включая порядок создания транспорта и учёт фьючерсов, повторяет
    asyncio 3.14.
    """

    from asyncio import exceptions
    from asyncio.trsock import TransportSocket

    def loop(f=None):
        try:
            if f is not None:
                conn, addr = f.result()

                protocol = protocol_factory()

                if server is not None and not server.is_serving():
                    # Сервер закрыли, пока соединение принималось.
                    conn.close()
                    return

                if sslcontext is not None:
                    self._make_ssl_transport(
                        conn,
                        protocol,
                        sslcontext,
                        server_side=True,
                        extra={"peername": addr},
                        server=server,
                        ssl_handshake_timeout=ssl_handshake_timeout,
                        ssl_shutdown_timeout=ssl_shutdown_timeout,
                    )
                else:
                    self._make_socket_transport(
                        conn,
                        protocol,
                        extra={"peername": addr},
                        server=server,
                    )

            if self.is_closed():
                return

            f = self._proactor.accept(sock)

        except OSError as exc:
            if sock.fileno() == -1 or self.is_closed():
                # Штатная остановка сервера.
                return

            if isinstance(
                exc, (ConnectionResetError, ConnectionAbortedError)
            ) or is_connection_abort(exc):
                # Клиент отвалился до accept: слушающий сокет цел.
                # Принимаем следующее соединение сразу - пауза здесь
                # копилась бы и сервер «зависал» на время флуда.
                self.call_soon(loop)
                return

            # Настоящую ошибку (нет дескрипторов и т.п.) показываем,
            # но сокет всё равно не закрываем: иначе сервер умрёт.
            self.call_exception_handler({
                "message": "Accept failed on a socket",
                "exception": exc,
                "socket": TransportSocket(sock),
            })

            try:
                self.call_later(RETRY_DELAY, loop)
            except RuntimeError:
                # Цикл событий уже закрыт.
                return

        except exceptions.CancelledError:
            sock.close()

        else:
            self._accept_futures[sock.fileno()] = f
            f.add_done_callback(loop)

    self.call_soon(loop)


def install() -> bool:
    """Включает защиту. Возвращает True, если патч стоит."""

    global _installed

    if _installed:
        return True

    if os.name != "nt":
        return False

    try:
        from asyncio import proactor_events

        proactor_events.BaseProactorEventLoop._start_serving = _start_serving
    except (ImportError, AttributeError) as exc:
        logger.warning("Защита accept на Windows не включена: %s", exc)
        return False

    _installed = True
    return True


def _quiet_exception_handler(loop, context):
    """Не пугает трейсбеком из-за клиента, который просто закрыл вкладку."""

    message = context.get("message", "")

    if message.startswith(_NOISY_MESSAGES) and is_connection_abort(
        context.get("exception")
    ):
        logger.debug(
            "Клиент оборвал соединение до accept: %s",
            context.get("exception"),
        )
        return

    loop.default_exception_handler(context)


def quiet_loop_noise(loop=None) -> None:
    """Ставит обработчик исключений на работающий цикл событий."""

    if os.name != "nt":
        return

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

    loop.set_exception_handler(_quiet_exception_handler)
