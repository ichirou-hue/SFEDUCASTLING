# Безопасность и production-конфигурация

Документ описывает исправления по отчёту теста защищённости от 01.11.2026
и обязательные переменные окружения для production.

## Обязательные переменные для production

При `APP_ENV=production` приложение проверяет конфигурацию при старте и
не запускается, если хотя бы одно условие не выполнено. Проверка находится
в `backend/api_gateway/config_guard.py`.

| Переменная | Пример | Зачем |
|---|---|---|
| `APP_ENV` | `production` | Включает строгие проверки и выключает `/docs`, `/redoc`, `/openapi.json`. |
| `DEBUG` | `false` | В production отладка должна быть выключена. |
| `JWT_SECRET_KEY` | случайные 64 символа | Подпись access-токенов. Дефолтное значение запрещено, минимум 32 символа. |
| `OAUTH_COOKIE_SECRET` | другие случайные 64 символа | Отдельная подпись OAuth-кук. Не должен совпадать с `JWT_SECRET_KEY`. |
| `OAUTH_PUBLIC_BASE_URL` | `https://gigachess.ru` | redirect_uri для Lichess/Chess.com. Убирает зависимость от заголовка `Host`. |
| `FRONTEND_BASE_URL` | `https://gigachess.ru` | Адрес возврата после OAuth. |
| `TRUSTED_HOSTS` | `["gigachess.ru","www.gigachess.ru"]` | JSON-массив доверенных значений `Host`. |
| `ALLOWED_ORIGINS` | `["https://gigachess.ru"]` | JSON-массив источников CORS. `["*"]` в production запрещён. |
| `OAUTH_COOKIE_SECURE` | `true` | Cookie OAuth только по HTTPS. |

Сгенерировать секреты можно так:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Пример `.env` для production:

```dotenv
APP_ENV=production
DEBUG=false

JWT_SECRET_KEY=<64 случайных символа>
OAUTH_COOKIE_SECRET=<другие 64 случайных символа>

OAUTH_PUBLIC_BASE_URL=https://gigachess.ru
FRONTEND_BASE_URL=https://gigachess.ru
OAUTH_COOKIE_SECURE=true

TRUSTED_HOSTS=["gigachess.ru","www.gigachess.ru"]
ALLOWED_ORIGINS=["https://gigachess.ru"]
```

В development строгие проверки не блокируют запуск: проблемы пишутся в лог
как `[Security] ...`. Пустой `TRUSTED_HOSTS` в development означает,
что проверка `Host` выключена. В production пустой список запрещён, как и
`TRUSTED_HOSTS=["*"]`, локальные адреса в `ALLOWED_ORIGINS` и
`OAUTH_COOKIE_SECURE=false`.

## Что исправлено

### OAuth и вход через шахматные аккаунты

- `redirect_uri` больше не строится из заголовка `Host`: используется
  `OAUTH_PUBLIC_BASE_URL`, а в development допускается `request.base_url`
  только для доверенного хоста.
- OAuth-куки подписываются отдельным `OAUTH_COOKIE_SECRET`, а не
  `JWT_SECRET_KEY`.
- Кука результата OAuth содержит `token_version`. `POST /api/auth/oauth/complete`
  сверяет её с текущим значением у пользователя, поэтому выход из аккаунта
  аннулирует и старые OAuth-куки.
- При привязке аккаунта callback также сверяет `token_version`, сохранённый
  в OAuth-flow.

### Заголовки безопасности

Все ответы получают:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- `Permissions-Policy`
- `Strict-Transport-Security` при HTTPS
- `Cache-Control: no-store` для `/api/*` и HTML

Политику CSP можно переопределить через `CONTENT_SECURITY_POLICY`.
Отключить заголовки целиком можно через `SECURITY_HEADERS_ENABLED=false`,
но в production так делать не рекомендуется.

### CORS

`backend/app.py` использует `ALLOWED_ORIGINS` из настроек. Раньше там был
жёстко прописан `["*"]`, а настройка игнорировалась. По умолчанию разрешены
только локальные адреса разработки.

### Лимиты частоты

Добавлен in-memory sliding window лимитер (`backend/api_gateway/rate_limit.py`):

- `/api/auth/login`: 5 попыток за 60 секунд;
- `/api/auth/register`, `/api/auth/refresh`, `/api/auth/oauth/complete`:
  20 запросов за 60 секунд;
- `/api/chat/ingest`, `/api/chat/ask`: 30 запросов за 60 секунд;
- дорогие endpoint'ы (`/api/analyze-image`, `/api/parse-pgn`,
  `/api/parse-pgn-text`, `/api/save-move-to-dataset`,
  `/api/auth/oauth/{provider}/start`): 10 запросов за 60 секунд.

Значения меняются через `LOGIN_RATE_LIMIT_*`, `AUTH_RATE_LIMIT_*`,
`CHAT_RATE_LIMIT_*`, `EXPENSIVE_RATE_LIMIT_*`, выключить лимиты можно
через `RATE_LIMIT_ENABLED=false`.

Ограничение работает в памяти одного процесса. Если запущено несколько
воркеров uvicorn или несколько машин, нужен внешний счётчик (например,
Redis). Для запросов через nginx реальный IP берётся из `X-Forwarded-For`,
но только если соединение пришло с частного или loopback-адреса.

Вход и регистрация всегда ограничиваются по IP: случайный заголовок
`Authorization` не создаёт новый счётчик. Остальные лимиты при валидном
access-токене считаются по пользователю, а при невалидном - по IP.

### Чат

- `/api/chat/ingest` требует авторизацию и записывает `user_id` автора.
- `/api/chat/messages` и `/api/chat/messages/count` требуют авторизацию
  и возвращают только сообщения текущего пользователя.
- Раньше записи попадали в общий поток без автора, и любой клиент мог
  прочитать все сообщения.

### Датасет

- `GET /api/dataset/export` доступен только администратору.
- `POST /api/save-move-to-dataset` требует авторизацию. `user_id` из тела
  игнорируется, запись привязывается к текущему пользователю.
- `POST /api/analyze-image` и `/api/parse-pgn` читают файл порциями и
  возвращают 413 при превышении `MAX_UPLOAD_BYTES` (по умолчанию 10 МиБ).

### Логи и лишние данные

- Содержимое PGN больше не печатается в stdout и не попадает в логи.
- `POST /api/learning/puzzle/check` больше не возвращает эталонный ход
  `solution`.
- Неиспользуемая настройка `SECRET_KEY` удалена.
- Swagger и OpenAPI выключены при `APP_ENV=production`.

### SQLite

`INSERT ... ON CONFLICT` в `training_reviews.py` и `dynamic_difficulty.py`
теперь выбирает диалект по фактическому подключению (`backend/db/upsert.py`).
Раньше использовался `pg_insert`, который не поддерживает `constraint=`
на SQLite.

### Достижения

- Разблокировки хранятся в `user_achievements` с уникальной парой
  `(user_id, code)`, поэтому награду нельзя получить дважды.
- Перед начислением сервер повторяет ходы партии и сверяет настоящий
  результат. Клиентские `result` и `status` не считаются доказательством.
- Награды за достижения не уменьшают дневной лимит GigaCoin.
- Каталог доступен в `GET /api/game/achievements`, новые достижения
  возвращаются в ответе `POST /api/game/finish`.

## Проверка после деплоя

```bash
# 1. Приложение должно стартовать без предупреждений [Security].
# 2. Заголовки безопасности
curl -sI https://gigachess.ru/api/health | grep -Ei "content-security-policy|x-frame-options|strict-transport-security"

# 3. Swagger выключен
curl -s -o /dev/null -w "%{http_code}" https://gigachess.ru/docs

# 4. Датасет закрыт для анонимного доступа
curl -s -o /dev/null -w "%{http_code}" https://gigachess.ru/api/dataset/export
```

Ожидаемо: `/docs` возвращает 404, `/api/dataset/export` без токена
возвращает 401.

## Тесты

```bash
python -m pytest tests/security tests/achievements -q
```

Тесты покрывают проверку конфигурации, доверенные хосты, заголовки
безопасности, отдельный секрет OAuth-куки, проверку `token_version`,
лимит частоты, изоляцию сообщений чата, отсутствие `solution` в ответе,
переносимый `ON CONFLICT` на SQLite, а также однократную выдачу
достижений и наград за проверенные партии.
